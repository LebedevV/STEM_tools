#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Per-seed worker.

Claims one ``seed_NNNNNN.todo`` as ``.running``, computes that phonon snapshot,
writes ``outputs/seed_NNNNNN_<channel>.{zarr,tif}``, and marks the seed
``.done`` on success.
"""
import argparse
import logging
import shutil
import signal
import sys
import threading
from pathlib import Path

import abtem
import ase.io
import numpy as np

from ._log import configure_default_logging
from .compat import warn_if_unpatched
from .job_io import load_job_config, seed_from_path
from .pipeline import cpu_fft_backend, make_potential, resolve_context
from .simulation import add_probe, add_scan, load_ground_state_atoms


log = logging.getLogger(__name__)


def _cleanup_seed_outputs(out_dir: Path, seed: int) -> None:
	"""Remove partial outputs for one seed."""
	if not out_dir.exists():
		return
	for p in out_dir.glob(f"seed_{seed:06d}_*"):
		try:
			if p.is_dir() and not p.is_symlink():
				shutil.rmtree(p)
			else:
				p.unlink()
		except OSError:
			pass


def _install_preemption_handler(
	out_dir: Path,
	seed: int,
	*,
	todo_path: Path | None = None,
	running_path: Path | None = None,
):
	"""Install signal cleanup and return a handler-restoration callable."""
	if threading.current_thread() is not threading.main_thread():
		return lambda: None

	def _handler(signum, frame):
		log.warning(
			f"worker: received signal {signum}, removing partial outputs "
			f"for seed {seed} in {out_dir}"
		)
		_cleanup_seed_outputs(out_dir, seed)
		if todo_path is not None and running_path is not None:
			try:
				if running_path.exists() and not todo_path.exists():
					running_path.rename(todo_path)
			except OSError:
				pass
		# Re-raise with the original signal semantics.
		signal.signal(signum, signal.SIG_DFL)
		signal.raise_signal(signum)

	prev_term = signal.signal(signal.SIGTERM, _handler)
	prev_int = signal.signal(signal.SIGINT, _handler)

	def restore():
		signal.signal(signal.SIGTERM, prev_term)
		signal.signal(signal.SIGINT, prev_int)

	return restore


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #


def _displaced_atoms(atoms, sigmas, seed: int):
	"""Per-seed phonon displacement, or atoms unchanged for σ ≤ 0 / None / False.

	Uses ``FrozenPhonons(num_configs=1, sigmas=σ, seed=seed)`` and pulls the
	first (only) trajectory snapshot. This matches the abtem boundary
	exercised by the existing reproducibility tests — same seed produces
	bit-identical positions.
	"""
	if sigmas is None or sigmas is False or sigmas == 0:
		return atoms.copy()
	fph = abtem.FrozenPhonons(
		atoms, num_configs=1, sigmas=float(sigmas), seed=int(seed)
	)
	return fph.to_atoms_ensemble().trajectory[0]


def run_scan(cfg, ctx, potential, detector_objs):
	"""Probe.scan over the lamella with the requested detectors.

	Returns a list ``[per-detector measurement]`` in the same order as
	``detector_objs``. Normalises the single-detector case (abtem returns
	a single ensemble there, not a 1-element list).
	"""
	# Probe and scan construction is shared with projection previews.
	probe = add_probe(cfg, potential)
	scan = add_scan(cfg, ctx, probe, potential)

	raw = probe.scan(potential, scan=scan, detectors=detector_objs).compute()
	if len(detector_objs) == 1:
		return [raw]
	return list(raw)


def run_diffraction(cfg, potential):
	"""Plane-wave multislice → diffraction patterns for one seed."""
	try:
		with abtem.config.set({"device": "cpu", "fft": cpu_fft_backend()}):
			pw = abtem.PlaneWave(energy=cfg.microscope.HT_value, device="cpu")
			exit_waves = pw.multislice(potential.to_cpu()).compute()
			return exit_waves.diffraction_patterns(
				max_angle="valid", block_direct=True
			).compute().to_cpu()
	except Exception:
		pw = abtem.PlaneWave(energy=cfg.microscope.HT_value)
		exit_waves = pw.multislice(potential).compute()
		return exit_waves.diffraction_patterns(
			max_angle="valid", block_direct=True
		).compute().to_cpu()


def run_cbed(cfg, ctx, potential):
	"""Probe-at-center CBED for one seed."""
	center = np.array([[
		0.5 * (ctx.scan_start[0] + ctx.scan_stop[0]),
		0.5 * (ctx.scan_start[1] + ctx.scan_stop[1]),
	]], dtype=float)

	try:
		with abtem.config.set({"device": "cpu", "fft": cpu_fft_backend()}):
			cpu_potential = potential.to_cpu()
			probe = add_probe(cfg, cpu_potential)
			exit_waves = probe.multislice(cpu_potential, scan=center).compute()
			cbed = exit_waves.diffraction_patterns(
				max_angle=cfg.microscope.cbed_max_angle, block_direct=False
			)
			if cbed.ensemble_dims > 0:
				cbed = cbed.reduce_ensemble()
			return cbed.compute().squeeze().to_cpu()
	except Exception:
		probe = add_probe(cfg, potential)
		exit_waves = probe.multislice(potential, scan=center).compute()
		cbed = exit_waves.diffraction_patterns(
			max_angle=cfg.microscope.cbed_max_angle, block_direct=False
		)
		if cbed.ensemble_dims > 0:
			cbed = cbed.reduce_ensemble()
		return cbed.compute().squeeze().to_cpu()


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def run_one_seed(job_dir, todo_path) -> None:
	"""Process one .todo file: build lamella, displace per seed, run multislice,
	write outputs, rename .todo → .done.

	Args:
		job_dir: path to the job directory
		           (``gen_*/<phase>_<hkl>_<tilt>/``).
		todo_path: path to one ``seeds/seed_NNNNNN.todo`` file inside that
		           job_dir.

	Side effects:
		- Writes ``outputs/seed_NNNNNN_potproj.{zarr,tif}`` (projection of
		  this seed's potential, always) + ``_<channel>.{zarr,tif}`` for each
		  scan detector if ``do_full_run`` + ``_diff.{zarr,tif}`` if
		  ``do_diffraction`` + ``_cbed.{zarr,tif}`` if ``do_cbed`` +
		  ``_displaced.xyz`` if ``test_enabled``.
		- Renames ``seeds/seed_NNNNNN.running`` → ``seeds/seed_NNNNNN.done``.

	On ordinary exception: the claim is rolled back to ``.todo`` so a retry can
	pick up the same work. The aggregator will see only completed seeds.

	On SIGTERM / SIGINT: partial seed_NNNNNN_* outputs are removed and the
	claim is rolled back to ``.todo`` on a best-effort basis.
	"""
	job_dir = Path(job_dir).resolve()
	todo_path = Path(todo_path).resolve()

	seed = seed_from_path(todo_path)
	running_path = todo_path.with_suffix(".running")
	done_path = todo_path.with_suffix(".done")
	try:
		todo_path.rename(running_path)
	except FileNotFoundError as e:
		if running_path.exists():
			raise FileExistsError(f"seed already claimed: {running_path}") from e
		if done_path.exists():
			raise FileExistsError(f"seed already completed: {done_path}") from e
		raise

	try:
		_, cfg = load_job_config(job_dir)
		ctx = resolve_context(cfg)

		out_dir = job_dir / "outputs"
		out_dir.mkdir(parents=True, exist_ok=True)

		# Signals clean partial outputs and requeue the claim when Python can run cleanup.
		restore_handlers = _install_preemption_handler(
			out_dir, seed, todo_path=todo_path, running_path=running_path,
		)
		try:
			lamella = load_ground_state_atoms(job_dir, cfg)
			displaced = _displaced_atoms(lamella, ctx.fph_sigma, seed)

			# Test mode leaves the displacement inspectable even if multislice fails.
			if cfg.simulations.test_enabled:
				ase.io.write(
					str(out_dir / f"seed_{seed:06d}_displaced.xyz"),
					displaced,
					"xyz",
				)

			potential = make_potential(displaced).build().compute()

			# Always write the projected potential used by this seed.
			proj = potential.project().to_cpu().compute()
			proj.to_tiff(str(out_dir / f"seed_{seed:06d}_potproj.tif"))
			proj.to_zarr(str(out_dir / f"seed_{seed:06d}_potproj.zarr"), overwrite=True)

			if cfg.microscope.do_diffraction:
				diff = run_diffraction(cfg, potential)
				diff.to_tiff(str(out_dir / f"seed_{seed:06d}_diff.tif"))
				diff.to_zarr(str(out_dir / f"seed_{seed:06d}_diff.zarr"), overwrite=True)

			if cfg.microscope.do_cbed:
				cbed = run_cbed(cfg, ctx, potential)
				cbed.to_tiff(str(out_dir / f"seed_{seed:06d}_cbed.tif"))
				cbed.to_zarr(str(out_dir / f"seed_{seed:06d}_cbed.zarr"), overwrite=True)

			if cfg.simulations.do_full_run and cfg.microscope.detectors:
				detectors_by_name = {
					"haadf": ctx.haadf_detector,
					"abf": ctx.abf_detector,
					"bf": ctx.bf_detector,
				}
				detector_objs = [detectors_by_name[n] for n in cfg.microscope.detectors]
				measurements = run_scan(cfg, ctx, potential, detector_objs)
				for det_name, m in zip(cfg.microscope.detectors, measurements):
					cpu = m.copy().to_cpu()
					cpu.to_tiff(str(out_dir / f"seed_{seed:06d}_{det_name}.tif"))
					cpu.to_zarr(str(out_dir / f"seed_{seed:06d}_{det_name}.zarr"), overwrite=True)

		finally:
			restore_handlers()

	except BaseException:
		try:
			if running_path.exists() and not todo_path.exists():
				running_path.rename(todo_path)
		except OSError:
			pass
		raise

	# Mark done after the signal handler is restored; completed outputs are no longer reclaimable.
	running_path.rename(done_path)


def main():
	"""Module entry point for ``python -m abtem_run.worker``."""
	configure_default_logging()
	warn_if_unpatched()
	parser = argparse.ArgumentParser(
		description=(
			"abtem_run worker: claim one .todo as .running, write per-seed "
			"outputs to <job_dir>/outputs/, and rename the claim to .done."
		),
	)
	parser.add_argument("job_dir", help="job directory (gen_*/<phase>_<hkl>_<tilt>/)")
	parser.add_argument("todo_path", help="path to one seeds/seed_NNNNNN.todo file")
	args = parser.parse_args()
	run_one_seed(args.job_dir, args.todo_path)
	return 0


if __name__ == "__main__":
	sys.exit(main())
