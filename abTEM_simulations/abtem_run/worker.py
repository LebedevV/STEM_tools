#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Push-mode per-seed worker for the abtem-run worker pipeline.

Consumes ONE ``seeds/seed_NNNNNN.todo`` file in a job directory, atomically
claims it as ``seed_NNNNNN.running``, runs the multislice for that single
phonon snapshot, writes outputs to ``outputs/seed_NNNNNN_<channel>.{zarr,tif}``,
and renames the claim to ``.done`` on success.

Push interface (the worker is stateless; the caller — bash loop, GNU
parallel, slurm, or the convenience wrapper — supplies the .todo path):

    python -m abtem_run.worker <job_dir> <todo_path>

Library entry:

    from abtem_run.worker import run_one_seed
    run_one_seed(job_dir, todo_path)

Every seed goes through the same code path — there is no special
"static lattice" branch. To produce a static-lattice result, configure
``fph_sigma = false`` and ``frozen_phonons = 1``; the worker will run
with zero displacement.
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
from .job_io import load_job_config, seed_from_path
from .pipeline import make_potential, resolve_context
from .simulation import add_probe, add_scan, load_ground_state_atoms


log = logging.getLogger(__name__)


def _cleanup_seed_outputs(out_dir: Path, seed: int) -> None:
	"""Remove this seed's partial outputs/seed_NNNNNN_* files. Used on a
	signal (SIGTERM from spot preemption / SIGINT from ^C): a half-written
	zarr would otherwise be picked up by the aggregator's seed_*_<ch>.zarr
	glob and silently contaminate the cross-seed mean.
	"""
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
	"""Trap SIGTERM (spot 2-minute warning) and SIGINT (^C) so we clean
	the partial seed outputs before exiting. Returns a callable that
	restores the previous handlers. No-op when not on the main thread
	(``signal.signal`` is main-thread-only)."""
	if threading.current_thread() is not threading.main_thread():
		return lambda: None

	def _handler(signum, frame):
		log.warning(
			f"worker: received signal {signum}, removing partial outputs "
			f"for seed {seed} in {out_dir}"
		)
		_cleanup_seed_outputs(out_dir, seed)
		if todo_path is not None and running_path is not None:
			_requeue_running(todo_path, running_path)
		# Restore default + re-raise so the process exits with the right
		# status code (SIGTERM => 143, SIGINT => 130) and any parent
		# orchestrator sees a clean signal-termination.
		signal.signal(signum, signal.SIG_DFL)
		signal.raise_signal(signum)

	prev_term = signal.signal(signal.SIGTERM, _handler)
	prev_int = signal.signal(signal.SIGINT, _handler)

	def restore():
		signal.signal(signal.SIGTERM, prev_term)
		signal.signal(signal.SIGINT, prev_int)

	return restore


# --------------------------------------------------------------------------- #
# Building blocks — factored so an in-memory wrapper can compose them without I/O.
# --------------------------------------------------------------------------- #


def _claim_todo(todo_path: Path) -> Path:
	"""Atomically claim ``seed_NNNNNN.todo`` as ``seed_NNNNNN.running``.

	This prevents two parallel workers from accidentally processing the same
	seed. A missing ``.todo`` with an existing ``.running`` or ``.done`` is
	reported as an already-claimed/already-finished seed rather than silently
	starting duplicate work.
	"""
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
	return running_path


def _requeue_running(todo_path: Path, running_path: Path) -> None:
	"""Best-effort rollback from ``.running`` to ``.todo`` after failure.

	If the process is killed too hard for Python cleanup to run, the ``.running``
	file is intentionally left behind as a visible stale claim that can be
	requeued manually after checking the worker is gone.
	"""
	try:
		if running_path.exists() and not todo_path.exists():
			running_path.rename(todo_path)
	except OSError:
		pass


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


def _detector_objects(ctx, names):
	"""Map ``["haadf", "abf", "bf"]`` → the AnnularDetector instances on ctx.

	Order is preserved so the output filenames match the user's intent.
	"""
	table = {
		"haadf": ctx.haadf_detector,
		"abf": ctx.abf_detector,
		"bf": ctx.bf_detector,
	}
	return [table[n] for n in names]


def run_scan(ctx, potential, detector_objs):
	"""Probe.scan over the lamella with the requested detectors.

	Returns a list ``[per-detector measurement]`` in the same order as
	``detector_objs``. Normalises the single-detector case (abtem returns
	a single ensemble there, not a 1-element list).
	"""
	# Reuse the proven probe/scan builders (single source of truth, shared
	# with the legacy pipeline + the aggregator's projection preview).
	probe = add_probe(ctx, potential)
	scan = add_scan(ctx, probe, potential)

	raw = probe.scan(potential, scan=scan, detectors=detector_objs).compute()
	if len(detector_objs) == 1:
		return [raw]
	return list(raw)


def run_diffraction(ctx, potential):
	"""Plane-wave multislice → diffraction patterns. Returns a CPU-side
	per-snapshot pattern (no ensemble averaging — that's the aggregator's
	job, since this worker handles a single seed)."""
	# CPU-side multislice first (matches the proven legacy plot_diffraction
	# path), with a real fallback to the potential's native device.
	pw = abtem.PlaneWave(energy=ctx.HT_value, device="cpu")
	try:
		exit_waves = pw.multislice(potential.to_cpu()).compute()
	except Exception:
		exit_waves = pw.multislice(potential).compute()
	return exit_waves.diffraction_patterns(
		max_angle="valid", block_direct=True
	).compute().to_cpu()


def run_cbed(ctx, potential):
	"""Probe-at-center CBED. Returns a CPU-side per-snapshot pattern."""
	probe = add_probe(ctx, potential)

	center = np.array([[
		0.5 * (ctx.scan_start[0] + ctx.scan_stop[0]),
		0.5 * (ctx.scan_start[1] + ctx.scan_stop[1]),
	]], dtype=float)

	try:
		exit_waves = probe.multislice(potential.to_cpu(), scan=center).compute()
	except Exception:
		exit_waves = probe.multislice(potential, scan=center).compute()
	cbed = exit_waves.diffraction_patterns(
		max_angle=ctx.cbed_max_angle, block_direct=False
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
	running_path = _claim_todo(todo_path)

	try:
		_, cfg = load_job_config(job_dir)
		ctx = resolve_context(cfg)

		out_dir = job_dir / "outputs"
		out_dir.mkdir(parents=True, exist_ok=True)

		# Preemption guard: SIGTERM (spot 2-minute warning) or SIGINT cleans the
		# partial seed_NNNNNN_* outputs before exit so the aggregator can't pick
		# up a half-written zarr. The .running claim is requeued to .todo on a
		# best-effort basis. The try/finally ensures the guard is uninstalled on
		# both the success path and any exception, so a caller looping over .todos
		# doesn't carry a stale handler into the next seed.
		restore_handlers = _install_preemption_handler(
			out_dir, seed, todo_path=todo_path, running_path=running_path,
		)
		try:
			# 1) Static (deterministic) lamella — read job_dir/surf.xyz (written by
			#    the generator) so worker, aggregator, and planning all see the same
			#    atoms. Falls back to a fresh build with a WARNING if surf.xyz is
			#    missing or unreadable.
			lamella = load_ground_state_atoms(job_dir, cfg)

			# 2) Per-seed phonon displacement.
			displaced = _displaced_atoms(lamella, ctx.fph_sigma, seed)

			# 3) Test mode: dump the displaced atoms BEFORE multislice
			#    (so a crashed run still leaves the displacements inspectable).
			if ctx.test_enabled:
				ase.io.write(
					str(out_dir / f"seed_{seed:06d}_displaced.xyz"),
					displaced,
					"xyz",
				)

			# 4) Build the per-seed Potential.
			potential = make_potential(displaced).build().compute()

			# 5) Projected potential of THIS seed's displaced lattice - i.e. the exact
			#    potential the multislice below propagates through. Written per seed
			#    (always, regardless of do_full_run) so the aggregator can mean it
			#    across seeds like the scan channels; the result then matches what was
			#    simulated rather than an idealised static lattice. With do_full_run
			#    off (and diffraction/cbed off) this is the only output - a cheap
			#    projection-only preview.
			proj = potential.project().to_cpu().compute()
			proj.to_tiff(str(out_dir / f"seed_{seed:06d}_potproj.tif"))
			proj.to_zarr(str(out_dir / f"seed_{seed:06d}_potproj.zarr"), overwrite=True)

			# 6) Optional plane-wave diffraction. Write .zarr too - the aggregator
			#    means seed_*_<channel>.zarr across seeds; .tif is for eyeballing.
			if ctx.do_diffraction:
				diff = run_diffraction(ctx, potential)
				diff.to_tiff(str(out_dir / f"seed_{seed:06d}_diff.tif"))
				diff.to_zarr(str(out_dir / f"seed_{seed:06d}_diff.zarr"), overwrite=True)

			# 7) Optional CBED.
			if ctx.do_cbed:
				cbed = run_cbed(ctx, potential)
				cbed.to_tiff(str(out_dir / f"seed_{seed:06d}_cbed.tif"))
				cbed.to_zarr(str(out_dir / f"seed_{seed:06d}_cbed.zarr"), overwrite=True)

			# 8) Optional scan (the main per-seed output).
			if ctx.do_full_run and ctx.detectors:
				detector_objs = _detector_objects(ctx, ctx.detectors)
				measurements = run_scan(ctx, potential, detector_objs)
				for det_name, m in zip(ctx.detectors, measurements):
					cpu = m.copy().to_cpu()
					cpu.to_tiff(str(out_dir / f"seed_{seed:06d}_{det_name}.tif"))
					cpu.to_zarr(str(out_dir / f"seed_{seed:06d}_{det_name}.zarr"), overwrite=True)

		finally:
			restore_handlers()

	except BaseException:
		_requeue_running(todo_path, running_path)
		raise

	# 9) Mark this seed done — only after the handlers are restored, so a late
	#    SIGTERM can't clean the finished outputs once the .running claim is
	#    already .done (deleting a complete, now-unreclaimable seed). Atomic rename
	#    keeps a polling aggregator's view coherent.
	done_path = todo_path.with_suffix(".done")
	running_path.rename(done_path)


def main():
	"""Module entry point for ``python -m abtem_run.worker``."""
	configure_default_logging()
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
