#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Push-mode per-seed worker for the abtem-run worker pipeline.

Consumes ONE ``seeds/seed_NNNNNN.todo`` file in a job directory, runs the
multislice for that single phonon snapshot, writes outputs to
``outputs/seed_NNNNNN_<channel>.{zarr,tif}``, and renames the
``.todo`` to ``.done``.

Push interface (the worker is stateless; the caller — bash loop, GNU
parallel, slurm, or the convenience wrapper — supplies the .todo path):

    abtem-run-worker <job_dir> <todo_path>

Library entry:

    from abtem_run.worker import run_one_seed
    run_one_seed(job_dir, todo_path)

Every seed goes through the same code path — there is no special
"static lattice" branch. To produce a static-lattice result, configure
``fph_sigma = false`` and ``frozen_phonons = 1``; the worker will run
with zero displacement.
"""
import argparse
import sys
from pathlib import Path

import abtem
import ase.io
import numpy as np

from .config import load_config
from .pipeline import make_potential, resolve_context
from .simulation import add_probe, add_scan, build_lamella_from_config


# --------------------------------------------------------------------------- #
# Building blocks — factored so an in-memory wrapper can compose them without I/O.
# --------------------------------------------------------------------------- #


def _seed_from_todo(todo_path: Path) -> int:
	"""Parse ``seed_NNNNNN.todo`` → ``NNNNNN``."""
	stem = todo_path.stem  # e.g. 'seed_000042'
	prefix = "seed_"
	if not stem.startswith(prefix):
		raise ValueError(
			f"Expected todo filename like 'seed_NNNNNN.todo', got {todo_path.name!r}"
		)
	try:
		return int(stem[len(prefix):])
	except ValueError as e:
		raise ValueError(
			f"Could not parse seed number from {todo_path.name!r}: {e}"
		) from e


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
		- Renames ``seeds/seed_NNNNNN.todo`` → ``seeds/seed_NNNNNN.done``.

	On exception: the .todo is NOT renamed, so a retry picks up the same
	work. The aggregator will see only .done seeds.
	"""
	job_dir = Path(job_dir).resolve()
	todo_path = Path(todo_path).resolve()

	seed = _seed_from_todo(todo_path)

	# Locate the job-local TOML (one *.toml at the job_dir root).
	toml_candidates = list(job_dir.glob("*.toml"))
	if not toml_candidates:
		raise FileNotFoundError(f"No *.toml in job_dir {job_dir}")
	if len(toml_candidates) > 1:
		raise ValueError(
			f"Expected one *.toml in {job_dir}, found {len(toml_candidates)}: "
			f"{[p.name for p in toml_candidates]}"
		)

	cfg = load_config(toml_candidates[0])
	ctx = resolve_context(cfg)

	out_dir = job_dir / "outputs"
	out_dir.mkdir(parents=True, exist_ok=True)

	# 1) Static (deterministic) lamella from cfg.job + cfg.lamella_settings.
	lamella = build_lamella_from_config(cfg, cfg.job.hkl_list[0])

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

	# 5) Projected potential of THIS seed's displaced lattice — i.e. the exact
	#    potential the multislice below propagates through. Written per seed
	#    (always, regardless of do_full_run) so the aggregator can mean it
	#    across seeds like the scan channels; the result then matches what was
	#    simulated rather than an idealised static lattice. With do_full_run
	#    off (and diffraction/cbed off) this is the only output — a cheap
	#    projection-only preview.
	proj = potential.project().to_cpu().compute()
	proj.to_tiff(str(out_dir / f"seed_{seed:06d}_potproj.tif"))
	proj.to_zarr(str(out_dir / f"seed_{seed:06d}_potproj.zarr"), overwrite=True)

	# 6) Optional plane-wave diffraction. Write .zarr too — the aggregator
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

	# 9) Mark this seed done. Atomic rename so concurrent readers
	#    (e.g. the aggregator polling for completion) see a coherent state.
	done_path = todo_path.with_suffix(".done")
	todo_path.rename(done_path)


def main():
	"""``abtem-run-worker`` console-script entry."""
	parser = argparse.ArgumentParser(
		description=(
			"abtem-run worker: process one .todo, write per-seed outputs to "
			"<job_dir>/outputs/, and rename the .todo to .done."
		),
	)
	parser.add_argument("job_dir", help="job directory (gen_*/<phase>_<hkl>_<tilt>/)")
	parser.add_argument("todo_path", help="path to one seeds/seed_NNNNNN.todo file")
	args = parser.parse_args()
	run_one_seed(args.job_dir, args.todo_path)
	return 0


if __name__ == "__main__":
	sys.exit(main())
