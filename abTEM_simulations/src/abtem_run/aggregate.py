#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Aggregator for the abtem-run worker pipeline.

Reads ``<job_dir>/outputs/seed_*_<channel>.zarr`` (written by the worker),
computes the mean across seeds for each channel, and writes a single
aggregate per channel into ``<job_dir>/aggregate/``. Also emits a
static-lattice projected-potential preview (one ``Potential.project()``
per job, no dependence on the per-seed runs).

Cleans up ``outputs/`` unless ``simulations.test_enabled`` is true.

CLI:
    abtem-run-aggregate <job_dir>

Library:
    from abtem_run.aggregate import aggregate_job
    aggregate_job(job_dir)

Design rationale lives in ``docs/worker.md`` (decisions #4 and #5).
"""
import argparse
import shutil
import sys
from pathlib import Path

import abtem
import matplotlib.pyplot as plt

from .config import load_config
from .pipeline import BLUR_SIGMAS, resolve_context
from .simulation import add_probe
# These two helpers live in worker.py as package-internal building blocks; we
# import them rather than duplicate the lamella/Potential construction logic.
from .worker import _build_lamella, _build_potential


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _phase_stem(phase: str) -> str:
	phase = str(phase)
	return phase[:-4] if phase.lower().endswith(".cif") else phase


def _mean_zarr_channel(out_dir: Path, channel_name: str):
	"""Mean ``seed_*_<channel_name>.zarr`` across seeds.

	Returns an abtem Measurement (Images / DiffractionPatterns / etc.) whose
	calibration is preserved by abtem's own stacking machinery and whose
	array is the cross-seed mean. Returns None if no per-seed files exist
	for this channel.
	"""
	zarr_files = sorted(out_dir.glob(f"seed_*_{channel_name}.zarr"))
	if not zarr_files:
		return None

	measurements = [abtem.from_zarr(str(f)) for f in zarr_files]

	# Public-API path: abtem.stack adds an ensemble axis, .mean(axis=0)
	# averages across it. Calibration (sampling, metadata) is preserved
	# automatically. Force compute now because downstream ops like
	# `gaussian_filter(boundary='constant')` expect a concrete numpy array
	# (scipy passes the boundary string to numpy's pad, which chokes on a
	# dask backing).
	mean = abtem.stack(measurements).mean(axis=0)
	return mean.compute() if hasattr(mean, "compute") else mean


def _emit_channel(out_dir: Path, agg_dir: Path, channel_name: str, *, with_blurs: bool) -> None:
	"""Aggregate one channel; write {channel}.{tif,zarr} (+ blurred TIFFs if requested)."""
	mean = _mean_zarr_channel(out_dir, channel_name)
	if mean is None:
		return

	mean.to_tiff(str(agg_dir / f"{channel_name}.tif"))
	mean.to_zarr(str(agg_dir / f"{channel_name}.zarr"), overwrite=True)

	if with_blurs:
		# Same blur set as the legacy save_images in pipeline.py.
		for sigma in BLUR_SIGMAS:
			tag = str(sigma).replace(".", "-")
			blurred = mean.gaussian_filter(sigma, boundary="constant")
			blurred.to_tiff(str(agg_dir / f"{channel_name}_{tag}.tif"))


def _emit_potential_projection(ctx, cfg, agg_dir: Path) -> None:
	"""Build + save the static-lattice projected potential preview.

	One ``Potential.project().compute()`` per job (independent of how many
	seeds the job had). Saves three files:
	  - ``potential_projection.png``   side-by-side projection + probe shape
	  - ``potential_projection.tif``   raw projection as TIFF
	  - ``potential_projection_scanned.tif``   cropped to scan area
	"""
	lamella = _build_lamella(ctx, cfg)
	potential = _build_potential(lamella)

	proj = potential.project().to_cpu().compute()
	probe = add_probe(ctx, potential)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	proj.show(cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1)
	probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(f"{cfg.paths.sample_name}, {_phase_stem(cfg.job.phase)}", fontsize=18)
	fig.tight_layout()
	fig.savefig(str(agg_dir / "potential_projection.png"), dpi=600)
	plt.close(fig)

	proj.to_tiff(str(agg_dir / "potential_projection.tif"))

	scan_s = cfg.lamella_settings.scan_s
	borders = cfg.lamella_settings.borders
	proj_cropped = proj.crop([scan_s, scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(agg_dir / "potential_projection_scanned.tif"))


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def aggregate_job(job_dir) -> None:
	"""Merge per-seed outputs in a job_dir into the aggregate/ subdirectory.

	Args:
		job_dir: path to the job directory
		           (``gen_*/<phase>_<hkl>_<tilt>/``).

	Steps:
		1. Verify no ``seeds/*.todo`` remain (job must be complete).
		2. Load the job's TOML, build a RunContext.
		3. For each scan detector in ``ctx.detectors``: mean per-seed
		   ``outputs/seed_*_<det>.zarr`` into ``aggregate/<det>.{tif,zarr}``,
		   plus three gaussian-blurred TIFF variants.
		4. If ``do_diffraction``: same for ``diff``.
		5. If ``do_cbed``: same for ``cbed``.
		6. Build the static-lattice potential projection preview.
		7. Delete ``outputs/`` unless ``simulations.test_enabled``.

	Raises:
		FileNotFoundError: no ``outputs/`` directory or no ``*.toml`` in job_dir.
		RuntimeError: ``seeds/`` still has ``.todo`` files (workers not done).
	"""
	job_dir = Path(job_dir).resolve()
	out_dir = job_dir / "outputs"
	agg_dir = job_dir / "aggregate"

	if not out_dir.exists():
		raise FileNotFoundError(f"No outputs/ directory in {job_dir}")

	seeds_dir = job_dir / "seeds"
	if seeds_dir.exists():
		remaining = list(seeds_dir.glob("*.todo"))
		if remaining:
			raise RuntimeError(
				f"Job incomplete: {len(remaining)} .todo file(s) remain in {seeds_dir}. "
				"Run all workers before aggregating."
			)

	toml_candidates = list(job_dir.glob("*.toml"))
	if not toml_candidates:
		raise FileNotFoundError(f"No *.toml in {job_dir}")
	if len(toml_candidates) > 1:
		raise ValueError(
			f"Expected one *.toml in {job_dir}, found {len(toml_candidates)}: "
			f"{[p.name for p in toml_candidates]}"
		)

	cfg = load_config(toml_candidates[0])
	ctx = resolve_context(cfg)

	agg_dir.mkdir(parents=True, exist_ok=True)

	# 1. Scan channels (with blurs)
	if ctx.do_full_run:
		for det_name in ctx.detectors:
			_emit_channel(out_dir, agg_dir, det_name, with_blurs=True)

	# 2. Plane-wave diffraction
	if ctx.do_diffraction:
		_emit_channel(out_dir, agg_dir, "diff", with_blurs=False)

	# 3. CBED
	if ctx.do_cbed:
		_emit_channel(out_dir, agg_dir, "cbed", with_blurs=False)

	# 4. Static-lattice projected potential preview
	_emit_potential_projection(ctx, cfg, agg_dir)

	# 5. Cleanup unless test mode is on
	if not ctx.test_enabled:
		shutil.rmtree(out_dir)


def main():
	"""``abtem-run-aggregate`` console-script entry."""
	parser = argparse.ArgumentParser(
		description=(
			"abtem-run aggregator: mean per-seed outputs/ into aggregate/, "
			"emit a static-lattice potential projection preview, and clean up "
			"outputs/ unless simulations.test_enabled is true."
		),
	)
	parser.add_argument("job_dir", help="job directory (gen_*/<phase>_<hkl>_<tilt>/)")
	args = parser.parse_args()
	aggregate_job(args.job_dir)
	return 0


if __name__ == "__main__":
	sys.exit(main())
