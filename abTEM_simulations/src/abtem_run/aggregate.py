#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Aggregator for the abtem-run worker pipeline.

Reads ``<job_dir>/outputs/seed_*_<channel>.zarr`` (written by the worker),
computes the mean across seeds for each channel, and writes a single
aggregate per channel into ``<job_dir>/aggregate/``. The projected-potential
channel (``seed_*_potproj``) means to a phonon-averaged projection preview, so
it reflects the potentials actually propagated through; setting
``simulations.emit_static_baseline`` adds a separate static-lattice projection.

Diff and CBED channels also get a matplotlib PNG preview alongside the
``.tif``/``.zarr``, mirroring the legacy in-process pipeline's
``plot_diffraction`` / ``plot_cbed`` output so the worker pipeline is the
single producer of those previews.

Cleans up ``outputs/`` unless ``simulations.test_enabled`` is true.

CLI:
    abtem-run-aggregate <job_dir>

Library:
    from abtem_run.aggregate import aggregate_job
    aggregate_job(job_dir)
"""
import argparse
import shutil
import sys
from pathlib import Path

import abtem
import matplotlib.pyplot as plt

from .config import load_config
from .pipeline import BLUR_SIGMAS, make_potential, resolve_context
from .simulation import add_probe, build_lamella_from_config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mean_zarr_channel(out_dir: Path, channel_name: str):
	"""Cross-seed mean of ``seed_*_<channel_name>.zarr``; None if none exist."""
	zarr_files = sorted(out_dir.glob(f"seed_*_{channel_name}.zarr"))
	if not zarr_files:
		return None

	measurements = [abtem.from_zarr(str(f)) for f in zarr_files]
	# .compute() now: downstream gaussian_filter needs a concrete numpy array
	# (scipy's pad chokes on a dask backing).
	mean = abtem.stack(measurements).mean(axis=0)
	return mean.compute() if hasattr(mean, "compute") else mean


def _emit_channel(out_dir: Path, agg_dir: Path, channel_name: str, *, with_blurs: bool):
	"""Aggregate one channel; write {channel}.{tif,zarr} (+ blurred TIFFs if
	requested) and return the cross-seed mean (or None if no seeds produced
	this channel — used for "no data, skip"; an abtem read/stack/mean error
	would raise, not return None)."""
	mean = _mean_zarr_channel(out_dir, channel_name)
	if mean is None:
		return None

	mean.to_tiff(str(agg_dir / f"{channel_name}.tif"))
	mean.to_zarr(str(agg_dir / f"{channel_name}.zarr"), overwrite=True)

	if with_blurs:
		# Same blur set as the legacy save_images.
		for sigma in BLUR_SIGMAS:
			tag = str(sigma).replace(".", "-")
			blurred = mean.gaussian_filter(sigma, boundary="constant")
			blurred.to_tiff(str(agg_dir / f"{channel_name}_{tag}.tif"))
	return mean


def _suptitle(cfg, kind_label: str) -> str:
	"""Common matplotlib suptitle for aggregate previews: ``sample, sg [hkl] — <kind>``."""
	sg = cfg.job.phase[:-4] if cfg.job.phase.lower().endswith('.cif') else cfg.job.phase
	hkl = "".join(str(x) for x in cfg.job.hkl_list[0])
	return f"{cfg.paths.sample_name}, {sg} [{hkl}] — {kind_label}"


def _write_projection(proj, probe, cfg, agg_dir: Path, stem: str, kind_label: str) -> None:
	"""Write a projected-potential preview from an already-computed projection
	``proj`` (an ``Images``) and a ``probe`` for the side panel. Saves:
	  - ``<stem>.png``          side-by-side projection + probe shape
	  - ``<stem>.tif``          raw projection as TIFF
	  - ``<stem>_scanned.tif``  cropped to the scan area
	"""
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	proj.show(cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1)
	probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(_suptitle(cfg, kind_label), fontsize=18)
	fig.tight_layout()
	fig.savefig(str(agg_dir / f"{stem}.png"), dpi=600)
	plt.close(fig)

	proj.to_tiff(str(agg_dir / f"{stem}.tif"))

	scan_s = cfg.lamella_settings.scan_s
	borders = cfg.lamella_settings.borders
	proj_cropped = proj.crop([scan_s, scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(agg_dir / f"{stem}_scanned.tif"))


def _write_pattern_preview(measurement, cfg, agg_dir: Path, stem: str,
		kind_label: str, *, figsize: tuple[float, float]) -> None:
	"""Matplotlib PNG preview for a 2D diffraction-style measurement (averaged
	plane-wave diffraction or CBED). Mirrors the legacy plot_diffraction /
	plot_cbed visual style so the worker pipeline produces an equivalent
	figure."""
	measurement.show(
		explode=False, power=0.2, units="mrad",
		figsize=figsize, cbar=True, common_color_scale=True,
	)
	fig = plt.gcf()
	fig.suptitle(_suptitle(cfg, kind_label), y=1.005)
	fig.savefig(str(agg_dir / f"{stem}.png"), dpi=600)
	plt.close(fig)


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
		4. If ``do_diffraction``: same for ``diff``, plus a ``diff.png`` preview.
		5. If ``do_cbed``: same for ``cbed``, plus a ``cbed.png`` preview.
		6. Mean ``seed_*_potproj.zarr`` into the phonon-averaged projection
		   preview; if ``emit_static_baseline``, also a separate static one.
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
		diff_mean = _emit_channel(out_dir, agg_dir, "diff", with_blurs=False)
		if diff_mean is not None:
			_write_pattern_preview(diff_mean, cfg, agg_dir, "diff", "diffraction", figsize=(10, 6))

	# 3. CBED
	if ctx.do_cbed:
		cbed_mean = _emit_channel(out_dir, agg_dir, "cbed", with_blurs=False)
		if cbed_mean is not None:
			_write_pattern_preview(cbed_mean, cfg, agg_dir, "cbed", "CBED", figsize=(8, 6))

	# 4. Projected potential preview(s). Default is the phonon-averaged
	#    projection: the mean of each seed's seed_*_potproj.zarr, i.e. the
	#    projection of the potentials actually propagated through (matches the
	#    simulation, not an idealised lattice). The probe-shape side panel needs
	#    a grid, so build the ground-state potential once here (one cheap build)
	#    and reuse it for both the probe and the optional static baseline.
	mean_proj = _mean_zarr_channel(out_dir, "potproj")
	if mean_proj is not None or cfg.simulations.emit_static_baseline:
		static_potential = make_potential(
			build_lamella_from_config(cfg, cfg.job.hkl_list[0])
		).build().compute()
		probe = add_probe(ctx, static_potential)

		if mean_proj is not None:
			_write_projection(mean_proj, probe, cfg, agg_dir,
				"potential_projection", "phonon-averaged projection")

		# emit_static_baseline: a separate static-lattice projection, kept on
		# its own and never averaged with the per-seed runs.
		if cfg.simulations.emit_static_baseline:
			static_proj = static_potential.project().to_cpu().compute()
			_write_projection(static_proj, probe, cfg, agg_dir,
				"potential_projection_static", "static lattice projection")

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
