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

Reads per-seed files from ``outputs/`` ∪ ``outputs_archive/`` so a follow-up
``abtem-run-extend`` batch landing in a fresh ``outputs/`` accumulates into
the next mean. On completion, moves ``outputs/`` contents into
``outputs_archive/`` (skipped if ``simulations.test_enabled`` is true).

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
from .pipeline import make_potential, resolve_context
from .simulation import add_probe, build_lamella_from_config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _collect_seed_zarrs(out_dir: Path, archive_dir: Path, channel_name: str) -> list[Path]:
	"""All ``seed_*_<channel>.zarr`` from outputs/ AND outputs_archive/,
	sorted by seed integer for deterministic ordering.

	Cumulative-aggregation entry point. After the first aggregator pass,
	per-seed files live in outputs_archive/; a subsequent abtem-run-extend
	batch lands fresh in outputs/. Both contribute to the mean. Duplicate
	seeds shouldn't happen (extend refuses an already-present seed); if one
	does, the outputs/ copy wins.
	"""
	def _seed_key(p: Path) -> int:
		return int(p.stem.split("_")[1])  # seed_NNNNNN_<channel> -> NNNNNN

	collected: dict[int, Path] = {}
	# archive first so a current-batch outputs/ entry wins on collision
	if archive_dir.exists():
		for p in archive_dir.glob(f"seed_*_{channel_name}.zarr"):
			collected[_seed_key(p)] = p
	if out_dir.exists():
		for p in out_dir.glob(f"seed_*_{channel_name}.zarr"):
			collected[_seed_key(p)] = p
	return [collected[k] for k in sorted(collected)]


def _archive_per_seed_outputs(out_dir: Path, archive_dir: Path) -> None:
	"""Move everything in outputs/ into outputs_archive/ and remove outputs/.
	Idempotent on a missing/empty outputs/. Lets a future abtem-run-extend
	batch land in a fresh outputs/ while historical seeds stay queryable for
	the next cumulative mean.
	"""
	if not out_dir.exists():
		return
	archive_dir.mkdir(parents=True, exist_ok=True)
	for child in out_dir.iterdir():
		dest = archive_dir / child.name
		if dest.exists():
			if dest.is_dir():
				shutil.rmtree(dest)
			else:
				dest.unlink()
		shutil.move(str(child), str(dest))
	out_dir.rmdir()


def _mean_zarr_channel(out_dir: Path, archive_dir: Path, channel_name: str, *, max_seeds: int | None = None):
	"""Cross-seed mean of ``seed_*_<channel_name>.zarr`` over outputs/ ∪
	outputs_archive/; None if none exist. ``max_seeds`` (used by
	aggregate_series) caps to the first N seeds (sorted by seed integer)."""
	zarr_files = _collect_seed_zarrs(out_dir, archive_dir, channel_name)
	if max_seeds is not None:
		zarr_files = zarr_files[:max_seeds]
	if not zarr_files:
		return None

	measurements = [abtem.from_zarr(str(f)) for f in zarr_files]
	# .compute() now: downstream gaussian_filter needs a concrete numpy array
	# (scipy's pad chokes on a dask backing).
	mean = abtem.stack(measurements).mean(axis=0)
	return mean.compute() if hasattr(mean, "compute") else mean


def _emit_channel(
	out_dir: Path,
	archive_dir: Path,
	agg_dir: Path,
	channel_name: str,
	*,
	with_blurs: bool,
	blur_sigmas: list[float] | None = None,
	blur_boundary: str = "nearest",
	max_seeds: int | None = None,
):
	"""Aggregate one channel; write {channel}.{tif,zarr} (+ blurred TIFFs if
	requested) and return the cross-seed mean (or None if no seeds produced
	this channel — used for "no data, skip"; an abtem read/stack/mean error
	would raise, not return None). ``max_seeds`` (used by aggregate_series)
	limits the mean to the first ``max_seeds`` seeds (sorted by seed integer)."""
	mean = _mean_zarr_channel(out_dir, archive_dir, channel_name, max_seeds=max_seeds)
	if mean is None:
		return None

	mean.to_tiff(str(agg_dir / f"{channel_name}.tif"))
	mean.to_zarr(str(agg_dir / f"{channel_name}.zarr"), overwrite=True)

	if with_blurs:
		for sigma in (blur_sigmas or []):
			tag = str(sigma).replace(".", "-")
			blurred = mean.gaussian_filter(sigma, boundary=blur_boundary)
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


def _write_projection_previews(out_dir: Path, archive_dir: Path, ctx, cfg, target_dir: Path) -> None:
	"""Projection-preview block shared by aggregate_job + aggregate_series.

	Writes the phonon-averaged projection (mean of ``seed_*_potproj.zarr``)
	at ``target_dir/potential_projection.*`` and, if
	``simulations.emit_static_baseline``, a separate static-lattice projection
	at ``target_dir/potential_projection_static.*``. No-op if neither applies.

	The probe-shape side panel needs a grid, so the static ground-state
	potential is built once here (one cheap build) and reused for both the
	probe and the optional static baseline.
	"""
	mean_proj = _mean_zarr_channel(out_dir, archive_dir, "potproj")
	if mean_proj is None and not cfg.simulations.emit_static_baseline:
		return

	static_potential = make_potential(
		build_lamella_from_config(cfg, cfg.job.hkl_list[0])
	).build().compute()
	probe = add_probe(ctx, static_potential)

	if mean_proj is not None:
		_write_projection(mean_proj, probe, cfg, target_dir,
			"potential_projection", "phonon-averaged projection")

	if cfg.simulations.emit_static_baseline:
		static_proj = static_potential.project().to_cpu().compute()
		_write_projection(static_proj, probe, cfg, target_dir,
			"potential_projection_static", "static lattice projection")


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
		   ``seed_*_<det>.zarr`` from outputs/ ∪ outputs_archive/ into
		   ``aggregate/<det>.{tif,zarr}`` + gaussian-blurred TIFF variants.
		4. If ``do_diffraction``: same for ``diff``, plus a ``diff.png`` preview.
		5. If ``do_cbed``: same for ``cbed``, plus a ``cbed.png`` preview.
		6. Mean ``seed_*_potproj.zarr`` into the phonon-averaged projection
		   preview; if ``emit_static_baseline``, also a separate static
		   projection preview from the same ground-state potential.
		7. Move outputs/ contents into outputs_archive/ unless
		   ``simulations.test_enabled``, so future abtem-run-extend batches
		   land in a fresh outputs/ and accumulate into the next mean.

	Raises:
		FileNotFoundError: neither ``outputs/`` nor ``outputs_archive/`` exists,
		    or no ``*.toml`` in job_dir.
		RuntimeError: ``seeds/`` still has ``.todo`` files (workers not done).
	"""
	job_dir = Path(job_dir).resolve()
	out_dir = job_dir / "outputs"
	archive_dir = job_dir / "outputs_archive"
	agg_dir = job_dir / "aggregate"

	# A pure re-aggregate against just the archive is legal (no fresh workers
	# ran since last time), so either dir suffices.
	if not out_dir.exists() and not archive_dir.exists():
		raise FileNotFoundError(
			f"No outputs/ or outputs_archive/ directory in {job_dir}"
		)

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
			_emit_channel(out_dir, archive_dir, agg_dir, det_name, with_blurs=True,
				blur_sigmas=ctx.blur_sigmas, blur_boundary=ctx.blur_boundary)

	# 2. Plane-wave diffraction
	if ctx.do_diffraction:
		diff_mean = _emit_channel(out_dir, archive_dir, agg_dir, "diff", with_blurs=False)
		if diff_mean is not None:
			_write_pattern_preview(diff_mean, cfg, agg_dir, "diff", "diffraction", figsize=(10, 6))

	# 3. CBED
	if ctx.do_cbed:
		cbed_mean = _emit_channel(out_dir, archive_dir, agg_dir, "cbed", with_blurs=False)
		if cbed_mean is not None:
			_write_pattern_preview(cbed_mean, cfg, agg_dir, "cbed", "CBED", figsize=(8, 6))

	# 4. Projection preview(s): phonon-averaged + optional static baseline.
	_write_projection_previews(out_dir, archive_dir, ctx, cfg, agg_dir)

	# 5. Archive the per-seed outputs so future extends can build on them
	#    (test_enabled keeps outputs/ in place for diagnostics).
	if not ctx.test_enabled:
		_archive_per_seed_outputs(out_dir, archive_dir)


def aggregate_series(job_dir, *, n_phonons: int | None = None) -> int:
	"""Emit cumulative-mean frames at <job_dir>/aggregate/n_<k:03d>/ for
	k in 1..N. Each subdir holds the per-channel aggregate computed from the
	first k seeds (sorted by seed integer) — useful for visualising 1/sqrt(N)
	convergence without re-running multislice.

	n_phonons caps N (default: all available seeds). Returns N emitted.
	The projection preview (phonon-averaged + optional static baseline) is
	written ONCE at aggregate/, over ALL available seeds — not per-k.
	Does NOT archive outputs/ (read-only over the per-seed data).

	Raises FileNotFoundError / RuntimeError on the same conditions as
	aggregate_job.
	"""
	job_dir = Path(job_dir).resolve()
	out_dir = job_dir / "outputs"
	archive_dir = job_dir / "outputs_archive"
	agg_dir = job_dir / "aggregate"

	if not out_dir.exists() and not archive_dir.exists():
		raise FileNotFoundError(f"No outputs/ or outputs_archive/ directory in {job_dir}")

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

	# Total seeds from the first channel that has any zarrs (scan detectors
	# first, then diff, then cbed).
	probe_channels: list[str] = []
	if ctx.do_full_run:
		probe_channels.extend(ctx.detectors)
	if ctx.do_diffraction:
		probe_channels.append("diff")
	if ctx.do_cbed:
		probe_channels.append("cbed")
	total_seeds = 0
	for ch in probe_channels:
		total_seeds = len(_collect_seed_zarrs(out_dir, archive_dir, ch))
		if total_seeds:
			break
	if total_seeds == 0:
		raise FileNotFoundError(
			f"No per-seed zarrs in outputs/ or outputs_archive/ under {job_dir}; "
			"nothing to aggregate."
		)
	n_max = total_seeds if n_phonons is None else min(int(n_phonons), total_seeds)
	if n_max < 1:
		raise ValueError(f"n_phonons must be >= 1, resolved to {n_max}")

	# Projection preview(s) — once, at agg_dir (not per-k):
	# phonon-averaged over ALL seeds + optional static baseline.
	_write_projection_previews(out_dir, archive_dir, ctx, cfg, agg_dir)

	# Per-k cumulative-mean frames.
	for k in range(1, n_max + 1):
		k_dir = agg_dir / f"n_{k:03d}"
		k_dir.mkdir(parents=True, exist_ok=True)
		if ctx.do_full_run:
			for det_name in ctx.detectors:
				_emit_channel(
					out_dir, archive_dir, k_dir, det_name,
					with_blurs=True, blur_sigmas=ctx.blur_sigmas, max_seeds=k,
				)
		if ctx.do_diffraction:
			_emit_channel(out_dir, archive_dir, k_dir, "diff", with_blurs=False, max_seeds=k)
		if ctx.do_cbed:
			_emit_channel(out_dir, archive_dir, k_dir, "cbed", with_blurs=False, max_seeds=k)

	return n_max


def main():
	"""``abtem-run-aggregate`` console-script entry."""
	parser = argparse.ArgumentParser(
		description=(
			"abtem-run aggregator: mean per-seed outputs/ ∪ outputs_archive/ "
			"into aggregate/, emit potential / diff / cbed PNG previews, and "
			"archive outputs/ -> outputs_archive/ unless simulations.test_enabled."
		),
	)
	parser.add_argument("job_dir", help="job directory (gen_*/<phase>_<hkl>_<tilt>/)")
	args = parser.parse_args()
	aggregate_job(args.job_dir)
	return 0


if __name__ == "__main__":
	sys.exit(main())
