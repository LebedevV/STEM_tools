#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Aggregate per-seed zarr outputs into versioned job results.

Reads ``outputs/`` plus ``outputs_archive/``, averages each channel across
seeds, writes scans/patterns/projections under ``aggregate/<UTC>_<hash>/``, and
records provenance in ``seed_counts.json``. Completed fresh outputs are archived
unless ``simulations.test_enabled`` is true.
"""
import argparse
import datetime
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import abtem
import matplotlib.pyplot as plt

from ._log import configure_default_logging
from .job_io import collect_seed_zarrs, contributing_seeds, load_job_config
from .pipeline import make_potential, resolve_context
from .simulation import add_probe, load_ground_state_atoms


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _config_hash(cfg) -> str:
	"""Stable 8-char hex digest of the full config. A re-aggregate with the
	same TOML rediscovers its version dir; an edited TOML starts a fresh one.
	The UTC dir-name prefix is display/sort only — never hashed."""
	blob = json.dumps(cfg.model_dump(mode="json"), sort_keys=True)
	return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _resolve_version_dir(agg_root: Path, cfg, *, force_new: bool) -> Path:
	"""Rediscover or create the aggregate version dir ``<UTC>_<hash>/``.
	Without force_new, reuses the newest existing dir whose ``<hash>`` matches
	the current config (so re-aggregate / post-extend accumulate into the same
	dir — the seed set is deliberately NOT in the hash). force_new always makes
	a fresh dir."""
	h = _config_hash(cfg)
	agg_root.mkdir(parents=True, exist_ok=True)
	if not force_new:
		matches = sorted(agg_root.glob(f"*_{h}"), key=lambda p: p.stat().st_mtime)
		if matches:
			return matches[-1]
	utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	vdir = agg_root / f"{utc}_{h}"
	i = 2
	while vdir.exists():  # same-second force_new collision; keep <hash> as the suffix
		vdir = agg_root / f"{utc}-{i}_{h}"
		i += 1
	vdir.mkdir(parents=True, exist_ok=True)
	return vdir


def version_dir_for(job_dir, *, force_new: bool = False) -> Path:
	"""Resolve the aggregate version dir for a job from its TOML — rediscover
	the newest dir matching the config hash, or create one. Public so the
	to_ensemble bridge co-locates its output with the matching aggregate."""
	job_dir = Path(job_dir).resolve()
	_, cfg = load_job_config(job_dir)
	return _resolve_version_dir(job_dir / "aggregate", cfg, force_new=force_new)


def _write_seed_provenance(target_dir: Path, seeds: list[int], counts: dict[str, int]) -> None:
	"""Write ``<target_dir>/seed_counts.json`` = {"seeds": [...], "channels":
	{channel: n}}. ``seeds`` is the atoms-level record of which frozen-phonon
	snapshots fed this aggregate (one seed feeds all channels); a per-channel
	count below len(seeds) flags a channel missing some snapshots."""
	if not counts:
		return
	payload = {"seeds": seeds, "channels": counts}
	(target_dir / "seed_counts.json").write_text(json.dumps(payload, indent=2) + "\n")
	log.info("aggregate: seeds=%d, per-channel %s", len(seeds), counts)


def _archive_per_seed_outputs(out_dir: Path, archive_dir: Path) -> None:
	"""Move outputs/ contents into outputs_archive/ and remove outputs/.

	Refuses if ``out_dir`` isn't a real directory named exactly ``outputs``
	living next to ``archive_dir`` — guards against symlink trickery or
	mis-resolved paths sending ``shutil.move`` / ``rmtree`` outside the
	job tree.
	"""
	if not out_dir.exists():
		return
	if (
		out_dir.is_symlink()
		or not out_dir.is_dir()
		or out_dir.name != "outputs"
		or archive_dir.name != "outputs_archive"
		or out_dir.parent != archive_dir.parent
	):
		raise ValueError(
			f"refuse to archive: {out_dir} must be a real 'outputs' "
			f"directory sibling of '{archive_dir.name}'"
		)
	archive_dir.mkdir(parents=True, exist_ok=True)
	for child in out_dir.iterdir():
		dest = archive_dir / child.name
		if dest.exists():
			if dest.is_dir() and not dest.is_symlink():
				shutil.rmtree(dest)
			else:
				dest.unlink()
		shutil.move(str(child), str(dest))
	out_dir.rmdir()


def _mean_zarr_channel(out_dir: Path, archive_dir: Path, channel_name: str, *, max_seeds: int | None = None):
	"""Cross-seed mean of ``seed_*_<channel_name>.zarr`` over outputs/ ∪
	outputs_archive/ + the contributing-seed count; ``(None, 0)`` if none
	exist. ``max_seeds`` caps to the first N."""
	zarr_files = collect_seed_zarrs(out_dir, archive_dir, channel_name)
	if max_seeds is not None:
		zarr_files = zarr_files[:max_seeds]
	if not zarr_files:
		return None, 0

	measurements = [abtem.from_zarr(str(f)) for f in zarr_files]
	# .compute(): downstream gaussian_filter wants a concrete numpy array.
	mean = abtem.stack(measurements).mean(axis=0)
	mean = mean.compute() if hasattr(mean, "compute") else mean
	return mean, len(zarr_files)


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
	"""Write {channel}.{tif,zarr} (+ blurred TIFFs) for the cross-seed mean.
	Returns ``(mean, n_seeds)``: the mean (None if no per-seed zarrs exist for
	this channel, in which case nothing is written) and the number of seeds
	that contributed — the caller surfaces a 0 via seed_counts.json."""
	mean, n_seeds = _mean_zarr_channel(out_dir, archive_dir, channel_name, max_seeds=max_seeds)
	if mean is None:
		return None, 0

	mean.to_tiff(str(agg_dir / f"{channel_name}.tif"))
	mean.to_zarr(str(agg_dir / f"{channel_name}.zarr"), overwrite=True)

	if with_blurs:
		for sigma in (blur_sigmas or []):
			tag = str(sigma).replace(".", "-")
			blurred = mean.gaussian_filter(sigma, boundary=blur_boundary)
			blurred.to_tiff(str(agg_dir / f"{channel_name}_{tag}.tif"))
	return mean, n_seeds


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


def _load_or_build_static_potential(job_dir: Path, cfg):
	"""Cached ground-state Potential. Lives at ``job_dir/static_potential.zarr``
	(literal `static` per the static-naming rule — survives cleanup, never
	picked up by the seed_* averaging glob). Cache hit when the zarr is at
	least as new as ``surf.xyz``; otherwise rebuild via
	``load_ground_state_atoms`` + ``make_potential``, save, return.
	"""
	cache = Path(job_dir) / "static_potential.zarr"
	surf = Path(job_dir) / "surf.xyz"
	if (
		cache.exists()
		and surf.exists()
		and cache.stat().st_mtime >= surf.stat().st_mtime
	):
		log.info(f"static_potential: cache hit ({cache})")
		return abtem.from_zarr(str(cache)).compute()
	log.info(f"static_potential: rebuilding (cache miss) -> {cache}")
	pot = make_potential(load_ground_state_atoms(job_dir, cfg)).build().compute()
	# Atomic publish: write a temp store, then rename onto the cache path, so
	# an interrupted write never leaves a partial zarr that mtime would treat
	# as a hit (zarr v2 reads missing chunks back as zeros, silently).
	tmp = cache.with_name(cache.name + ".tmp")
	if tmp.exists():
		shutil.rmtree(tmp)
	pot.to_zarr(str(tmp), overwrite=True)
	if cache.exists():
		shutil.rmtree(cache)
	tmp.replace(cache)
	return pot


def _write_projection_previews(out_dir: Path, archive_dir: Path, cfg, job_dir: Path, target_dir: Path) -> None:
	"""Phonon-averaged projection at ``target_dir/potential_projection.*``,
	plus a static-lattice one at ``..._static.*`` if ``emit_static_baseline``.
	No-op if neither applies. The ground-state Potential is loaded (or built
	and cached on miss) via ``_load_or_build_static_potential`` at ``job_dir``
	(shared across aggregate version dirs) and reused for the probe grid
	(abtem's Grid.match wants a Potential / a thing exposing ``gpts``, not an
	Images) and the optional static projection."""
	mean_proj, _ = _mean_zarr_channel(out_dir, archive_dir, "potproj")
	if mean_proj is None and not cfg.simulations.emit_static_baseline:
		return

	static_potential = _load_or_build_static_potential(job_dir, cfg)
	probe = add_probe(cfg, static_potential)

	if mean_proj is not None:
		_write_projection(mean_proj, probe, cfg, target_dir,
			"potential_projection", "phonon-averaged projection")

	if cfg.simulations.emit_static_baseline:
		static_proj = static_potential.project().to_cpu().compute()
		_write_projection(static_proj, probe, cfg, target_dir,
			"potential_projection_static", "static lattice projection")


def _write_pattern_preview(measurement, cfg, agg_dir: Path, stem: str,
		kind_label: str, *, figsize: tuple[float, float]) -> None:
	"""Write a PNG preview for an averaged diffraction-style measurement."""
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


def aggregate_job(job_dir, *, force_new: bool = False) -> None:
	"""Average completed seed outputs into ``aggregate/<UTC>_<hash>/``."""
	job_dir = Path(job_dir).resolve()
	out_dir = job_dir / "outputs"
	archive_dir = job_dir / "outputs_archive"
	agg_dir = job_dir / "aggregate"

	# Re-aggregating archived-only jobs is valid.
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

	_, cfg = load_job_config(job_dir)
	ctx = resolve_context(cfg)

	vdir = _resolve_version_dir(agg_dir, cfg, force_new=force_new)
	scans_dir = vdir / "scans"
	patterns_dir = vdir / "patterns"
	proj_dir = vdir / "projections"

	seed_counts: dict[str, int] = {}
	# 1. Scan channels (with blurs) -> scans/
	if cfg.simulations.do_full_run:
		scans_dir.mkdir(exist_ok=True)
		for det_name in cfg.microscope.detectors:
			_, seed_counts[det_name] = _emit_channel(out_dir, archive_dir, scans_dir, det_name,
				with_blurs=True, blur_sigmas=cfg.simulations.blur_sigmas, blur_boundary=cfg.simulations.blur_boundary)

	# 2/3. Plane-wave diffraction + CBED -> patterns/ (+ PNG previews)
	if cfg.microscope.do_diffraction or cfg.microscope.do_cbed:
		patterns_dir.mkdir(exist_ok=True)
	if cfg.microscope.do_diffraction:
		diff_mean, seed_counts["diff"] = _emit_channel(out_dir, archive_dir, patterns_dir, "diff", with_blurs=False)
		if diff_mean is not None:
			_write_pattern_preview(diff_mean, cfg, patterns_dir, "diff", "diffraction", figsize=(10, 6))
	if cfg.microscope.do_cbed:
		cbed_mean, seed_counts["cbed"] = _emit_channel(out_dir, archive_dir, patterns_dir, "cbed", with_blurs=False)
		if cbed_mean is not None:
			_write_pattern_preview(cbed_mean, cfg, patterns_dir, "cbed", "CBED", figsize=(8, 6))

	# 4. Seed provenance -> <vdir>/seed_counts.json (atoms-level seeds + counts).
	seeds = contributing_seeds(out_dir, archive_dir, list(seed_counts))
	_write_seed_provenance(vdir, seeds, seed_counts)

	# 5. Projection preview(s) -> projections/: phonon-averaged + optional static.
	proj_dir.mkdir(exist_ok=True)
	_write_projection_previews(out_dir, archive_dir, cfg, job_dir, proj_dir)

	# Keep diagnostic outputs visible in test mode.
	if not cfg.simulations.test_enabled:
		_archive_per_seed_outputs(out_dir, archive_dir)


def aggregate_series(job_dir, *, n_phonons: int | None = None, force_new: bool = False) -> int:
	"""Write cumulative means for the first k seeds under ``series/n_<k>/``."""
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

	_, cfg = load_job_config(job_dir)
	ctx = resolve_context(cfg)
	vdir = _resolve_version_dir(agg_dir, cfg, force_new=force_new)
	proj_dir = vdir / "projections"

	# Total seeds from the first channel that has any zarrs (scan detectors
	# first, then diff, then cbed).
	probe_channels: list[str] = []
	if cfg.simulations.do_full_run:
		probe_channels.extend(cfg.microscope.detectors)
	if cfg.microscope.do_diffraction:
		probe_channels.append("diff")
	if cfg.microscope.do_cbed:
		probe_channels.append("cbed")
	total_seeds = 0
	for ch in probe_channels:
		total_seeds = len(collect_seed_zarrs(out_dir, archive_dir, ch))
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

	# Projection preview is written once for all available seeds.
	proj_dir.mkdir(exist_ok=True)
	_write_projection_previews(out_dir, archive_dir, cfg, job_dir, proj_dir)

	# Per-k cumulative-mean frames at <vdir>/series/n_<k>/.
	series_dir = vdir / "series"
	for k in range(1, n_max + 1):
		k_dir = series_dir / f"n_{k:03d}"
		k_scans = k_dir / "scans"
		k_patterns = k_dir / "patterns"
		seed_counts: dict[str, int] = {}
		if cfg.simulations.do_full_run:
			k_scans.mkdir(parents=True, exist_ok=True)
			for det_name in cfg.microscope.detectors:
				_, seed_counts[det_name] = _emit_channel(
					out_dir, archive_dir, k_scans, det_name,
					with_blurs=True, blur_sigmas=cfg.simulations.blur_sigmas,
					blur_boundary=cfg.simulations.blur_boundary, max_seeds=k,
				)
		if cfg.microscope.do_diffraction or cfg.microscope.do_cbed:
			k_patterns.mkdir(parents=True, exist_ok=True)
		if cfg.microscope.do_diffraction:
			diff_mean, seed_counts["diff"] = _emit_channel(
				out_dir, archive_dir, k_patterns, "diff",
				with_blurs=False, max_seeds=k,
			)
			if diff_mean is not None:
				_write_pattern_preview(
					diff_mean, cfg, k_patterns, "diff", "diffraction", figsize=(10, 6),
				)
		if cfg.microscope.do_cbed:
			cbed_mean, seed_counts["cbed"] = _emit_channel(
				out_dir, archive_dir, k_patterns, "cbed",
				with_blurs=False, max_seeds=k,
			)
			if cbed_mean is not None:
				_write_pattern_preview(
					cbed_mean, cfg, k_patterns, "cbed", "CBED", figsize=(8, 6),
				)
		k_dir.mkdir(parents=True, exist_ok=True)
		seeds = contributing_seeds(out_dir, archive_dir, list(seed_counts), max_seeds=k)
		_write_seed_provenance(k_dir, seeds, seed_counts)

	return n_max


def main():
	"""Module entry point."""
	configure_default_logging()
	parser = argparse.ArgumentParser(
		description="Aggregate one abtem_run job directory."
	)
	parser.add_argument("job_dir", help="job directory (gen_*/<phase>_<hkl>_<tilt>/)")
	args = parser.parse_args()
	aggregate_job(args.job_dir)
	return 0


if __name__ == "__main__":
	sys.exit(main())
