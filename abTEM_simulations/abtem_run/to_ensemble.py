#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Convert per-seed zarrs into abTEM-native ensemble measurements.

Each channel becomes one ``*_ensemble.zarr`` with a ``FrozenPhononsAxis``.
Calling ``.reduce_ensemble()`` on that measurement gives the same thermal mean
as the regular aggregator.
"""
import argparse
import sys
from pathlib import Path

import abtem

from .compat import bootstrap
from .job_io import collect_seed_zarrs


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _discover_channels(job_dir: Path) -> list[str]:
	"""Auto-discover available channels by globbing per-seed zarrs.

	Returns the sorted set of ``<channel>`` strings found in
	``outputs/seed_*_<channel>.zarr`` and
	``outputs_archive/seed_*_<channel>.zarr``.
	"""
	channels: set[str] = set()
	for sub in ("outputs", "outputs_archive"):
		d = job_dir / sub
		if not d.exists():
			continue
		for p in d.glob("seed_*_*.zarr"):
			# seed_NNNNNN_<channel>.zarr  ->  ['seed', 'NNNNNN', '<channel>']
			# split with maxsplit=2 so a channel that ever contained an
			# underscore (none today, but harmless to be defensive) is
			# captured whole.
			parts = p.stem.split("_", 2)
			if len(parts) == 3:
				channels.add(parts[2])
	return sorted(channels)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def load_ensemble(job_dir, channel: str):
	"""Stack per-seed ``seed_*_<channel>.zarr`` into a single abtem
	Measurement with ``FrozenPhononsAxis(_ensemble_mean=True)`` on the
	new ensemble axis.

	Reads from BOTH ``<job_dir>/outputs/`` and
	``<job_dir>/outputs_archive/`` and sorts by seed integer so the
	returned ensemble has a deterministic order across calls.

	Args:
		job_dir: path to the job directory (``gen_*/<phase>_<hkl>_<tilt>/``).
		channel: channel name (e.g. ``"haadf"``, ``"abf"``, ``"diff"``,
		         ``"cbed"``).

	Returns:
		An abtem Measurement (``Images`` or ``DiffractionPatterns``)
		of shape ``(N, ...)`` with the ensemble axis labelled
		``FrozenPhononsAxis``. ``None`` if no per-seed zarrs exist for
		this channel.

	Note:
		``.reduce_ensemble()`` on the returned object produces the
		thermal average — same value as the regular aggregator writes
		to ``<vdir>/scans/<channel>.zarr``.
	"""
	job_dir = Path(job_dir).resolve()
	out_dir = job_dir / "outputs"
	archive_dir = job_dir / "outputs_archive"

	# Use the same deterministic seed ordering and duplicate-handling as
	# the cumulative-mean pathway.
	zarrs = collect_seed_zarrs(out_dir, archive_dir, channel)
	if not zarrs:
		return None

	measurements = [abtem.from_zarr(str(f)) for f in zarrs]
	# We want abtem.stack(arrays, axis_metadata=FrozenPhononsAxis(...)),
	# but abtem 1.0.9's `validate_axis_metadata` (called inside
	# abtem.stack) only lets `OrdinalAxis` subclasses through the
	# AxisMetadata branch — every other AxisMetadata subclass
	# (FrozenPhononsAxis included) is rejected with a misleading
	# "axis_metadata must be a dict, sequence of strings or an
	# AxisMetadata object" message. The underlying ``_stack`` classmethod
	# accepts any AxisMetadata, so we route around the validator. If a
	# future abtem release widens validate_axis_metadata, switch back
	# to the public ``abtem.stack`` call.
	from abtem.core.axes import FrozenPhononsAxis
	stacked = type(measurements[0])._stack(
		measurements,
		FrozenPhononsAxis(_ensemble_mean=True),
		0,
	)
	return stacked.compute() if hasattr(stacked, "compute") else stacked


def to_ensemble_files(
	job_dir,
	*,
	channels: list[str] | None = None,
	out_dir=None,
) -> list[tuple[str, Path]]:
	"""For each channel, write
	``<out_dir>/<channel>_ensemble.zarr`` carrying the abtem-native
	N-snapshot ensemble (see ``load_ensemble`` for shape + metadata).

	Args:
		job_dir: path to the job directory.
		channels: list of channel names to emit. ``None`` (default)
		          auto-discovers from existing per-seed zarrs.
		out_dir: where to write the ``*_ensemble.zarr`` files. ``None``
		         (default) uses ``<vdir>/ensemble/`` — the aggregate
		         version dir matching the job's config (rediscovered, or
		         created if no aggregate has run yet).

	Returns:
		List of ``(channel, output_zarr_path)`` for channels actually
		written. Channels with no per-seed zarrs are silently skipped.

	Raises:
		FileNotFoundError: ``job_dir`` doesn't exist.
		ValueError: explicit ``channels`` was given but none of them
		    have per-seed zarrs in this job dir.
	"""
	job_dir = Path(job_dir).resolve()
	if not job_dir.is_dir():
		raise FileNotFoundError(f"job_dir does not exist: {job_dir}")

	if out_dir is None:
		from .aggregate import version_dir_for
		out_dir = version_dir_for(job_dir) / "ensemble"
	else:
		out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	# Narrow channels to list[str] in a way mypy can follow.
	if channels is None:
		channels = _discover_channels(job_dir)
		auto = True
	else:
		auto = False

	results: list[tuple[str, Path]] = []
	for ch in channels:
		ensemble = load_ensemble(job_dir, ch)
		if ensemble is None:
			# explicit-list miss is an error (user asked for something
			# we can't deliver); auto-discover miss is impossible
			# (channel was found by globbing).
			continue
		out_path = out_dir / f"{ch}_ensemble.zarr"
		ensemble.to_zarr(str(out_path), overwrite=True)
		results.append((ch, out_path))

	if not auto and not results:
		raise ValueError(
			f"None of the requested channels {channels!r} have per-seed "
			f"zarrs in {job_dir}. Available: {_discover_channels(job_dir)!r}."
		)
	return results


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def main():
	"""Module entry point."""
	bootstrap()
	parser = argparse.ArgumentParser(
		description="Convert per-seed zarrs into abTEM ensemble zarrs."
	)
	parser.add_argument(
		"job_dir",
		help="job directory (gen_*/<phase>_<hkl>_<tilt>/)",
	)
	parser.add_argument(
		"--channel",
		action="append",
		metavar="CH",
		help=(
			"channel name to convert (haadf / abf / bf / diff / cbed). "
			"May be repeated; if omitted, all channels with per-seed "
			"zarrs are converted."
		),
	)
	parser.add_argument(
		"--out",
		metavar="DIR",
		help="output directory (default: the matching aggregate version dir's ensemble/)",
	)
	args = parser.parse_args()

	channels = args.channel if args.channel else None
	out_dir = Path(args.out) if args.out else None
	results = to_ensemble_files(args.job_dir, channels=channels, out_dir=out_dir)
	if not results:
		print("No per-seed zarrs found — nothing to write.")
		return 0
	for ch, path in results:
		print(f"  {ch:<8s} -> {path}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
