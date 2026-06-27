"""Shared job-directory naming and discovery helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .config import AppConfig, load_config


SEED_PREFIX = "seed_"
SEED_WIDTH = 6
SEED_STATES = {"todo", "running", "done"}


def load_job_config(job_dir) -> tuple[Path, AppConfig]:
	"""Load the single job-local TOML from ``job_dir``.

	Raises when the job directory contains zero or more than one ``*.toml``.
	This is the common entry check for worker, aggregate, extend, and any
	future job-level utility.
	"""
	job_dir = Path(job_dir).resolve()
	toml_candidates = list(job_dir.glob("*.toml"))
	if not toml_candidates:
		raise FileNotFoundError(f"No *.toml in {job_dir}")
	if len(toml_candidates) > 1:
		raise ValueError(
			f"Expected one *.toml in {job_dir}, found {len(toml_candidates)}: "
			f"{[p.name for p in toml_candidates]}"
		)
	return toml_candidates[0], load_config(toml_candidates[0])


def seed_from_path(path) -> int:
	"""Parse the seed integer from ``seed_NNNNNN...`` filenames.

	Works for both seed-state files, e.g. ``seed_000042.todo``, and per-seed
	output files, e.g. ``seed_000042_haadf.zarr``.
	"""
	path = Path(path)
	parts = path.stem.split("_", 2)
	if len(parts) < 2 or parts[0] != "seed":
		raise ValueError(f"Expected filename beginning with 'seed_NNNNNN', got {path.name!r}")
	try:
		return int(parts[1])
	except ValueError as e:
		raise ValueError(f"Could not parse seed number from {path.name!r}: {e}") from e


def collect_seed_zarrs(out_dir: Path, archive_dir: Path, channel_name: str) -> list[Path]:
	"""Return per-seed zarrs for one channel, sorted by seed integer.

	Files are read from ``outputs_archive/`` and ``outputs/``. If both contain
	the same seed/channel, ``outputs/`` wins so a fresh extension batch can
	override an archived stale file deterministically.
	"""
	collected: dict[int, Path] = {}
	if archive_dir.exists():
		for p in archive_dir.glob(f"seed_*_{channel_name}.zarr"):
			collected[seed_from_path(p)] = p
	if out_dir.exists():
		for p in out_dir.glob(f"seed_*_{channel_name}.zarr"):
			collected[seed_from_path(p)] = p
	return [collected[k] for k in sorted(collected)]


def contributing_seeds(
	out_dir: Path,
	archive_dir: Path,
	channels: Iterable[str],
	*,
	max_seeds: int | None = None,
) -> list[int]:
	"""Sorted union of seed integers contributing to the requested channels."""
	ids: set[int] = set()
	for ch in channels:
		files = collect_seed_zarrs(out_dir, archive_dir, ch)
		if max_seeds is not None:
			files = files[:max_seeds]
		for p in files:
			ids.add(seed_from_path(p))
	return sorted(ids)


def scan_used_seeds(job_dir: Path) -> set[int]:
	"""Seed integers already present in a job directory.

	Includes per-seed zarr outputs from ``outputs/`` and ``outputs_archive/``
	plus seed-state files in ``seeds/``. Pending and claimed seeds are counted
	so ``extend_job`` cannot re-emit work that is already queued or running.
	"""
	job_dir = Path(job_dir)
	used: set[int] = set()
	for sub in ("outputs", "outputs_archive"):
		d = job_dir / sub
		if not d.exists():
			continue
		for p in d.glob("seed_*_*.zarr"):
			try:
				used.add(seed_from_path(p))
			except ValueError:
				pass
	seeds_dir = job_dir / "seeds"
	if seeds_dir.exists():
		for state in sorted(SEED_STATES):
			for p in seeds_dir.glob(f"seed_*.{state}"):
				try:
					used.add(seed_from_path(p))
				except ValueError:
					pass
	return used


def write_seed_todo(seeds_dir: Path, seed: int, *, replace: bool = False) -> Path:
	"""Write ``seeds_dir/seed_NNNNNN.todo`` atomically and return its path.

	By default, refuses to overwrite an existing todo. ``replace=True`` is useful
	when generating a fresh job tree whose parent directory has just been created.
	"""
	seeds_dir = Path(seeds_dir)
	seeds_dir.mkdir(parents=True, exist_ok=True)
	todo = seeds_dir / f"{SEED_PREFIX}{int(seed):0{SEED_WIDTH}d}.todo"
	if not replace and todo.exists():
		raise FileExistsError(todo)
	tmp = todo.with_suffix(todo.suffix + ".tmp")
	tmp.write_text(f"{int(seed)}\n", encoding="utf-8")
	if replace:
		os.replace(tmp, todo)
	else:
		try:
			os.link(tmp, todo)
		except FileExistsError:
			tmp.unlink(missing_ok=True)
			raise
		else:
			tmp.unlink(missing_ok=True)
	return todo
