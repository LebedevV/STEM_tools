#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
``abtem-run-extend`` — append more phonon snapshots to an existing job dir.

Emits new ``seeds/seed_*.todo`` files with non-overlapping seed integers so
a follow-up worker + aggregator pass produces the cumulative mean over the
union of old (archived) + new seeds.

CLI:
  abtem-run-extend <job_dir> --add N
  abtem-run-extend <job_dir> --seeds 23,24,25

Library: extend_job(job_dir, add=N) or extend_job(job_dir, seeds=[...]).

Refuses when no prior batch exists (outputs/ and outputs_archive/ both
empty). Each call appends to ``extensions.json`` (timestamp, added seeds,
source).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config


def _scan_used_seeds(job_dir: Path) -> set[int]:
	"""Seed integers ever processed: outputs/ + outputs_archive/ zarrs, plus
	seeds/*.{todo,done} so still-pending seeds aren't re-queued."""
	used: set[int] = set()
	for sub in ("outputs", "outputs_archive"):
		d = job_dir / sub
		if not d.exists():
			continue
		for p in d.glob("seed_*_*.zarr"):
			try:
				used.add(int(p.stem.split("_")[1]))
			except (ValueError, IndexError):
				pass
	seeds_dir = job_dir / "seeds"
	if seeds_dir.exists():
		for pattern in ("seed_*.todo", "seed_*.done"):
			for p in seeds_dir.glob(pattern):
				try:
					used.add(int(p.stem.split("_")[1]))
				except (ValueError, IndexError):
					pass
	return used


def _append_extension_log(job_dir: Path, *, added: list[int], source: str) -> None:
	"""Append a record to extensions.json."""
	log_path = job_dir / "extensions.json"
	if log_path.exists():
		try:
			history = json.loads(log_path.read_text())
		except json.JSONDecodeError:
			history = []
	else:
		history = []
	history.append({
		"utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
		"source": source,
		"added_seeds": list(added),
		"count": len(added),
	})
	log_path.write_text(json.dumps(history, indent=2) + "\n")


def extend_job(
	job_dir,
	*,
	add: int | None = None,
	seeds: list[int] | None = None,
) -> list[int]:
	"""Emit additional ``seeds/seed_NNNNNN.todo`` files. Returns emitted seeds.

	Exactly one of ``add`` or ``seeds`` must be provided.
	  add: auto-pick N new seeds starting at max(used)+1.
	  seeds: explicit list; no overlap with already-used seeds.

	Raises ValueError on bad args / no prior batch.
	Raises FileNotFoundError on missing job_dir or *.toml.
	"""
	job_dir = Path(job_dir).resolve()
	if not job_dir.is_dir():
		raise FileNotFoundError(f"job_dir does not exist: {job_dir}")

	if (add is None) == (seeds is None):
		raise ValueError("extend_job requires exactly one of `add` or `seeds`")

	toml_candidates = list(job_dir.glob("*.toml"))
	if not toml_candidates:
		raise FileNotFoundError(f"No *.toml in {job_dir}")
	if len(toml_candidates) > 1:
		raise ValueError(
			f"Expected one *.toml in {job_dir}, found {len(toml_candidates)}: "
			f"{[p.name for p in toml_candidates]}"
		)
	# load_config is invoked for parity with the worker / aggregator entry
	# checks; the schema validation surfaces malformed TOMLs early.
	load_config(toml_candidates[0])

	used = _scan_used_seeds(job_dir)
	if not used:
		raise ValueError(
			f"no prior seeds in {job_dir} (outputs/, outputs_archive/, and "
			"seeds/*.{todo,done} are all empty). Run the initial pipeline first."
		)

	if add is not None:
		if not isinstance(add, int) or add < 1:
			raise ValueError(f"--add must be a positive integer, got {add!r}")
		start = max(used) + 1
		new_seeds = list(range(start, start + add))
		source = "add"
	else:
		if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
			raise ValueError("--seeds must be a list of ints")
		if not seeds:
			raise ValueError("--seeds list is empty")
		if len(set(seeds)) != len(seeds):
			raise ValueError(f"--seeds list has duplicates: {seeds}")
		overlap = sorted(set(seeds) & used)
		if overlap:
			raise ValueError(
				f"--seeds list overlaps with already-used seeds: {overlap}. "
				f"Pick seed integers not in {sorted(used)}."
			)
		new_seeds = list(seeds)
		source = "seeds"

	seeds_dir = job_dir / "seeds"
	seeds_dir.mkdir(parents=True, exist_ok=True)
	for s in new_seeds:
		todo = seeds_dir / f"seed_{s:06d}.todo"
		with todo.open("x") as f:
			f.write(f"{s}\n")

	_append_extension_log(job_dir, added=new_seeds, source=source)
	return new_seeds


def main():
	"""``abtem-run-extend`` console-script entry."""
	parser = argparse.ArgumentParser(
		description=(
			"abtem-run-extend: append more phonon snapshots to an existing "
			"job dir. Emits new seeds/seed_*.todo with non-overlapping integers."
		),
	)
	parser.add_argument(
		"job_dir",
		help="job directory (gen_*/<phase>_<hkl>_<tilt>/)",
	)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument(
		"--add",
		type=int,
		metavar="N",
		help="auto-pick N new seeds starting at max(used)+1",
	)
	group.add_argument(
		"--seeds",
		type=str,
		metavar="A,B,C",
		help="explicit comma-separated seed integers",
	)
	args = parser.parse_args()

	if args.seeds is not None:
		try:
			seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
		except ValueError as e:
			parser.error(f"--seeds must be comma-separated integers: {e}")
		emitted = extend_job(args.job_dir, seeds=seeds)
	else:
		emitted = extend_job(args.job_dir, add=args.add)

	print(f"Emitted {len(emitted)} new seed(s): {emitted}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
