#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Append non-overlapping phonon seeds to an existing job directory.

Writes new ``seeds/seed_*.todo`` files and appends the action to
``extensions.json``. Refuses empty jobs so extension always means cumulative
refinement of an existing run.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .job_io import load_job_config, scan_used_seeds, write_seed_todo


def _append_extension_log(job_dir: Path, *, added: list[int], source: str) -> None:
	"""Append a record to extensions.json (atomic tmp + os.replace so a
	crash mid-write doesn't truncate the history)."""
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
	tmp = log_path.with_suffix(log_path.suffix + ".tmp")
	tmp.write_text(json.dumps(history, indent=2) + "\n")
	os.replace(tmp, log_path)


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

	# Load for parity with the worker / aggregator entry checks; schema
	# validation surfaces malformed TOMLs early.
	load_job_config(job_dir)

	used = scan_used_seeds(job_dir)
	if not used:
		raise ValueError(
			f"no prior seeds in {job_dir} (outputs/, outputs_archive/, and "
			"seeds/*.{todo,running,done} are all empty). Run the initial pipeline first."
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
	for s in new_seeds:
		write_seed_todo(seeds_dir, s)

	_append_extension_log(job_dir, added=new_seeds, source=source)
	return new_seeds


def main():
	"""Module entry point."""
	parser = argparse.ArgumentParser(
		description="Append non-overlapping seed todos to an existing job."
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
