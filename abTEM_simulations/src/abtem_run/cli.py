#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Convenience wrapper that drives the worker pipeline in-process:

    generate  ->  for each .todo:  run_one_seed  ->  aggregate

Same effective behavior as running ``abtem-run-generate``, then a bash
loop over ``abtem-run-worker``, then ``abtem-run-aggregate`` — but in
one Python process, no orchestration. For sweeps that need parallel
workers (slurm, GNU parallel, multiple GPUs), use the lower-level
console scripts directly.

CLI:
    abtem-run                              # uses ./config.toml
    abtem-run --config my_config.toml
    abtem-run --generate-only              # plan + planning artifacts, no GPU
"""

import argparse
import sys
from pathlib import Path

from .aggregate import aggregate_job
from .config import load_config
from .generator_run import generate_run
from .worker import run_one_seed


__all__ = ["main", "run_pipeline"]


def run_pipeline(config_path, *, generate_only: bool = False) -> Path:
	"""Library entry point for the in-process pipeline.

	Args:
		config_path: path to the TOML config (absolute or CWD-relative).
		generate_only: if True, run the generator and stop (no workers,
		               no aggregator). Equivalent in effect to setting
		               ``simulations.dry_run = true`` in the config.

	Returns:
		Path to the generated ``gen_<UTC>/`` run directory.

	Behavior:
		1. Read the config (honoring ``simulations.dry_run``).
		2. Call ``generate_run(config_path)`` — emits the job tree with
		   per-job TOMLs, planning artifacts (surf.xyz + combined.png),
		   and per-seed .todo files.
		3. If ``generate_only`` or ``cfg.simulations.dry_run``: return.
		4. For each job dir: iterate ``seeds/*.todo`` in order, calling
		   ``run_one_seed`` for each. Workers atomically rename
		   .todo -> .done.
		5. For each job dir: call ``aggregate_job`` to mean per-seed
		   outputs into ``aggregate/`` (and clean up ``outputs/`` unless
		   ``simulations.test_enabled`` is set).
	"""
	cfg = load_config(config_path)

	print(f"abtem-run: generating queue from {config_path}")
	run_dir = generate_run(config_path)
	print(f"abtem-run: queue at {run_dir}")

	if generate_only or cfg.simulations.dry_run:
		reason = "--generate-only" if generate_only else "simulations.dry_run=true"
		print(f"abtem-run: stopping after generation ({reason}).")
		return run_dir

	job_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
	print(f"abtem-run: {len(job_dirs)} job(s) to process")

	for job_dir in job_dirs:
		todos = sorted((job_dir / "seeds").glob("*.todo"))
		print(f"abtem-run: [{job_dir.name}] {len(todos)} seed(s)")
		for todo in todos:
			print(f"abtem-run: [{job_dir.name}]   {todo.name}")
			run_one_seed(job_dir, todo)
		print(f"abtem-run: [{job_dir.name}] aggregating")
		aggregate_job(job_dir)

	print("abtem-run: finished")
	return run_dir


def main():
	"""``abtem-run`` console-script entry."""
	parser = argparse.ArgumentParser(
		description=(
			"abtem-run: in-process pipeline driver. "
			"Generates the per-seed work queue from the TOML config, then "
			"runs all workers serially and aggregates each job. "
			"For parallel execution, call abtem-run-worker / "
			"abtem-run-aggregate directly."
		),
	)
	parser.add_argument(
		"--config",
		default="config.toml",
		help="TOML config file (default: config.toml in CWD)",
	)
	parser.add_argument(
		"--generate-only",
		action="store_true",
		help=(
			"plan + emit planning artifacts only; skip workers and "
			"aggregation. Same effect as simulations.dry_run=true."
		),
	)
	args = parser.parse_args()
	run_pipeline(args.config, generate_only=args.generate_only)
	return 0


if __name__ == "__main__":
	sys.exit(main())
