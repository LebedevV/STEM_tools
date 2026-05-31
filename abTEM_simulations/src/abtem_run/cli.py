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
    abtem-run --resume gen_<UTC>           # finish a partially-run sweep
"""

import argparse
import sys
from pathlib import Path

from .aggregate import aggregate_job, aggregate_series
from .generator_run import generate_run
from .worker import run_one_seed


__all__ = ["main", "run_pipeline"]


def run_pipeline(
	config_path=None,
	*,
	generate_only: bool = False,
	resume_dir=None,
) -> Path:
	"""Library entry point for the in-process pipeline.

	Args:
		config_path: path to the TOML config (absolute or CWD-relative).
		             Required unless ``resume_dir`` is given; ignored when it is.
		generate_only: if True, run the generator and stop (no workers, no
		               aggregator). Cannot be combined with ``resume_dir``.
		resume_dir: path to an existing ``gen_<UTC>/`` run directory. Skips the
		            generator, picks up any remaining ``seeds/*.todo``, and
		            aggregates each job. Idempotent — a safe no-op on a
		            fully-complete sweep.

	Returns:
		Path to the run directory (newly generated, or the resumed one).

	Behavior:
		1. With ``resume_dir``: pick up the existing run directory (skips the
		   generator). Otherwise: call ``generate_run(config_path)`` — reads
		   + validates the config and emits the job tree with per-job TOMLs,
		   planning artifacts (surf.xyz + combined.png), and per-seed .todo
		   files.
		2. If ``generate_only``: return.
		3. For each job dir: iterate ``seeds/*.todo`` in order, calling
		   ``run_one_seed`` for each. Workers atomically rename
		   .todo -> .done.
		4. For each job dir: call ``aggregate_job`` to mean per-seed outputs
		   into ``aggregate/``. If outputs/ is already archived (a prior
		   aggregator ran), skip — use ``--aggregate`` to force a rebuild.
	"""
	# Exactly one of config_path / resume_dir is required.
	if resume_dir is None:
		if config_path is None:
			raise ValueError("Either config_path or resume_dir must be provided")
	elif generate_only:
		raise ValueError("generate_only cannot be combined with resume_dir")

	if resume_dir is None:
		print(f"abtem-run: generating queue from {config_path}")
		run_dir = generate_run(config_path)
		print(f"abtem-run: queue at {run_dir}")
		if generate_only:
			print("abtem-run: stopping after generation (--generate-only).")
			return run_dir
	else:
		run_dir = Path(resume_dir).resolve()
		if not run_dir.is_dir():
			raise FileNotFoundError(f"resume_dir not a directory: {run_dir}")
		job_candidates = [p for p in run_dir.iterdir() if p.is_dir() and (p / "seeds").exists()]
		if not job_candidates:
			raise ValueError(
				f"No job directories (subdir/seeds/) under {run_dir}. "
				"Is this really a 'gen_<UTC>/' directory?"
			)
		print(f"abtem-run: resuming {run_dir}")

	job_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
	print(f"abtem-run: {len(job_dirs)} job(s) to process")

	for job_dir in job_dirs:
		todos = sorted((job_dir / "seeds").glob("*.todo"))
		if todos:
			print(f"abtem-run: [{job_dir.name}] {len(todos)} seed(s) to run")
			for todo in todos:
				print(f"abtem-run: [{job_dir.name}]   {todo.name}")
				run_one_seed(job_dir, todo)
		else:
			print(f"abtem-run: [{job_dir.name}] all seeds done")

		# If outputs/ is gone, a previous aggregator already archived it —
		# nothing fresh to re-aggregate (use --aggregate to force a rebuild).
		if not (job_dir / "outputs").exists():
			print(f"abtem-run: [{job_dir.name}] outputs/ missing — aggregate already run; skipping")
			continue
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
		help="plan + emit planning artifacts only; skip workers and aggregation.",
	)
	parser.add_argument(
		"--resume",
		default=None,
		metavar="RUN_DIR",
		help=(
			"finish a partially-run sweep at the given gen_<UTC>/ directory. "
			"Skips the generator; picks up remaining .todo files and aggregates "
			"each job. Idempotent. Cannot be combined with --generate-only."
		),
	)
	parser.add_argument(
		"--aggregate",
		default=None,
		metavar="JOB_DIR",
		help=(
			"shorthand for `abtem-run-aggregate <JOB_DIR>`: aggregate one job "
			"dir (no generator, no workers). Useful when workers ran out-of-band."
		),
	)
	parser.add_argument(
		"--aggregate-series",
		default=None,
		metavar="JOB_DIR",
		help=(
			"emit cumulative-mean frames at <JOB_DIR>/aggregate/n_<k:03d>/ for "
			"k in 1..N (N = --n-phonons or all available seeds); for visualising "
			"1/sqrt(N) convergence."
		),
	)
	parser.add_argument(
		"--n-phonons",
		type=int,
		default=None,
		metavar="N",
		help="cap N for --aggregate-series (default: all available seeds).",
	)
	args = parser.parse_args()

	# Standalone aggregate modes are mutually exclusive with the pipeline
	# and with each other.
	aggregate_modes = sum(x is not None for x in (args.aggregate, args.aggregate_series))
	if aggregate_modes > 1:
		parser.error("--aggregate and --aggregate-series are mutually exclusive")
	if aggregate_modes == 1 and (args.resume is not None or args.generate_only):
		parser.error("--aggregate / --aggregate-series cannot be combined with --resume or --generate-only")
	if args.n_phonons is not None and args.aggregate_series is None:
		parser.error("--n-phonons only applies to --aggregate-series")

	if args.aggregate is not None:
		aggregate_job(args.aggregate)
	elif args.aggregate_series is not None:
		n_emitted = aggregate_series(args.aggregate_series, n_phonons=args.n_phonons)
		print(f"abtem-run: emitted {n_emitted} aggregate/n_<k>/ frame(s)")
	elif args.resume is not None:
		if args.generate_only:
			parser.error("--generate-only cannot be combined with --resume")
		run_pipeline(resume_dir=args.resume)
	else:
		run_pipeline(args.config, generate_only=args.generate_only)
	return 0


if __name__ == "__main__":
	sys.exit(main())
