#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""Local source-tree driver: generate jobs, run pending seeds, aggregate."""

import argparse
import logging
import os
import sys
from pathlib import Path

from ._log import configure_default_logging
from .aggregate import aggregate_job, aggregate_series
from .compat import ensure_patched_environment
from .config import load_config
from .generator_run import generate_run
from .worker import run_one_seed


log = logging.getLogger(__name__)


__all__ = ["main", "run_pipeline"]


def run_pipeline(
	config_path=None,
	*,
	generate_only: bool = False,
	resume_dir=None,
	force_new: bool = False,
	show_estimate: bool = True,
) -> Path:
	"""Run the serial local pipeline or resume an existing run directory."""
	# Exactly one of config_path / resume_dir is required.
	if resume_dir is None:
		if config_path is None:
			raise ValueError("Either config_path or resume_dir must be provided")
	elif generate_only:
		raise ValueError("generate_only cannot be combined with resume_dir")

	if resume_dir is None:
		# Pre-flight cost estimate (logged at INFO; silence with
		# show_estimate=False or env ABTEM_RUN_NO_ESTIMATE=1).
		if show_estimate and not os.environ.get("ABTEM_RUN_NO_ESTIMATE"):
			from ._estimate import estimate_run_cost, format_run_cost
			log.info(format_run_cost(estimate_run_cost(load_config(config_path))))
		log.info(f"abtem_run: generating queue from {config_path}")
		run_dir = generate_run(config_path)
		log.info(f"abtem_run: queue at {run_dir}")
		if generate_only:
			log.info("abtem_run: stopping after generation (--generate-only).")
			log.info("Next step: inspect the generated combined.png files, then run:")
			log.info(f"  python run.py --resume {run_dir}")
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
		log.info(f"abtem_run: resuming {run_dir}")

	job_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
	log.info(f"abtem_run: {len(job_dirs)} job(s) to process")

	for job_dir in job_dirs:
		todos = sorted((job_dir / "seeds").glob("*.todo"))
		if todos:
			log.info(f"abtem_run: [{job_dir.name}] {len(todos)} seed(s) to run")
			for todo in todos:
				log.info(f"abtem_run: [{job_dir.name}]   {todo.name}")
				run_one_seed(job_dir, todo)
		else:
			log.info(f"abtem_run: [{job_dir.name}] all seeds done")

		# If outputs/ is gone, a previous aggregator already archived it —
		# nothing fresh to re-aggregate (use --aggregate to force a rebuild).
		if not (job_dir / "outputs").exists():
			log.info(f"abtem_run: [{job_dir.name}] outputs/ missing — aggregate already run; skipping")
			continue
		log.info(f"abtem_run: [{job_dir.name}] aggregating")
		aggregate_job(job_dir, force_new=force_new)

	log.info("abtem_run: finished")
	return run_dir


def _resolve_patch_consent(args) -> bool | None:
	"""Map the patch flags / env var to ensure_patched_environment's assume_yes:
	explicit decline, explicit consent, or None (decide at runtime)."""
	if args.no_patches:
		return False
	if args.apply_patches or os.environ.get("ABTEM_RUN_APPLY_PATCHES"):
		return True
	return None


def main():
	"""Source-tree entry point used by ``python run.py``."""
	configure_default_logging()
	parser = argparse.ArgumentParser(
		prog="python run.py",
		description=(
			"Local serial driver for abtem_run. "
			"Generates the per-seed work queue from the TOML config, then "
			"runs all workers serially and aggregates each job. "
			"For parallel execution, run `abtem-run-worker` / `abtem-run-aggregate` "
			"from the deployment image, whose abTEM is patched at build."
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
			"aggregate one job dir (same effect as "
			"`python -m abtem_run.aggregate <JOB_DIR>`; no generator, no workers). "
			"Useful when workers ran out-of-band."
		),
	)
	parser.add_argument(
		"--aggregate-series",
		default=None,
		metavar="JOB_DIR",
		help=(
			"emit cumulative-mean frames at <vdir>/series/n_<k:03d>/ for "
			"k in 1..N (N = --n-phonons or all available seeds), where <vdir> is "
			"the aggregate version dir; for visualising 1/sqrt(N) convergence."
		),
	)
	parser.add_argument(
		"--n-phonons",
		type=int,
		default=None,
		metavar="N",
		help="cap N for --aggregate-series (default: all available seeds).",
	)
	parser.add_argument(
		"--force-new",
		action="store_true",
		help=(
			"force a fresh aggregate version dir (aggregate/<UTC>_<hash>/) instead "
			"of rediscovering and reusing the newest one matching the config hash. "
			"Applies to --aggregate / --aggregate-series and the resume re-aggregate."
		),
	)
	parser.add_argument(
		"--no-estimate",
		action="store_true",
		help=(
			"suppress the pre-flight cost estimate before the generator runs "
			"(also suppressed by ABTEM_RUN_NO_ESTIMATE=1). No effect on --resume."
		),
	)
	patch_group = parser.add_mutually_exclusive_group()
	patch_group.add_argument(
		"--apply-patches",
		action="store_true",
		help=(
			"apply abTEM compat shims without prompting (also via "
			"ABTEM_RUN_APPLY_PATCHES=1); for headless / CI / pipeline runs."
		),
	)
	patch_group.add_argument(
		"--no-patches",
		action="store_true",
		help="run against the bare abTEM environment; never apply compat shims.",
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

	ensure_patched_environment(assume_yes=_resolve_patch_consent(args))

	if args.aggregate is not None:
		aggregate_job(args.aggregate, force_new=args.force_new)
	elif args.aggregate_series is not None:
		n_emitted = aggregate_series(args.aggregate_series, n_phonons=args.n_phonons, force_new=args.force_new)
		log.info(f"abtem_run: emitted {n_emitted} series/n_<k>/ frame(s)")
	elif args.resume is not None:
		if args.generate_only:
			parser.error("--generate-only cannot be combined with --resume")
		run_pipeline(resume_dir=args.resume, force_new=args.force_new)
	else:
		run_pipeline(args.config, generate_only=args.generate_only, force_new=args.force_new, show_estimate=not args.no_estimate)
	return 0


if __name__ == "__main__":
	sys.exit(main())
