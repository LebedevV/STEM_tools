#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Pre-flight cost estimator: counts jobs / seeds / multislice calls / scan
positions from a config, no GPU work. No wall-time estimate (no hardware
calibration) — structural counts only.

Used by ``cli.run_pipeline``; skip with ``--no-estimate`` or
``ABTEM_RUN_NO_ESTIMATE=1``.
"""

import logging
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobCost:
	"""Per-job structural counts. Multislice total = n_seeds × {scan, diff,
	cbed}. ``emit_static_baseline`` adds only a projection of the cached static
	potential, not a multislice scan, so it carries no multislice cost."""
	scan_per_seed: int          # 0/1
	diffraction_per_seed: int   # 0/1
	cbed_per_seed: int          # 0/1
	scan_positions: int
	n_detectors: int
	n_seeds: int                # frozen_phonons (1 if 'None')
	thickness_a: float

	@property
	def per_seed_multislices(self) -> int:
		return self.scan_per_seed + self.diffraction_per_seed + self.cbed_per_seed

	@property
	def total_multislices(self) -> int:
		return self.n_seeds * self.per_seed_multislices


@dataclass(frozen=True)
class RunCost:
	"""Aggregate cost across expanded (phase, hkl, tilt, sweep-axes) jobs."""
	n_jobs: int
	per_job: list[JobCost]

	@property
	def total_seeds(self) -> int:
		return sum(j.n_seeds for j in self.per_job)

	@property
	def total_multislices(self) -> int:
		return sum(j.total_multislices for j in self.per_job)

	@property
	def total_scan_position_evaluations(self) -> int:
		"""Sum over jobs of (n_scan_multislices × scan_positions) — a
		single-number proxy for wall-time on a given GPU + lamella thickness."""
		return sum(j.n_seeds * j.scan_per_seed * j.scan_positions for j in self.per_job)


def _estimate_scan_positions(cfg) -> int:
	"""Scan-grid size matching worker._run_scan: override_sampling if set,
	else 0.9 × abtem's nyquist_sampling, imported (not re-derived) so the scan
	grid stays a single source of truth shared with the worker.
	"""
	sim = cfg.simulations
	scan_s = cfg.lamella_settings.scan_s
	if sim.override_sampling and not isinstance(sim.override_sampling, bool):
		sampling = float(sim.override_sampling)
	else:
		from abtem.transfer import nyquist_sampling
		sampling = 0.9 * nyquist_sampling(
			float(cfg.microscope.convergence_angle), float(cfg.microscope.HT_value))
	if sampling <= 0:
		return 0
	per_axis = max(1, int(scan_s / sampling))
	return per_axis * per_axis


def estimate_run_cost(cfg) -> RunCost:
	"""Cost across all expand_cfg jobs. Lazy-imports pipeline so the estimator
	stays usable without triggering abtem monkey-patches."""
	from .pipeline import expand_cfg

	per_job: list[JobCost] = []
	for cfg_run in expand_cfg(cfg):
		sim = cfg_run.simulations
		mic = cfg_run.microscope
		# frozen_phonons: int >= 1, or the string 'None' (single static run).
		fp = sim.frozen_phonons
		if isinstance(fp, str) and fp == "None":
			n_seeds = 1
		else:
			n_seeds = int(fp)
		# The generator emits one job dir per (phase, hkl, tilt). The tilt is
		# scalar here (expand_cfg yields per-tilt); multiply by the remaining
		# phase × hkl axes.
		n_phases = len(cfg_run.job.phase_list)
		n_hkl = len(cfg_run.job.hkl_list)
		# Mirror worker.run_one_seed's scan gate so the estimator doesn't
		# overcount multislices the runtime would skip.
		scan_runs = bool(sim.do_full_run and mic.detectors)
		for _ in range(n_phases * n_hkl):
			per_job.append(JobCost(
				scan_per_seed=1 if scan_runs else 0,
				diffraction_per_seed=1 if mic.do_diffraction else 0,
				cbed_per_seed=1 if mic.do_cbed else 0,
				scan_positions=_estimate_scan_positions(cfg_run),
				n_detectors=len(mic.detectors),
				n_seeds=n_seeds,
				thickness_a=float(cfg_run.lamella_settings.thickness),
			))
	return RunCost(n_jobs=len(per_job), per_job=per_job)


def format_run_cost(cost: RunCost) -> str:
	"""Multi-line human-readable RunCost block."""
	lines = []
	lines.append("=" * 64)
	lines.append("abtem-run: pre-flight cost estimate")
	lines.append("-" * 64)
	lines.append(f"  jobs (phase × hkl × tilt × sweep-axes): {cost.n_jobs}")
	lines.append(f"  total seeds across all jobs:            {cost.total_seeds}")
	lines.append(f"  total multislice calls:                 {cost.total_multislices}")
	if cost.total_scan_position_evaluations > 0:
		lines.append(
			f"  total scan-position evaluations:        "
			f"{cost.total_scan_position_evaluations:_} "
			f"(scan-only proxy for wall-time)"
		)

	if cost.per_job:
		ref = cost.per_job[0]
		lines.append("-" * 64)
		lines.append("  per-job breakdown (showing job 0; others sweep params):")
		lines.append(f"    n_seeds (frozen_phonons):     {ref.n_seeds}")
		lines.append(f"    multislices per seed:         {ref.per_seed_multislices}")
		if ref.scan_per_seed:
			lines.append(f"      scan: 1 (detectors={ref.n_detectors}, ~{ref.scan_positions:_} positions)")
		if ref.diffraction_per_seed:
			lines.append("      diff: 1 (plane-wave, single multislice)")
		if ref.cbed_per_seed:
			lines.append("      cbed: 1 (probe-at-center, single multislice)")
		lines.append(f"    lamella thickness (Å):        {ref.thickness_a:.1f}")

	lines.append("=" * 64)
	return "\n".join(lines)
