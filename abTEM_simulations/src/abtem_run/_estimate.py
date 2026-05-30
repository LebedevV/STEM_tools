"""
Pre-flight cost estimator for ``abtem-run``.

Given a resolved ``AppConfig``, compute how much multislice work the
pipeline is about to commit to: number of jobs, number of seeds per
job, number of scan positions per multislice, etc. Doesn't pretend to
give a wall-time estimate (would need hardware calibration), but
prints enough structural info that the user can extrapolate from a
small benchmark to a full run before kicking off.

Surfaces the "12-hour run, killed at 1h" failure mode pre-emptively:
hard to misjudge a run when the estimator says "8 jobs × 16 seeds ×
~15600 scan positions per multislice = ~2 million scan positions, plus
8 static-baseline scans".

Used by ``cli.run_pipeline`` before generator + workers. Skip with
``--no-estimate`` or the env var ``ABTEM_RUN_NO_ESTIMATE=1``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JobCost:
	"""Per-job cost breakdown.

	Multislice counts are per-seed (so total = n_seeds × {scan, diff,
	cbed}) plus the one-off static-baseline (per job, not per seed).
	"""
	scan_per_seed: int          # 0 or 1 (do_full_run)
	diffraction_per_seed: int   # 0 or 1 (do_diffraction)
	cbed_per_seed: int          # 0 or 1 (do_cbed)
	static_baseline: int        # 0 or 1 (emit_static_baseline)
	scan_positions: int         # estimated scan grid size for this lamella
	n_detectors: int            # configured scan detectors
	n_seeds: int                # frozen_phonons (or 1 for "no phonons")
	thickness_a: float          # lamella thickness for time scaling

	@property
	def per_seed_multislices(self) -> int:
		return self.scan_per_seed + self.diffraction_per_seed + self.cbed_per_seed

	@property
	def total_multislices(self) -> int:
		"""Total multislice calls for this job (worker + aggregator side)."""
		return self.n_seeds * self.per_seed_multislices + self.static_baseline


@dataclass(frozen=True)
class RunCost:
	"""Aggregate cost across all expanded (phase, hkl, tilt) jobs."""
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
		"""Sum over jobs of (n_scan_multislices × scan_positions). Useful
		single-number proxy for total wall-time; per-position multislice
		cost is roughly constant for a given lamella thickness on a
		given GPU."""
		return sum(
			(j.n_seeds * j.scan_per_seed + j.static_baseline) * j.scan_positions
			for j in self.per_job
		)


def _estimate_scan_positions(cfg) -> int:
	"""Approximate number of scan positions per scan multislice.

	Matches the runtime logic in worker._run_scan: use override_sampling
	if set (float), else probe.ctf.nyquist_sampling * 0.9. We don't
	build a Probe here (slow + requires abtem import); instead compute
	nyquist directly: nyquist = λ / (2 × semiangle), with
	λ = h / sqrt(2*m_e*e*V*(1 + eV/(2*m_e*c²))) — abtem's
	energy2wavelength formula.

	Returns floor(scan_s / sampling)² — integer scan grid size.
	"""
	sim = cfg.simulations
	scan_s = cfg.lamella_settings.scan_s
	if sim.override_sampling and not isinstance(sim.override_sampling, bool):
		sampling = float(sim.override_sampling)
	else:
		# abtem.energy2wavelength (verified against abtem.transfer):
		# λ [Å] = h / sqrt(2 m_e e V (1 + eV/(2 m_e c²))) × 1e10
		# constants in SI
		h = 6.62607015e-34
		m_e = 9.1093837015e-31
		e = 1.602176634e-19
		c = 299792458.0
		V = float(cfg.microscope.HT_value)
		lam_m = h / math.sqrt(2 * m_e * e * V * (1 + e * V / (2 * m_e * c * c)))
		lam_A = lam_m * 1e10
		semiangle_rad = float(cfg.microscope.convergence_angle) * 1e-3
		nyquist = lam_A / (2.0 * semiangle_rad)
		sampling = nyquist * 0.9
	if sampling <= 0:
		return 0
	per_axis = max(1, int(scan_s / sampling))
	return per_axis * per_axis


def estimate_run_cost(cfg) -> RunCost:
	"""Given a base ``AppConfig`` (pre-expand_cfg), compute the cost
	across all expanded (phase, hkl, tilt, sweep-axes) jobs.

	Imports ``expand_cfg`` from ``.pipeline`` lazily because pipeline
	pulls in abtem; we want the estimator to be importable without
	triggering the abtem monkey-patches when used as a library
	pre-flight tool (e.g. from a notebook).
	"""
	from .pipeline import expand_cfg

	per_job: list[JobCost] = []
	for cfg_run in expand_cfg(cfg):
		sim = cfg_run.simulations
		mic = cfg_run.microscope
		# Per the existing pipeline: frozen_phonons can be int >= 1 or the
		# string 'None'. 'None' means a single static-lattice run.
		fp = sim.frozen_phonons
		if isinstance(fp, str) and fp == "None":
			n_seeds = 1
		else:
			n_seeds = int(fp)
		# v6's generator emits one job dir per (phase, hkl, tilt). The tilt
		# is scalar here (expand_cfg yields per-tilt); multiply by
		# len(phase_list) × len(hkl_list) for the remaining axes.
		n_phases = len(cfg_run.job.phase_list)
		n_hkl = len(cfg_run.job.hkl_list)
		for _ in range(n_phases * n_hkl):
			per_job.append(JobCost(
				scan_per_seed=1 if sim.do_full_run else 0,
				diffraction_per_seed=1 if mic.do_diffraction else 0,
				cbed_per_seed=1 if mic.do_cbed else 0,
				static_baseline=1 if sim.emit_static_baseline else 0,
				scan_positions=_estimate_scan_positions(cfg_run),
				n_detectors=len(mic.detectors),
				n_seeds=n_seeds,
				thickness_a=float(cfg_run.lamella_settings.thickness),
			))
	return RunCost(n_jobs=len(per_job), per_job=per_job)


def format_run_cost(cost: RunCost) -> str:
	"""Render a RunCost as a human-readable multi-line block. Returned
	as a single string so the caller can log it or print it as fits."""
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
		if ref.static_baseline:
			lines.append(f"    static_baseline:              1 (once per job, "
			             f"detectors={ref.n_detectors}, ~{ref.scan_positions:_} positions)")
		lines.append(f"    lamella thickness (Å):        {ref.thickness_a:.1f}")

	lines.append("=" * 64)
	return "\n".join(lines)
