"""
Tests for aggregate_series — cumulative-mean frame series under
``aggregate/n_<k:03d>/``. Each subdir holds the per-channel mean of the
first k seeds (sorted by seed integer).

    PYTHONPATH=src python3 tests/test_aggregate_series.py
    PYTHONPATH=src pytest tests/test_aggregate_series.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import abtem
import numpy as np
import pytest

from abtem_run.aggregate import aggregate_series
from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


def _setup_three_seed_job(tmp: Path) -> Path:
	"""Materialize a job_dir + surf.xyz, then run 3 workers so per-seed
	zarrs exist for the series tests."""
	cif_dir = setup_cif_dir(tmp)
	job_dir = tmp / "test_job"
	job_dir.mkdir()
	(job_dir / "seeds").mkdir()
	(job_dir / "outputs").mkdir()
	write_tiny_config(
		job_dir / "test_job.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=3,
		fph_sigma=0.05,
		phonons_seed=42,
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)
	for seed in (42, 43, 44):
		todo = job_dir / "seeds" / f"seed_{seed:06d}.todo"
		todo.write_text(f"{seed}\n")
		run_one_seed(job_dir, todo)
	return job_dir


def test_aggregate_series_emits_per_n_subdirs():
	"""--n-phonons=3 -> three subdirs aggregate/n_001..n_003 each with
	a haadf.{tif,zarr}."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		n_emitted = aggregate_series(job, n_phonons=3)
		assert n_emitted == 3

		agg = job / "aggregate"
		for k in (1, 2, 3):
			d = agg / f"n_{k:03d}"
			assert d.is_dir(), f"missing n_{k:03d} dir"
			assert (d / "haadf.tif").exists()
			assert (d / "haadf.zarr").exists()
			# default blur_sigmas = [0.025, 0.1, 0.25] -> 3 blurred variants
			assert (d / "haadf_0-025.tif").exists()
			assert (d / "haadf_0-1.tif").exists()
			assert (d / "haadf_0-25.tif").exists()


def test_aggregate_series_n_phonons_caps_to_available(caplog):
	"""Asking for more than the available seeds emits only as many as exist
	AND surfaces the cap via a log.info message — silent caps would lead
	users to think they got more frames than they did."""
	import logging
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		with caplog.at_level(logging.INFO, logger="abtem_run.aggregate"):
			n_emitted = aggregate_series(job, n_phonons=99)
		assert n_emitted == 3  # only 3 seeds were run
		assert any(
			"capped to 3" in rec.getMessage() and "n_phonons=99" in rec.getMessage()
			for rec in caplog.records
		), f"cap log message not emitted; got: {[r.getMessage() for r in caplog.records]}"


def test_aggregate_series_default_uses_all_seeds():
	"""n_phonons=None -> use every seed in outputs/ + outputs_archive/."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		n_emitted = aggregate_series(job)
		assert n_emitted == 3


def test_aggregate_series_n1_equals_first_seed():
	"""n_<001>/haadf should equal the per-seed zarr for the smallest seed
	BIT-FOR-BIT — the mean of one element is the element itself, no FP
	round-off is acceptable here."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		aggregate_series(job, n_phonons=3)

		n1 = abtem.from_zarr(str(job / "aggregate" / "n_001" / "haadf.zarr")).array
		seed42 = abtem.from_zarr(str(job / "outputs" / "seed_000042_haadf.zarr")).array
		n1_arr = np.asarray(n1.compute() if hasattr(n1, "compute") else n1)
		s42_arr = np.asarray(seed42.compute() if hasattr(seed42, "compute") else seed42)
		assert np.array_equal(n1_arr, s42_arr), "n_001 mean diverged from seed_000042"


def test_aggregate_series_n3_matches_three_seed_mean():
	"""n_<003>/haadf should equal the mean of all three seeds."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		aggregate_series(job, n_phonons=3)

		n3 = abtem.from_zarr(str(job / "aggregate" / "n_003" / "haadf.zarr")).array
		raw = []
		for seed in (42, 43, 44):
			m = abtem.from_zarr(str(job / "outputs" / f"seed_{seed:06d}_haadf.zarr")).array
			arr = np.asarray(m.compute() if hasattr(m, "compute") else m)
			raw.append(arr)
		expected = np.mean(raw, axis=0)
		n3_arr = np.asarray(n3.compute() if hasattr(n3, "compute") else n3)
		assert np.allclose(n3_arr, expected), "n_003 mean diverged from manual 3-seed mean"


def test_aggregate_series_rejects_invalid_n_phonons():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		with pytest.raises(ValueError, match=">= 1"):
			aggregate_series(job, n_phonons=0)


def test_aggregate_series_static_block_runs_once_at_agg_root():
	"""projection preview goes at <job_dir>/aggregate/, not under n_<k>/."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_three_seed_job(Path(tmp).resolve())
		aggregate_series(job, n_phonons=2)

		agg = job / "aggregate"
		assert (agg / "potential_projection.png").exists()
		# NOT under the per-N subdirs
		assert not (agg / "n_001" / "potential_projection.png").exists()
		assert not (agg / "n_002" / "potential_projection.png").exists()
