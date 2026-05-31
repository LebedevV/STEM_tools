"""
Tests for the simulations.blur_sigmas config field (post-v0.1.2: BLUR_SIGMAS
moved off the module level and into the config).

    PYTHONPATH=src python3 tests/test_blur_sigmas.py
    PYTHONPATH=src pytest tests/test_blur_sigmas.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

import abtem_run  # noqa: F401
from abtem_run.aggregate import aggregate_job
from abtem_run.config import Simulations
from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


def _sim(**overrides):
	defaults = dict(
		override_sampling=False,
		frozen_phonons=2,
		fph_sigma=0.05,
		do_full_run=True,
	)
	defaults.update(overrides)
	return Simulations(**defaults)


def test_default_blur_sigmas_matches_legacy_set():
	"""Default mirrors the pre-config BLUR_SIGMAS constant so existing configs
	keep producing the same blur previews."""
	s = _sim()
	assert s.blur_sigmas == [0.025, 0.1, 0.25]


def test_blur_sigmas_accepts_custom_list():
	s = _sim(blur_sigmas=[0.5, 1.0, 2.0])
	assert s.blur_sigmas == [0.5, 1.0, 2.0]


def test_blur_sigmas_accepts_empty_list_for_no_blurs():
	"""[] disables blur previews entirely."""
	s = _sim(blur_sigmas=[])
	assert s.blur_sigmas == []


def test_blur_sigmas_accepts_zero():
	"""0.0 is degenerate but allowed (no-op blur); the validator only
	rejects negatives + non-numerics."""
	s = _sim(blur_sigmas=[0.0, 0.5])
	assert s.blur_sigmas == [0.0, 0.5]


def test_blur_sigmas_rejects_negative():
	with pytest.raises(ValidationError, match=">= 0"):
		_sim(blur_sigmas=[0.1, -0.5])


def test_blur_sigmas_rejects_non_list():
	with pytest.raises(ValidationError, match="must be a list"):
		_sim(blur_sigmas=0.5)


def test_blur_sigmas_rejects_bool_entries():
	"""bool is a subclass of int — guard against True/False slipping in."""
	with pytest.raises(ValidationError, match="numeric"):
		_sim(blur_sigmas=[0.1, True])


def test_blur_sigmas_rejects_non_numeric_entries():
	with pytest.raises(ValidationError, match="numeric"):
		_sim(blur_sigmas=[0.1, "0.5"])


# --------------------------------------------------------------------------- #
# End-to-end: custom blur_sigmas in TOML -> exact blur TIFF filenames emitted
# --------------------------------------------------------------------------- #


def _setup_single_seed_job(tmp: Path, *, blur_sigmas):
	cif_dir = setup_cif_dir(tmp)
	job_dir = tmp / "test_job"
	job_dir.mkdir()
	(job_dir / "seeds").mkdir()
	(job_dir / "outputs").mkdir()
	write_tiny_config(
		job_dir / "test_job.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=1,
		fph_sigma=0.05,
		phonons_seed=42,
		blur_sigmas=blur_sigmas,
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)
	todo = job_dir / "seeds" / "seed_000042.todo"
	todo.write_text("42\n")
	run_one_seed(job_dir, todo)
	return job_dir


def test_custom_blur_sigmas_emit_exactly_those_tiffs():
	"""TOML blur_sigmas=[0.5] -> exactly one blur TIFF (_0-5.tif), none
	from the default set. Regression guard against blur_sigmas being
	dropped on the way from config -> RunContext -> aggregator."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_single_seed_job(Path(tmp).resolve(), blur_sigmas=[0.5])
		aggregate_job(job)

		agg = job / "aggregate"
		# Custom sigma present
		assert (agg / "haadf_0-5.tif").exists(), "custom sigma 0.5 missing"
		# Default sigmas absent
		assert not (agg / "haadf_0-025.tif").exists(), "default 0.025 leaked through"
		assert not (agg / "haadf_0-1.tif").exists(), "default 0.1 leaked through"
		assert not (agg / "haadf_0-25.tif").exists(), "default 0.25 leaked through"


def test_empty_blur_sigmas_skips_blur_emission():
	"""TOML blur_sigmas=[] -> base haadf.{tif,zarr} present, no blur variants."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_single_seed_job(Path(tmp).resolve(), blur_sigmas=[])
		aggregate_job(job)

		agg = job / "aggregate"
		# Base mean files are still emitted
		assert (agg / "haadf.tif").exists()
		assert (agg / "haadf.zarr").exists()
		# But no blur variants
		blur_tiffs = list(agg.glob("haadf_*.tif"))
		# allow haadf_static.tif if static-baseline ever sneaks in, but
		# the tiny config doesn't enable it
		blur_tiffs = [p for p in blur_tiffs if "_static" not in p.name]
		assert not blur_tiffs, f"blur_sigmas=[] produced blur TIFFs: {blur_tiffs}"
