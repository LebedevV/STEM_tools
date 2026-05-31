"""
Tests for the abTEM cross-compatibility bridge (``to_ensemble.py``).

Closes architecture-intent open item #3: the v6 per-seed zarr structure
must be alignable with abtem's idiomatic single-Measurement-with-
ensemble-axis representation. The bridge stacks per-seed zarrs into one
abtem Measurement with ``FrozenPhononsAxis(_ensemble_mean=True)`` so
``.reduce_ensemble()`` produces the thermal average — verified to match
the regular aggregator's output to floating-point precision.

Slow (~25-40s): one 2-seed multislice to produce real per-seed zarrs
to bridge from. Run via:

    PYTHONPATH=src python3 tests/test_to_ensemble.py
    PYTHONPATH=src pytest tests/test_to_ensemble.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import abtem
import numpy as np
import pytest

import abtem_run  # noqa: F401  — registers patches
from abtem_run.aggregate import aggregate_job
from abtem_run.to_ensemble import (
	_discover_channels,
	load_ensemble,
	to_ensemble_files,
)
from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


# --------------------------------------------------------------------------- #
# Setup helper — 2-seed job with all three scan detectors
# --------------------------------------------------------------------------- #


def _setup_two_seed_haadf_abf_bf(tmp: Path) -> Path:
	cif_dir = setup_cif_dir(tmp)
	job_dir = tmp / "test_job"
	job_dir.mkdir()
	(job_dir / "seeds").mkdir()
	(job_dir / "outputs").mkdir()
	write_tiny_config(
		job_dir / "test_job.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=2,
		fph_sigma=0.05,
		phonons_seed=42,
		detectors=("haadf", "abf", "bf"),
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)
	for seed in (42, 43):
		todo = job_dir / "seeds" / f"seed_{seed:06d}.todo"
		todo.write_text(f"{seed}\n")
		run_one_seed(job_dir, todo)
	return job_dir


# --------------------------------------------------------------------------- #
# _discover_channels
# --------------------------------------------------------------------------- #


def test_discover_channels_empty_job():
	with tempfile.TemporaryDirectory() as tmp:
		# fresh dir, no outputs/ or archive — discover returns empty.
		assert _discover_channels(Path(tmp)) == []


def test_discover_channels_finds_outputs_and_archive(tmp_path):
	# Hand-roll structure without running multislice: just touch the zarr
	# dirs so the glob picks them up. Mirrors what worker + aggregator
	# would leave behind, minus the actual binary content.
	job = tmp_path / "j"
	(job / "outputs").mkdir(parents=True)
	(job / "outputs_archive").mkdir()
	(job / "outputs" / "seed_000005_haadf.zarr").mkdir()
	(job / "outputs_archive" / "seed_000001_abf.zarr").mkdir()
	(job / "outputs_archive" / "seed_000002_haadf.zarr").mkdir()
	assert _discover_channels(job) == ["abf", "haadf"]


# --------------------------------------------------------------------------- #
# load_ensemble — the heart of the bridge
# --------------------------------------------------------------------------- #


def test_load_ensemble_returns_none_when_no_seeds(tmp_path):
	# Job exists but no per-seed zarrs for the channel.
	(tmp_path / "outputs").mkdir()
	assert load_ensemble(tmp_path, "haadf") is None


def test_load_ensemble_stacks_with_frozen_phonons_axis():
	"""The 2-seed haadf ensemble has shape (2, ny, nx) with the
	new axis labelled FrozenPhononsAxis(_ensemble_mean=True). This is
	the property that lets abtem's .reduce_ensemble() do its thing."""
	from abtem.core.axes import FrozenPhononsAxis

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		ensemble = load_ensemble(job, "haadf")
		assert ensemble is not None
		# Per-seed shapes match a single image, so stacked = (N=2, ny, nx).
		assert ensemble.array.shape[0] == 2
		assert len(ensemble.ensemble_shape) == 1
		assert ensemble.ensemble_shape[0] == 2
		# The new axis MUST be FrozenPhononsAxis with ensemble_mean=True,
		# otherwise reduce_ensemble() would not average over it.
		axes = ensemble.ensemble_axes_metadata
		assert len(axes) == 1
		assert isinstance(axes[0], FrozenPhononsAxis), (
			f"new axis is {type(axes[0]).__name__}, not FrozenPhononsAxis"
		)
		assert axes[0]._ensemble_mean is True


def test_load_ensemble_reduce_matches_aggregate():
	"""End-to-end correctness: load_ensemble(...).reduce_ensemble()
	must yield the SAME image as the regular aggregator's mean.
	Floating-point precision (max |Δ| < 1e-9 expected; .mean is
	commutative, so the only floats are the per-pixel sums)."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		aggregate_job(job)

		# Aggregator mean (saved to aggregate/haadf.zarr)
		agg_mean = np.asarray(
			abtem.from_zarr(str(job / "aggregate" / "haadf.zarr")).array
		)

		# Bridge: load_ensemble then reduce_ensemble (note: at this
		# point per-seed zarrs live in outputs_archive/ — aggregate
		# already moved them. load_ensemble must still find them.)
		ensemble = load_ensemble(job, "haadf")
		assert ensemble is not None
		reduced = ensemble.reduce_ensemble()
		bridge_mean = np.asarray(reduced.array)

		assert agg_mean.shape == bridge_mean.shape
		max_diff = float(np.max(np.abs(agg_mean - bridge_mean)))
		assert max_diff < 1e-9, (
			f"bridge mean diverges from aggregator mean by {max_diff:.3e}"
		)


def test_load_ensemble_round_trip_preserves_axis():
	"""Write ensemble to zarr, load it back via abtem.from_zarr, verify
	the FrozenPhononsAxis is preserved and reduce_ensemble still works.
	This is the user-facing promise: 'abtem.from_zarr(<channel>_ensemble.zarr)
	gives you an abtem-native ensemble.'"""
	from abtem.core.axes import FrozenPhononsAxis

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		ensemble = load_ensemble(job, "haadf")
		assert ensemble is not None

		zarr_path = tmp_path / "test_ensemble.zarr"
		ensemble.to_zarr(str(zarr_path), overwrite=True)
		roundtrip = abtem.from_zarr(str(zarr_path))

		# Type + shape preserved.
		assert type(roundtrip).__name__ == type(ensemble).__name__
		assert np.array_equal(np.asarray(roundtrip.array), np.asarray(ensemble.array))
		# Axis metadata preserved.
		axes = roundtrip.ensemble_axes_metadata
		assert len(axes) == 1
		assert isinstance(axes[0], FrozenPhononsAxis)
		assert axes[0]._ensemble_mean is True
		# reduce_ensemble still does the right thing on the round-tripped object.
		reduced_orig = np.asarray(ensemble.reduce_ensemble().array)
		reduced_rt = np.asarray(roundtrip.reduce_ensemble().array)
		assert np.allclose(reduced_orig, reduced_rt, atol=1e-12)


# --------------------------------------------------------------------------- #
# to_ensemble_files — the CLI-driven side
# --------------------------------------------------------------------------- #


def test_to_ensemble_files_auto_discovers_all_channels():
	"""Default (channels=None) writes one ensemble zarr per channel
	with per-seed files."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		results = to_ensemble_files(job)
		written = {ch for ch, _ in results}
		assert written == {"haadf", "abf", "bf"}
		for ch, path in results:
			assert path.name == f"{ch}_ensemble.zarr"
			assert path.exists()


def test_to_ensemble_files_specific_channel_only():
	"""Explicit channels= argument restricts output to that subset."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		results = to_ensemble_files(job, channels=["haadf"])
		assert [ch for ch, _ in results] == ["haadf"]
		# The other detectors did NOT get _ensemble.zarr written.
		assert not (job / "aggregate" / "abf_ensemble.zarr").exists()
		assert not (job / "aggregate" / "bf_ensemble.zarr").exists()


def test_to_ensemble_files_raises_on_missing_explicit_channel():
	"""Asking for a channel that doesn't exist is an error (vs auto
	which silently produces no output for an empty job)."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		with pytest.raises(ValueError, match="None of the requested"):
			to_ensemble_files(job, channels=["nonexistent"])


def test_to_ensemble_files_custom_out_dir():
	"""out_dir override writes to a non-default location."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job = _setup_two_seed_haadf_abf_bf(tmp_path)
		custom = tmp_path / "elsewhere"
		results = to_ensemble_files(job, channels=["haadf"], out_dir=custom)
		assert results[0][1].parent == custom
		assert (custom / "haadf_ensemble.zarr").exists()


def _run_all():
	import inspect
	mod = inspect.getmodule(_run_all)
	for name, fn in inspect.getmembers(mod, inspect.isfunction):
		if not name.startswith("test_"):
			continue
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {name}: {e}")
			return 1
		except Exception as e:
			print(f"ERROR {name}: {type(e).__name__}: {e}")
			return 1
		else:
			print(f"PASS  {name}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
