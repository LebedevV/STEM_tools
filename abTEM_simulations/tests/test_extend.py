"""
Tests for ``abtem-run-extend`` — append phonon snapshots to an existing job.

V. Lebedev's 2026-05-20 requirement: support adding more phonons after
the fact, with non-overlapping seeds and cumulative aggregation. This
file covers the seed-emission contract and the validation rules; the
cumulative-mean integration is covered in test_aggregate.py
(test_aggregate_cumulative_across_archive).

Fast (~no multislice, just file emission).

Runnable two ways:
    PYTHONPATH=src python3 tests/test_extend.py
    PYTHONPATH=src pytest tests/test_extend.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import abtem_run  # noqa: F401  — registers patches
from abtem_run.extend import extend_job, _scan_used_seeds

from _fixtures import setup_cif_dir, write_tiny_config


def _setup_empty_job(tmp: Path) -> Path:
	"""Make a job_dir with a valid TOML but no prior seeds."""
	cif_dir = setup_cif_dir(tmp)
	job_dir = tmp / "test_job"
	job_dir.mkdir()
	(job_dir / "seeds").mkdir()
	write_tiny_config(
		job_dir / "test_job.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=2,
		fph_sigma=0.05,
		phonons_seed=42,
	)
	return job_dir


def _seed_done_records(job_dir: Path, *seed_ints: int) -> None:
	"""Drop minimal seed_*.done files so _scan_used_seeds picks them up
	WITHOUT actually running multislice. Stand-in for 'a previous batch
	completed'."""
	for s in seed_ints:
		(job_dir / "seeds" / f"seed_{s:06d}.done").write_text(f"{s}\n")
		# Also seed_*_haadf.zarr (file, not directory) so the zarr-glob
		# path of _scan_used_seeds finds it too. We don't read contents
		# in this test; existence is enough.
		archive = job_dir / "outputs_archive"
		archive.mkdir(exist_ok=True)
		(archive / f"seed_{s:06d}_haadf.zarr").write_text("stub")


# --------------------------------------------------------------------------- #
# _scan_used_seeds
# --------------------------------------------------------------------------- #


def test_scan_used_seeds_empty_job():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		assert _scan_used_seeds(job) == set()


def test_scan_used_seeds_archive_only():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		assert _scan_used_seeds(job) == {42, 43}


def test_scan_used_seeds_picks_up_pending_todos():
	"""A queued-but-not-run .todo also counts as 'used' — refusing
	to emit it again avoids accidental double-queue."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		(job / "seeds" / "seed_000099.todo").write_text("99\n")
		assert _scan_used_seeds(job) == {99}


# --------------------------------------------------------------------------- #
# extend_job — arg validation
# --------------------------------------------------------------------------- #


def test_extend_requires_exactly_one_of_add_or_seeds():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		with pytest.raises(ValueError, match="exactly one"):
			extend_job(job)
		with pytest.raises(ValueError, match="exactly one"):
			extend_job(job, add=2, seeds=[100])


def test_extend_refuses_empty_job():
	"""No prior seeds => extension is meaningless."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		with pytest.raises(ValueError, match="no prior seeds"):
			extend_job(job, add=2)


def test_extend_refuses_nonpositive_add():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42)
		for bad in (0, -1, -5):
			with pytest.raises(ValueError, match="positive"):
				extend_job(job, add=bad)


def test_extend_refuses_overlapping_explicit_seeds():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43, 44)
		with pytest.raises(ValueError, match="overlap"):
			extend_job(job, seeds=[44, 45])


def test_extend_refuses_duplicate_seeds_in_request():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42)
		with pytest.raises(ValueError, match="duplicate"):
			extend_job(job, seeds=[100, 100, 101])


def test_extend_refuses_empty_seeds_list():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42)
		with pytest.raises(ValueError, match="empty"):
			extend_job(job, seeds=[])


# --------------------------------------------------------------------------- #
# extend_job — emission
# --------------------------------------------------------------------------- #


def test_extend_add_picks_seeds_past_max_used():
	"""--add N starts at max(used)+1."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		emitted = extend_job(job, add=3)
		assert emitted == [44, 45, 46]
		for s in emitted:
			assert (job / "seeds" / f"seed_{s:06d}.todo").exists()


def test_extend_explicit_seeds_emits_exactly_those():
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		emitted = extend_job(job, seeds=[100, 101, 102])
		assert emitted == [100, 101, 102]
		for s in emitted:
			assert (job / "seeds" / f"seed_{s:06d}.todo").exists()
		# Did NOT emit auto-picked 44, 45 etc.
		assert not (job / "seeds" / "seed_000044.todo").exists()


def test_extend_writes_extensions_log():
	"""Each extend call appends a record to extensions.json."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		extend_job(job, add=2)
		extend_job(job, seeds=[100])

		log = json.loads((job / "extensions.json").read_text())
		assert isinstance(log, list)
		assert len(log) == 2
		assert log[0]["source"] == "add"
		assert log[0]["added_seeds"] == [44, 45]
		assert log[0]["count"] == 2
		assert log[1]["source"] == "seeds"
		assert log[1]["added_seeds"] == [100]


def test_extend_repeated_add_increments_correctly():
	"""--add called twice should keep incrementing past the newly-queued
	seeds (because the prior .todo files count as 'used')."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42, 43)
		first = extend_job(job, add=3)        # 44, 45, 46
		second = extend_job(job, add=2)       # 47, 48 — picks up after the .todos
		assert first == [44, 45, 46]
		assert second == [47, 48]


def test_extend_refuses_to_overwrite_existing_todo():
	"""Defense in depth: if a .todo exists for a requested seed, fail
	rather than overwriting. _scan_used_seeds should already catch
	this, but the exclusive-open guard in extend_job is a backstop."""
	with tempfile.TemporaryDirectory() as tmp:
		job = _setup_empty_job(Path(tmp))
		_seed_done_records(job, 42)
		# manually create a .todo for seed 99; explicit --seeds [99] must refuse
		(job / "seeds" / "seed_000099.todo").write_text("99\n")
		with pytest.raises(ValueError, match="overlap"):
			extend_job(job, seeds=[99])


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
