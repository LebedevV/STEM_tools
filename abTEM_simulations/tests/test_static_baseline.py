"""
Tests for ``simulations.emit_static_baseline``.

Current state (post PR #8 review): the flag controls only the
static-lattice **projection preview** in the aggregator. The static-scan
path was dropped — the aggregator no longer orchestrates multislice;
for a static-lattice scan, run a separate job with
``frozen_phonons = "None"``.

Verifies:
- when the flag is True, ``aggregate/potential_projection_static.{png,tif,
  _scanned.tif}`` is written alongside the phonon-averaged
  ``potential_projection.*``;
- when the flag is False (default), no ``*_static*`` files are produced.

Slow-ish (~10s; one worker pass + projection rendering). Run via:

    PYTHONPATH=src python3 tests/test_static_baseline.py
    PYTHONPATH=src pytest tests/test_static_baseline.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from abtem_run.aggregate import aggregate_job
from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


def _setup_one_seed_job(tmp: Path, *, emit_static_baseline: bool):
	"""Minimal job_dir with a single .todo, scan-only (haadf)."""
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
		fph_sigma=0.08,
		phonons_seed=42,
		emit_static_baseline=emit_static_baseline,
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)
	todo = job_dir / "seeds" / "seed_000042.todo"
	todo.write_text("42\n")
	return job_dir, todo


def test_static_projection_files_present_when_flag_on():
	"""With emit_static_baseline=true, the static projection preview lands."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo = _setup_one_seed_job(tmp_path, emit_static_baseline=True)
		run_one_seed(job_dir, todo)
		aggregate_job(job_dir)

		agg = job_dir / "aggregate"
		# phonon-averaged projection (always emitted on a successful run)
		assert (agg / "potential_projection.png").exists()
		assert (agg / "potential_projection.tif").exists()
		# static projection (only with the flag on)
		assert (agg / "potential_projection_static.png").exists()
		assert (agg / "potential_projection_static.tif").exists()
		assert (agg / "potential_projection_static_scanned.tif").exists()


def test_no_static_files_when_flag_off():
	"""Default (emit_static_baseline=false): no ``*_static*`` files anywhere."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo = _setup_one_seed_job(tmp_path, emit_static_baseline=False)
		run_one_seed(job_dir, todo)
		aggregate_job(job_dir)

		agg = job_dir / "aggregate"
		# phonon-averaged projection still produced
		assert (agg / "potential_projection.tif").exists()
		# but no _static variants
		assert not list(agg.glob("*_static*")), (
			f"unexpected _static files when flag is off: "
			f"{list(agg.glob('*_static*'))}"
		)


def _run_all():
	for fn in (
		test_no_static_files_when_flag_off,
		test_static_projection_files_present_when_flag_on,
	):
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {fn.__name__}: {e}")
			return 1
		except Exception as e:
			print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
			return 1
		else:
			print(f"PASS  {fn.__name__}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
