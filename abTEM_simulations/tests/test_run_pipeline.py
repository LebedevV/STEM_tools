"""
End-to-end test for the ``abtem-run`` convenience wrapper
(``cli.run_pipeline``), which drives the full v6 flow in-process:
generator -> workers -> aggregator.

Heavier than test_aggregate (everything that test runs, plus the
generator), so kept to a single comprehensive test.

Runnable two ways:
    PYTHONPATH=src python3 tests/test_run_pipeline.py
    PYTHONPATH=src pytest tests/test_run_pipeline.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from abtem_run.cli import run_pipeline

from _fixtures import setup_cif_dir, write_tiny_config


def _setup_config(tmp: Path) -> Path:
	cif_dir = setup_cif_dir(tmp)
	return write_tiny_config(
		tmp / "test.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=2,
		phonons_seed=50,
	)


def test_run_pipeline_full_flow():
	"""generator -> workers -> aggregator, all in-process. Verifies the final
	aggregate/ contents are present and outputs/ has been cleaned up
	(test_enabled=false)."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		run_dir = run_pipeline(cfg_path)

		assert run_dir.is_dir()
		job_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
		assert len(job_dirs) == 1
		job_dir = job_dirs[0]

		# Planning artifacts from the generator stage
		assert (job_dir / "surf.xyz").exists()
		assert (job_dir / "combined.png").exists()

		# Workers ran — all .todo became .done
		assert not list((job_dir / "seeds").glob("*.todo"))
		done = list((job_dir / "seeds").glob("*.done"))
		assert len(done) == 2, f"expected 2 .done seeds, got {len(done)}"

		# Aggregator ran — aggregate/ populated, outputs/ cleaned up
		agg = job_dir / "aggregate"
		assert (agg / "haadf.tif").exists()
		assert (agg / "haadf.zarr").exists()
		assert (agg / "haadf_0-1.tif").exists()
		assert (agg / "potential_projection.png").exists()
		assert (agg / "potential_projection.tif").exists()

		assert not (job_dir / "outputs").exists(), (
			"outputs/ should have been deleted (test_enabled=false)"
		)


# --- Resume path ---


def test_run_pipeline_resume_completes_a_partial_sweep():
	"""generate_only first, then resume to actually run workers + aggregator."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		# Stage 1: plan only — no workers, no aggregator.
		run_dir = run_pipeline(cfg_path, generate_only=True)
		job_dir = next(p for p in run_dir.iterdir() if p.is_dir())
		assert len(list((job_dir / "seeds").glob("*.todo"))) == 2
		assert not list((job_dir / "seeds").glob("*.done"))
		assert not list((job_dir / "aggregate").iterdir())

		# Stage 2: resume — should pick up the .todo files and complete the run.
		returned = run_pipeline(resume_dir=run_dir)
		assert returned == run_dir

		assert not list((job_dir / "seeds").glob("*.todo"))
		done = list((job_dir / "seeds").glob("*.done"))
		assert len(done) == 2

		assert (job_dir / "aggregate" / "haadf.tif").exists()
		assert (job_dir / "aggregate" / "potential_projection.png").exists()

		# Cleanup happened (test_enabled=false default)
		assert not (job_dir / "outputs").exists()


def test_run_pipeline_resume_is_idempotent_on_complete_sweep():
	"""Resuming a fully-completed sweep should be a graceful no-op."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		# Run to completion once.
		run_dir = run_pipeline(cfg_path)
		job_dir = next(p for p in run_dir.iterdir() if p.is_dir())
		first_aggregate_files = sorted(p.name for p in (job_dir / "aggregate").iterdir())

		# Resume on the same directory — should NOT crash, should NOT mess up aggregate/.
		run_pipeline(resume_dir=run_dir)

		second_aggregate_files = sorted(p.name for p in (job_dir / "aggregate").iterdir())
		assert second_aggregate_files == first_aggregate_files


def test_run_pipeline_resume_with_partial_worker_progress():
	"""If half the seeds are .done and half are still .todo, resume finishes them all."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		# generate_only -> we have 2 .todo files
		run_dir = run_pipeline(cfg_path, generate_only=True)
		job_dir = next(p for p in run_dir.iterdir() if p.is_dir())
		todos = sorted((job_dir / "seeds").glob("*.todo"))
		assert len(todos) == 2

		# Simulate a half-done state: manually mark the first todo as .done
		# (without running its work — outputs/ won't have anything for that seed).
		# This is contrived but exercises the resume path's tolerance.
		todos[0].rename(todos[0].with_suffix(".done"))

		# Run resume — should process only the remaining .todo.
		run_pipeline(resume_dir=run_dir)

		# Both should now be .done.
		assert not list((job_dir / "seeds").glob("*.todo"))
		done = list((job_dir / "seeds").glob("*.done"))
		assert len(done) == 2


def test_run_pipeline_resume_rejects_combined_generate_only():
	"""resume + generate_only must error — they don't make sense together."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		try:
			run_pipeline(resume_dir=tmp_path, generate_only=True)
		except ValueError as e:
			assert "generate" in str(e).lower() and "resume" in str(e).lower()
		else:
			raise AssertionError("expected ValueError combining resume + generate_only")


def test_run_pipeline_resume_rejects_bad_dir():
	"""resume on a directory that isn't a generated run dir should fail loudly."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		# An empty dir with no job subdirs
		try:
			run_pipeline(resume_dir=tmp_path)
		except ValueError as e:
			assert "job dir" in str(e).lower() or "seeds" in str(e).lower()
		else:
			raise AssertionError("expected ValueError on empty resume_dir")


def test_run_pipeline_generate_only_flag_stops_after_generator():
	"""generate_only=True: plan + planning artifacts, no workers/aggregate."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		run_dir = run_pipeline(cfg_path, generate_only=True)

		job_dir = next(p for p in run_dir.iterdir() if p.is_dir())
		assert (job_dir / "surf.xyz").exists()
		assert (job_dir / "combined.png").exists()
		assert len(list((job_dir / "seeds").glob("*.todo"))) == 2
		assert not list((job_dir / "seeds").glob("*.done"))
		assert not list((job_dir / "aggregate").iterdir())


def _run_all():
	for fn in (
		test_run_pipeline_generate_only_flag_stops_after_generator,
		test_run_pipeline_resume_rejects_combined_generate_only,
		test_run_pipeline_resume_rejects_bad_dir,
		test_run_pipeline_resume_with_partial_worker_progress,
		test_run_pipeline_resume_is_idempotent_on_complete_sweep,
		test_run_pipeline_resume_completes_a_partial_sweep,
		test_run_pipeline_full_flow,
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
