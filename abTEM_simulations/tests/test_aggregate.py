"""
End-to-end test for the aggregator: spin up a tiny two-seed job, run both
workers, then aggregate. Verifies:

- aggregate/ ends up with the expected per-channel files
- gaussian-blurred TIFF variants exist for scan channels
- potential_projection.{png,tif} + _scanned.tif are written
- outputs/ is deleted when test_enabled=false (cleanup happens)
- outputs/ is preserved when test_enabled=true

Slow (two multislices + one Potential.project()), ~25-40s on CPU. Run via:

    PYTHONPATH=src python3 tests/test_aggregate.py
    PYTHONPATH=src pytest tests/test_aggregate.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from abtem_run.aggregate import aggregate_job
from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


def _setup_two_seed_job(tmp: Path, *, test_enabled: bool = False):
	"""Materialize a job_dir with two .todo files (seeds 42 and 43)."""
	cif_dir = setup_cif_dir(tmp)

	job_dir = tmp / "test_job"
	job_dir.mkdir()
	(job_dir / "seeds").mkdir()
	(job_dir / "outputs").mkdir()

	# fph_sigma > 0 + 2 seeds so the cross-seed mean is non-trivial
	# (otherwise testing aggregation against a degenerate constant).
	write_tiny_config(
		job_dir / "test_job.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=2,
		fph_sigma=0.05,
		phonons_seed=42,
		test_enabled=test_enabled,
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)

	todo_paths = []
	for seed in (42, 43):
		p = job_dir / "seeds" / f"seed_{seed:06d}.todo"
		p.write_text(f"{seed}\n")
		todo_paths.append(p)

	return job_dir, todo_paths


def _run_both_workers(job_dir, todo_paths):
	for p in todo_paths:
		run_one_seed(job_dir, p)


def test_aggregate_end_to_end_archives():
	"""Two seeds, scan-only (haadf), test_enabled=false → outputs/ moved
	to outputs_archive/. Per-seed files survive so future extend batches
	can accumulate into the cross-seed mean."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path, test_enabled=False)
		_run_both_workers(job_dir, todo_paths)

		# Pre-aggregate: outputs/ has per-seed files, no aggregate/ yet
		out_dir = job_dir / "outputs"
		archive_dir = job_dir / "outputs_archive"
		assert (out_dir / "seed_000042_haadf.zarr").exists()
		assert (out_dir / "seed_000043_haadf.zarr").exists()
		assert not archive_dir.exists(), "fresh job: no archive yet"

		aggregate_job(job_dir)

		agg_dir = job_dir / "aggregate"
		assert agg_dir.is_dir()

		# Scan-channel outputs
		assert (agg_dir / "haadf.tif").exists()
		assert (agg_dir / "haadf.zarr").exists()
		# Gaussian-blurred variants (BLUR_SIGMAS = [0.025, 0.1, 0.25])
		assert (agg_dir / "haadf_0-025.tif").exists()
		assert (agg_dir / "haadf_0-1.tif").exists()
		assert (agg_dir / "haadf_0-25.tif").exists()

		# Potential projection preview
		assert (agg_dir / "potential_projection.png").exists()
		assert (agg_dir / "potential_projection.tif").exists()
		assert (agg_dir / "potential_projection_scanned.tif").exists()

		# Archive happened: outputs/ is gone, archive holds the per-seed
		# files so the next extend round can read them as historical input.
		assert not out_dir.exists(), "outputs/ should be removed post-archive"
		assert archive_dir.is_dir(), "outputs_archive/ must exist post-aggregate"
		assert (archive_dir / "seed_000042_haadf.zarr").exists()
		assert (archive_dir / "seed_000043_haadf.zarr").exists()


def test_aggregate_preserves_outputs_in_test_mode():
	"""Two seeds, test_enabled=true → outputs/ preserved in-place (no
	archive move). Diagnostic-mode behavior: per-seed displaced.xyz and
	zarr both stay where the worker put them."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path, test_enabled=True)
		_run_both_workers(job_dir, todo_paths)
		aggregate_job(job_dir)

		out_dir = job_dir / "outputs"
		archive_dir = job_dir / "outputs_archive"
		agg_dir = job_dir / "aggregate"

		# In-place preservation: archive NOT created, outputs/ untouched.
		assert out_dir.exists(), "test_enabled=true should preserve outputs/"
		assert not archive_dir.exists(), "test_enabled=true should NOT archive"
		assert (out_dir / "seed_000042_haadf.zarr").exists()
		assert (out_dir / "seed_000043_haadf.zarr").exists()
		# Worker dropped displaced.xyz for each seed under test mode
		assert (out_dir / "seed_000042_displaced.xyz").exists()
		assert (out_dir / "seed_000043_displaced.xyz").exists()

		# Aggregate still produced
		assert (agg_dir / "haadf.tif").exists()
		assert (agg_dir / "potential_projection.png").exists()


def test_aggregate_cumulative_across_archive():
	"""Add a third seed AFTER an initial 2-seed aggregate has archived.
	Re-aggregate; the mean must include all THREE seeds (archive + new).
	This is the cumulative-mean contract that abtem-run-extend depends on."""
	import numpy as np
	import abtem

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path, test_enabled=False)
		_run_both_workers(job_dir, todo_paths)
		aggregate_job(job_dir)  # archives seeds 42, 43

		out_dir = job_dir / "outputs"
		archive_dir = job_dir / "outputs_archive"
		agg_dir = job_dir / "aggregate"

		# Stash the 2-seed mean for comparison.
		mean_2seed = np.asarray(abtem.from_zarr(str(agg_dir / "haadf.zarr")).array).copy()

		# Add seed 44 via a fresh .todo + worker run.
		new_todo = job_dir / "seeds" / "seed_000044.todo"
		new_todo.write_text("44\n")
		run_one_seed(job_dir, new_todo)

		# At this point outputs/ has seed 44; archive has 42, 43.
		assert (out_dir / "seed_000044_haadf.zarr").exists()
		assert (archive_dir / "seed_000042_haadf.zarr").exists()
		assert (archive_dir / "seed_000043_haadf.zarr").exists()

		aggregate_job(job_dir)

		# New mean must be the mean over THREE per-seed zarrs.
		# We can verify two ways: the value changed from mean_2seed, AND
		# the archive now contains all three.
		mean_3seed = np.asarray(abtem.from_zarr(str(agg_dir / "haadf.zarr")).array)
		assert mean_3seed.shape == mean_2seed.shape
		assert not np.array_equal(mean_3seed, mean_2seed), (
			"cumulative mean must change when a 3rd seed is added"
		)
		assert (archive_dir / "seed_000044_haadf.zarr").exists(), (
			"seed 44 should now be archived after the second aggregate"
		)
		assert not out_dir.exists(), "outputs/ should be re-cleaned"


def test_aggregate_refuses_when_todos_remain():
	"""If .todo files are still in seeds/, aggregate_job must fail loudly."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path)
		# Run only the first worker — second .todo stays as .todo
		run_one_seed(job_dir, todo_paths[0])

		try:
			aggregate_job(job_dir)
		except RuntimeError as e:
			assert ".todo" in str(e).lower() or "incomplete" in str(e).lower()
		else:
			raise AssertionError("aggregate_job should have refused on remaining .todo")


def test_aggregate_pure_reaggregate_against_archive():
	"""Audit item 3: re-aggregate is legal when outputs/ doesn't exist
	(only outputs_archive/). User scenario: a normal aggregate runs,
	archives outputs/, then the user immediately re-runs aggregate
	(e.g. accidentally, or after editing the TOML to tweak the static
	scan). The aggregator should NOT crash on the missing outputs/.
	"""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path, test_enabled=False)
		_run_both_workers(job_dir, todo_paths)
		aggregate_job(job_dir)  # archives outputs/

		out_dir = job_dir / "outputs"
		archive_dir = job_dir / "outputs_archive"
		agg_dir = job_dir / "aggregate"

		# Sanity: outputs/ is gone, archive has the seeds, aggregate exists.
		assert not out_dir.exists()
		assert archive_dir.is_dir()
		assert (agg_dir / "haadf.tif").exists()

		# Re-aggregate. Should not raise, should not change the haadf mean
		# (same seeds), and should still leave the archive in place.
		import numpy as np
		import abtem
		mean_before = np.asarray(abtem.from_zarr(str(agg_dir / "haadf.zarr")).array).copy()
		aggregate_job(job_dir)
		mean_after = np.asarray(abtem.from_zarr(str(agg_dir / "haadf.zarr")).array)
		assert np.array_equal(mean_before, mean_after), (
			"pure re-aggregate produced a different mean — should be bit-equal"
		)
		assert not out_dir.exists(), "no fresh seeds means no outputs/ to archive again"
		assert (archive_dir / "seed_000042_haadf.zarr").exists()


def test_aggregate_extend_with_test_enabled_preserves_outputs():
	"""Audit item 4: extend should work when the original aggregate ran
	in test_enabled=true mode (outputs/ preserved in-place, no archive).
	Extend reads from outputs/ ∪ outputs_archive/ for the 'used' check,
	so finding seeds in outputs/ works fine.
	"""
	import abtem
	import numpy as np
	from abtem_run.extend import extend_job

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_paths = _setup_two_seed_job(tmp_path, test_enabled=True)
		_run_both_workers(job_dir, todo_paths)
		aggregate_job(job_dir)  # test_enabled=true → outputs/ preserved

		out_dir = job_dir / "outputs"
		archive_dir = job_dir / "outputs_archive"
		assert out_dir.exists(), "test_enabled=true should preserve outputs/"
		assert not archive_dir.exists(), "test_enabled=true should not create archive"

		# Extend by 2 — should find seeds 42, 43 in outputs/ and pick 44, 45.
		emitted = extend_job(job_dir, add=2)
		assert emitted == [44, 45]

		# Run workers on the extension.
		for s in (44, 45):
			run_one_seed(job_dir, job_dir / "seeds" / f"seed_{s:06d}.todo")

		# Re-aggregate. In test_enabled=true mode, outputs/ should now have
		# all four seeds (42-45 — extend didn't move anything; new workers
		# wrote into the same outputs/). archive still absent.
		aggregate_job(job_dir)
		for s in (42, 43, 44, 45):
			assert (out_dir / f"seed_{s:06d}_haadf.zarr").exists(), (
				f"seed {s} should be in outputs/ under test_enabled=true"
			)
		assert not archive_dir.exists()

		# And the new mean reflects all four seeds (manual mean check).
		mean_agg = np.asarray(abtem.from_zarr(str(job_dir / "aggregate" / "haadf.zarr")).array)
		mean_manual = sum(
			np.asarray(abtem.from_zarr(str(out_dir / f"seed_{s:06d}_haadf.zarr")).array)
			for s in (42, 43, 44, 45)
		) / 4
		assert np.allclose(mean_agg, mean_manual, atol=1e-9)


def test_aggregate_static_baseline_is_cached_between_aggregates():
	"""Audit item 7: the static-lattice block (projected-potential preview
	+ optional static-baseline scan) is the heaviest non-per-seed work
	the aggregator does. After an extend → workers → re-aggregate cycle
	with the TOML untouched, the static artifacts should be skipped
	(their mtimes don't change), but they should regenerate if the TOML
	is touched (semantic change).

	This is a re-aggregation efficiency test, not a correctness test:
	the cumulative-mean correctness is already covered by
	``test_aggregate_cumulative_across_archive``.
	"""
	import time

	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()

		# Same 2-seed setup but with emit_static_baseline=true so the
		# static-scan path is exercised in addition to the projection.
		cif_dir = setup_cif_dir(tmp_path)
		job_dir = tmp_path / "test_job"
		job_dir.mkdir()
		(job_dir / "seeds").mkdir()
		(job_dir / "outputs").mkdir()
		toml_path = write_tiny_config(
			job_dir / "test_job.toml",
			folder_sim=tmp_path,
			cif_dir=cif_dir,
			frozen_phonons=2,
			fph_sigma=0.05,
			phonons_seed=42,
			emit_static_baseline=True,
		)
		write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)
		todo_paths = []
		for seed in (42, 43):
			p = job_dir / "seeds" / f"seed_{seed:06d}.todo"
			p.write_text(f"{seed}\n")
			todo_paths.append(p)
		_run_both_workers(job_dir, todo_paths)
		aggregate_job(job_dir)

		agg_dir = job_dir / "aggregate"
		canonical = {
			"projection_png": agg_dir / "potential_projection.png",
			"projection_tif": agg_dir / "potential_projection.tif",
			"haadf_static_zarr": agg_dir / "haadf_static.zarr",
		}
		for name, p in canonical.items():
			assert p.exists(), f"missing post-first-aggregate: {name}"
		mtimes_first = {name: p.stat().st_mtime for name, p in canonical.items()}

		# Extend by 1 seed, worker, re-aggregate. TOML untouched.
		# Sleep briefly so we'd notice an mtime bump on the inch scale.
		time.sleep(0.05)
		extra_todo = job_dir / "seeds" / "seed_000044.todo"
		extra_todo.write_text("44\n")
		run_one_seed(job_dir, extra_todo)
		aggregate_job(job_dir)

		mtimes_after_extend = {name: p.stat().st_mtime for name, p in canonical.items()}
		for name in canonical:
			assert mtimes_after_extend[name] == mtimes_first[name], (
				f"{name} was regenerated despite unchanged TOML — cache miss "
				f"(mtime {mtimes_first[name]:.6f} -> {mtimes_after_extend[name]:.6f})"
			)

		# Now touch the TOML and re-aggregate. Cache should invalidate
		# and the static artifacts should be regenerated.
		time.sleep(0.05)
		toml_path.touch()
		aggregate_job(job_dir)
		mtimes_after_touch = {name: p.stat().st_mtime for name, p in canonical.items()}
		for name in canonical:
			assert mtimes_after_touch[name] > mtimes_first[name], (
				f"{name} was NOT regenerated after TOML touch — cache stuck "
				f"(mtime {mtimes_first[name]:.6f} -> {mtimes_after_touch[name]:.6f})"
			)


def _run_all():
	for fn in (
		test_aggregate_refuses_when_todos_remain,
		test_aggregate_end_to_end_archives,
		test_aggregate_preserves_outputs_in_test_mode,
		test_aggregate_cumulative_across_archive,
		test_aggregate_pure_reaggregate_against_archive,
		test_aggregate_extend_with_test_enabled_preserves_outputs,
		test_aggregate_static_baseline_is_cached_between_aggregates,
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
