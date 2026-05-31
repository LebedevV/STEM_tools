"""
Tests for the push-mode worker (``abtem_run.worker``).

End-to-end test runs ``run_one_seed`` on a tiny lamella, asserts outputs
and .done rename. Phonon-displacement reproducibility, independent-vs-
batched divergence, and the FrozenPhonons baseline behavior live in
``tests/phonons/``.

    PYTHONPATH=src python3 tests/test_worker.py
    PYTHONPATH=src pytest tests/test_worker.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from abtem_run.worker import run_one_seed

from _fixtures import setup_cif_dir, write_tiny_config, write_tiny_ground_xyz


def _setup_job_dir(
	tmp: Path,
	*,
	test_enabled: bool = False,
	fph_sigma: float | bool = 0.05,
) -> tuple[Path, Path]:
	"""Materialize a minimal job_dir + surf.xyz + one .todo file."""
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
		fph_sigma=fph_sigma,
		phonons_seed=42,
		test_enabled=test_enabled,
	)
	write_tiny_ground_xyz(job_dir, cif_dir=cif_dir)

	todo_path = job_dir / "seeds" / "seed_000042.todo"
	todo_path.write_text("42\n")

	return job_dir, todo_path


def test_run_one_seed_end_to_end_minimal():
	"""Single seed through the full worker. Slow (~tens of seconds on CPU)."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_path = _setup_job_dir(tmp_path)

		assert todo_path.exists()
		assert not todo_path.with_suffix(".done").exists()

		run_one_seed(job_dir, todo_path)

		assert not todo_path.exists(), "worker did not remove the .todo"
		assert todo_path.with_suffix(".done").exists(), "worker did not create .done"

		out_dir = job_dir / "outputs"
		assert (out_dir / "seed_000042_haadf.tif").exists()
		assert (out_dir / "seed_000042_haadf.zarr").exists()
		assert not (out_dir / "seed_000042_displaced.xyz").exists()
		assert not (out_dir / "seed_000042_diff.tif").exists()
		assert not (out_dir / "seed_000042_cbed.tif").exists()


def test_run_one_seed_rejects_multi_hkl_job():
	"""A per-job TOML must have exactly one hkl. The generator never emits
	multi-hkl TOMLs, but a hand-edit could — the worker refuses, doesn't
	silently pick the first."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_path = _setup_job_dir(tmp_path)

		toml = job_dir / "test_job.toml"
		text = toml.read_text()
		text = text.replace(
			"hkl_to_do = [0, 0, 1]",
			"hkl_to_do = [[0, 0, 1], [1, 1, 0]]",
		)
		toml.write_text(text)

		try:
			run_one_seed(job_dir, todo_path)
		except ValueError as e:
			assert "single-hkl" in str(e).lower() or "hkl_to_do" in str(e).lower()
		else:
			raise AssertionError("run_one_seed should have rejected a multi-hkl job")


def test_run_one_seed_rejects_zero_sigma_with_phonons():
	"""frozen_phonons=int + fph_sigma<=0 would produce N identical frames —
	the worker must refuse rather than silently fall through."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_path = _setup_job_dir(tmp_path, fph_sigma=0.0)

		try:
			run_one_seed(job_dir, todo_path)
		except ValueError as e:
			msg = str(e).lower()
			assert "fph_sigma" in msg and "identical" in msg, (
				f"expected fph_sigma + identical-frames message, got: {e!r}"
			)
		else:
			raise AssertionError(
				"run_one_seed should have refused frozen_phonons=int + sigma<=0"
			)


def test_run_one_seed_missing_surf_xyz_errors():
	"""Worker reads surf.xyz from the job dir. Missing → FileNotFoundError
	with a helpful pointer to abtem-run-generate."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_path = _setup_job_dir(tmp_path)
		(job_dir / "surf.xyz").unlink()

		try:
			run_one_seed(job_dir, todo_path)
		except FileNotFoundError as e:
			assert "surf.xyz" in str(e)
		else:
			raise AssertionError("run_one_seed should have failed on missing surf.xyz")


def test_run_one_seed_test_enabled_dumps_displaced_xyz():
	"""With test_enabled=true and displacement actually happening, the worker
	writes the per-seed displaced atoms."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		job_dir, todo_path = _setup_job_dir(tmp_path, test_enabled=True)

		run_one_seed(job_dir, todo_path)

		xyz = job_dir / "outputs" / "seed_000042_displaced.xyz"
		assert xyz.exists(), "test_enabled=true did not produce displaced.xyz"
		first_line = xyz.read_text().splitlines()[0].strip()
		assert first_line.isdigit() and int(first_line) > 0


def _run_all():
	for fn in (
		test_run_one_seed_rejects_multi_hkl_job,
		test_run_one_seed_missing_surf_xyz_errors,
		test_run_one_seed_rejects_zero_sigma_with_phonons,
		test_run_one_seed_end_to_end_minimal,
		test_run_one_seed_test_enabled_dumps_displaced_xyz,
	):
		try:
			fn()
		except AssertionError as e:
			print(f"FAIL  {fn.__name__}: {e}")
			return 1
		except Exception as e:
			print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
			for tmp_root in Path(tempfile.gettempdir()).glob("tmp*/cifs"):
				try:
					tmp_root.unlink()
				except OSError:
					pass
			return 1
		else:
			print(f"PASS  {fn.__name__}")
	return 0


if __name__ == "__main__":
	raise SystemExit(_run_all())
