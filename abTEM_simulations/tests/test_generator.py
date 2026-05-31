"""
Tests for the generator (``abtem_run.generator_run``):

- Job-dir layout matches what worker + aggregator expect.
- Planning artifacts ``surf.xyz`` and ``combined.png`` are emitted per job
  (docs/worker.md decision #5).
- Seed counts in ``seeds/`` match cfg.simulations.frozen_phonons with the
  right offsets from cfg.job.phonons_seed.

Uses the tiny Pm3m lamella so the test runs in a few seconds.

Runnable two ways:
    PYTHONPATH=src python3 tests/test_generator.py
    PYTHONPATH=src pytest tests/test_generator.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from abtem_run.generator_run import generate_run

from _fixtures import setup_cif_dir, write_tiny_config


def _setup_config(tmp: Path) -> Path:
	"""Write a tiny config to tmp + symlink cifs/. Returns the config path."""
	cif_dir = setup_cif_dir(tmp)
	return write_tiny_config(
		tmp / "test.toml",
		folder_sim=tmp,
		cif_dir=cif_dir,
		frozen_phonons=3,
		phonons_seed=100,
	)


def test_generator_emits_expected_layout():
	"""Job dir has the planning artifacts, seeds/, outputs/, aggregate/, .toml."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		run_dir = generate_run(cfg_path)

		assert run_dir.is_dir()
		assert run_dir.name.startswith("gen_")
		assert (run_dir / "run_manifest.json").exists()

		# Exactly one job dir for (TaTe2-style stem)_(hkl)_(tilt)
		job_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
		assert len(job_dirs) == 1, f"expected 1 job dir, got {len(job_dirs)}: {job_dirs}"
		job_dir = job_dirs[0]
		assert "Pm3m" in job_dir.name
		assert "001" in job_dir.name  # hkl encoded in stem
		assert "ta0.0_tb0.0" in job_dir.name

		# Layout matches what worker + aggregator expect
		assert (job_dir / "seeds").is_dir()
		assert (job_dir / "outputs").is_dir()
		assert (job_dir / "aggregate").is_dir()
		assert next(job_dir.glob("*.toml")).exists()

		# Planning artifacts emitted (Decision #5)
		assert (job_dir / "surf.xyz").exists(), "generator did not emit surf.xyz"
		assert (job_dir / "combined.png").exists(), "generator did not emit combined.png"
		assert (job_dir / "surf.xyz").stat().st_size > 0
		assert (job_dir / "combined.png").stat().st_size > 0


def test_generator_emits_correct_seed_files():
	"""frozen_phonons=3, phonons_seed=100 -> seed_000100, seed_000101, seed_000102."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		run_dir = generate_run(cfg_path)
		job_dir = next(p for p in run_dir.iterdir() if p.is_dir())

		todos = sorted((job_dir / "seeds").glob("*.todo"))
		assert len(todos) == 3
		assert todos[0].name == "seed_000100.todo"
		assert todos[1].name == "seed_000101.todo"
		assert todos[2].name == "seed_000102.todo"


def test_generator_manifest_lists_the_job():
	"""run_manifest.json points at the actual job dir + tracks task count."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_config(tmp_path)

		run_dir = generate_run(cfg_path)
		manifest = json.loads((run_dir / "run_manifest.json").read_text())

		assert manifest["n_frames"] == 1
		assert len(manifest["jobs"]) == 1
		job = manifest["jobs"][0]
		assert job["phase"] == "Pm3m.cif"
		assert job["hkl"] == [0, 0, 1]
		assert job["is_uvw"] is False
		assert job["n_tasks"] == 3
		assert (run_dir / job["job_dir"]).is_dir()


# --------------------------------------------------------------------------- #
# Direction 6: multiple phases per job
# --------------------------------------------------------------------------- #


def _setup_multiphase_config(tmp: Path, phases: list[str]) -> Path:
	"""Write a tiny config with `job.phase` as a list of CIF names."""
	import tomllib
	import tomli_w

	cfg_path = _setup_config(tmp)
	data = tomllib.loads(cfg_path.read_text())
	data["job"]["phase"] = phases
	with cfg_path.open("wb") as f:
		tomli_w.dump(data, f)
	return cfg_path


def test_generator_emits_one_job_dir_per_phase():
	"""With phase=[A, B], generator emits 2 job dirs (one per phase)
	sharing the same seed integers — phase-to-phase comparison should
	use locked RNG draws."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		# Pm3m.cif and CC.cif are both in the repo's cifs/.
		cfg_path = _setup_multiphase_config(tmp_path, ["Pm3m.cif", "CC.cif"])
		run_dir = generate_run(cfg_path)

		manifest = json.loads((run_dir / "run_manifest.json").read_text())
		assert len(manifest["jobs"]) == 2
		phases_in_manifest = {j["phase"] for j in manifest["jobs"]}
		assert phases_in_manifest == {"Pm3m.cif", "CC.cif"}

		# Two distinct job dirs.
		job_dirs = sorted(j["job_dir"] for j in manifest["jobs"])
		assert len(job_dirs) == 2
		assert "Pm3m" in job_dirs[1] or "Pm3m" in job_dirs[0]
		assert "CC" in job_dirs[0] or "CC" in job_dirs[1]

		# Same seed integers in both: phase-to-phase locked RNG.
		seed_sets = []
		for j in manifest["jobs"]:
			seeds = sorted(p.name for p in (run_dir / j["job_dir"] / "seeds").glob("*.todo"))
			seed_sets.append(seeds)
		assert seed_sets[0] == seed_sets[1], "seeds should be identical across phases"


def test_generator_per_job_toml_has_scalar_phase():
	"""Each job dir's frozen TOML carries a single phase string, not the
	list — the worker reads that scalar to know which CIF to load."""
	import tomllib
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		cfg_path = _setup_multiphase_config(tmp_path, ["Pm3m.cif", "CC.cif"])
		run_dir = generate_run(cfg_path)
		manifest = json.loads((run_dir / "run_manifest.json").read_text())

		for j in manifest["jobs"]:
			toml_path = run_dir / j["cfg"]
			data = tomllib.loads(toml_path.read_text())
			phase = data["job"]["phase"]
			assert isinstance(phase, str), (
				f"per-job TOML at {toml_path} carries non-scalar phase: {phase!r}"
			)
			assert phase == j["phase"]


def _run_all():
	for fn in (
		test_generator_emits_expected_layout,
		test_generator_emits_correct_seed_files,
		test_generator_manifest_lists_the_job,
		test_generator_emits_one_job_dir_per_phase,
		test_generator_per_job_toml_has_scalar_phase,
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
