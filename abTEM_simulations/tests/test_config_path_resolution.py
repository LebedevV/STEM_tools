"""
Verifies that paths.folder and paths.folder_sim in the TOML are resolved
relative to the config file's directory when they're not absolute.
Closes phase-2 item #1 of the packaging roadmap (kill hardcoded paths).

Runnable two ways:
    PYTHONPATH=src python3 tests/test_config_path_resolution.py
    PYTHONPATH=src pytest tests/test_config_path_resolution.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from textwrap import dedent

from abtem_run.config import load_config


# Minimal TOML that satisfies every required field in AppConfig.
# Only paths.folder and paths.folder_sim are interesting for this test;
# everything else is filler.
_MIN_TOML = dedent("""
    [paths]
    folder_sim = "{folder_sim}"
    extr = "./"
    folder = "{folder}"
    sample_name = "test"

    [gpu_related]
    use_gpu = false
    dask_cuda = false
    cupy_fft_cache_size = "256 MB"
    dask_chunk_size_gpu = "256 MB"
    dask_chunk_size = "512 MB"

    [simulations]
    override_sampling = false
    frozen_phonons = "None"
    fph_sigma = false
    do_full_run = false

    [microscope]
    HT_value = 200000
    do_diffraction = false
    haadfinner = 99
    haadfouter = 200
    abfinner = 15
    abfouter = 33
    bfinner = 0.01
    bfouter = 9

    [lamella_settings]
    max_uvw = 10
    sblock_size = 50.0
    scan_s = 10.0
    borders = 2.0
    thickness = 5.0
    extra_shift_z = 0.0
    tol = 0.1
    atom_to_zero = "Ta"
    global_tilt_a = 0.0
    global_tilt_b = 0.0
    tilt_degrees = true
    add_vacancies_toggle = false
    element_to_remove = "Ti"
    probability_of_vac = 0.0

    [job]
    phase = "TaTe2.cif"
    hkl_to_do = [0, 0, 1]
    is_uvw = false
    phonons_seed = 0
""").strip()


def _write_config(folder: str, folder_sim: str, dir_path: Path) -> Path:
	toml_path = dir_path / "test_config.toml"
	toml_path.write_text(_MIN_TOML.format(folder=folder, folder_sim=folder_sim))
	return toml_path


def test_relative_paths_resolve_against_config_dir():
	"""'./' and 'out_full/' in the TOML should resolve to dirs next to the config."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		toml_path = _write_config("./", "./out_full/", tmp_path)

		cfg = load_config(toml_path)

		# folder is './' → absolute = tmp_path + '/'
		assert cfg.paths.folder == str(tmp_path) + "/", (
			f"folder not resolved against config dir: {cfg.paths.folder!r}"
		)
		# folder_sim is './out_full/' → tmp_path + '/out_full/'
		expected_sim = str(tmp_path / "out_full") + "/"
		assert cfg.paths.folder_sim == expected_sim, (
			f"folder_sim not resolved against config dir: "
			f"{cfg.paths.folder_sim!r}, expected {expected_sim!r}"
		)


def test_absolute_paths_pass_through():
	"""Absolute paths in the TOML must NOT be rewritten."""
	with tempfile.TemporaryDirectory() as tmp:
		tmp_path = Path(tmp).resolve()
		# Use the tmp dir itself as the absolute path so we don't depend on
		# any specific filesystem layout for the test to make sense.
		abs_path = str(tmp_path) + "/"
		toml_path = _write_config(abs_path, abs_path, tmp_path)

		cfg = load_config(toml_path)

		# Trailing slash is always normalized to exactly one '/'.
		assert cfg.paths.folder == abs_path
		assert cfg.paths.folder_sim == abs_path


def test_resolution_does_not_depend_on_cwd():
	"""Same config, different CWDs, identical resolved paths."""
	import os

	with tempfile.TemporaryDirectory() as cfg_tmp, tempfile.TemporaryDirectory() as cwd_tmp:
		cfg_dir = Path(cfg_tmp).resolve()
		toml_path = _write_config("./", "./out_full/", cfg_dir)

		original_cwd = os.getcwd()
		try:
			os.chdir(cwd_tmp)
			cfg = load_config(toml_path)
			# folder/folder_sim should resolve against the *config dir*, not CWD.
			assert cfg.paths.folder == str(cfg_dir) + "/", (
				f"folder unexpectedly resolved against CWD: {cfg.paths.folder!r}"
			)
			assert cfg.paths.folder_sim == str(cfg_dir / "out_full") + "/"
		finally:
			os.chdir(original_cwd)


def _run_all():
	for fn in (
		test_relative_paths_resolve_against_config_dir,
		test_absolute_paths_pass_through,
		test_resolution_does_not_depend_on_cwd,
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
