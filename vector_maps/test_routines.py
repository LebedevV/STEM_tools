#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for routines helpers.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routines import read_frame_calib, resolve_frame_path, rotate_vec, vector_map_calc


def test_resolves_tiff_variant(tmp_path):
	(tmp_path / "f.tiff").write_bytes(b"abc")
	assert resolve_frame_path(str(tmp_path), "f").endswith("f.tiff")


def test_trailing_extension_on_fname_is_ignored(tmp_path):
	(tmp_path / "f.TIF").write_bytes(b"abc")
	assert resolve_frame_path(str(tmp_path), "f.tif").endswith("f.TIF")


def test_identical_variants_ok(tmp_path):
	(tmp_path / "f.tif").write_bytes(b"abc")
	(tmp_path / "f.tiff").write_bytes(b"abc")
	assert os.path.exists(resolve_frame_path(str(tmp_path), "f"))


def test_conflicting_variants_raise(tmp_path):
	(tmp_path / "f.tif").write_bytes(b"abc")
	(tmp_path / "f.tiff").write_bytes(b"xyz")
	with pytest.raises(ValueError, match="byte-wise"):
		resolve_frame_path(str(tmp_path), "f")


def test_missing_raises(tmp_path):
	with pytest.raises(FileNotFoundError):
		resolve_frame_path(str(tmp_path), "nope")


def test_rotate_vec_cardinal_angles():
	assert np.allclose(rotate_vec((1.0, 0.0), 0.0), (1.0, 0.0))
	assert np.allclose(rotate_vec((1.0, 0.0), 90.0), (0.0, 1.0), atol=1e-12)
	assert np.allclose(rotate_vec((1.0, 0.0), 180.0), (-1.0, 0.0), atol=1e-12)


def test_read_frame_calib_isotropic(tmp_path):
	(tmp_path / "f_frame.txt").write_text(
		"xres_px\t1024\nyres_px\t1024\nxreal_nm\t10.24\nyreal_nm\t10.24\n")
	assert np.isclose(read_frame_calib(str(tmp_path), "f"), 0.01)


def test_read_frame_calib_anisotropic_raises(tmp_path):
	(tmp_path / "f_frame.txt").write_text(
		"xres_px\t1024\nyres_px\t1024\nxreal_nm\t10.24\nyreal_nm\t20.48\n")
	with pytest.raises(ValueError, match="Anisotropic"):
		read_frame_calib(str(tmp_path), "f")


def test_read_frame_calib_fallback_when_absent(tmp_path):
	assert read_frame_calib(str(tmp_path), "nope", fallback=0.5) == 0.5
	with pytest.raises(FileNotFoundError):
		read_frame_calib(str(tmp_path), "nope")


def test_vector_map_calc_diff_dist_and_projection():
	df = pd.DataFrame({
		"x_obs": [1.0, 0.0], "y_obs": [0.0, 2.0],
		"x_theor_new": [0.0, 0.0], "y_theor_new": [0.0, 0.0],
	})
	std, out = vector_map_calc(0.0, df)
	# vdiff = [[1,0],[0,2]] -> vdist [1,2]; std(|vdiff|, axis 0) = [0.5, 1.0]
	assert np.allclose(out["vdist"].to_numpy(), [1.0, 2.0])
	assert np.allclose(std, [0.5, 1.0])
	# phi = 0 -> projection equals the raw difference
	assert np.allclose(np.array(out["vproj"].tolist()), [[1.0, 0.0], [0.0, 2.0]])
