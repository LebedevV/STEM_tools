#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for routines helpers.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routines import calib_from_frame_size, read_toml_calib, resolve_frame_path


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


def test_read_toml_calib_recomputes_from_actual_pixels(tmp_path):
	# single source of truth: calib = (scan_s/10) / n_px from the frame's REAL grid,
	# not a Nyquist estimate -- a 40-px image of a 50 A frame -> 0.125 nm/px
	import cv2
	import numpy as np
	folder = os.path.join(str(tmp_path), "")
	with open(os.path.join(folder, "f.toml"), "w") as fh:
		fh.write("[lamella_settings]\nscan_s = 50.0\n")
	cv2.imwrite(os.path.join(folder, "frame.tif"), np.zeros((40, 40), np.float32))
	calib = read_toml_calib(folder, "frame", os.path.join(folder, "f.toml"))
	assert calib == pytest.approx((50.0 / 10) / 40)


def test_read_toml_calib_rejects_non_square(tmp_path):
	# the fit calib is one isotropic scalar; a non-square frame must fail loud
	import cv2
	import numpy as np
	folder = os.path.join(str(tmp_path), "")
	with open(os.path.join(folder, "f.toml"), "w") as fh:
		fh.write("[lamella_settings]\nscan_s = 50.0\n")
	cv2.imwrite(os.path.join(folder, "frame.tif"), np.zeros((40, 50), np.float32))
	with pytest.raises(ValueError, match="non-square"):
		read_toml_calib(folder, "frame", os.path.join(folder, "f.toml"))


def test_calib_from_frame_size_recomputes_from_actual_pixels(tmp_path):
	# the value path used by the batch sweep: scan_s (A) / n_px, no toml read
	import cv2
	import numpy as np
	folder = os.path.join(str(tmp_path), "")
	cv2.imwrite(os.path.join(folder, "frame.tif"), np.zeros((40, 40), np.float32))
	assert calib_from_frame_size(folder, "frame", 50.0) == pytest.approx((50.0 / 10) / 40)
