#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for routines helpers.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routines import resolve_frame_path


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
