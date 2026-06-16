#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for unit_cell_average: resample vs raw, full-cells, fit bridge.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unit_cell_average import average_unit_cell, lattice_px_from_fit

A, B, ORIG = (22.0, 0.0), (3.0, 19.0), (7.0, 5.0)   # sheared test lattice (px)


def _synth(a, b, o, H=220, W=200, noise=0.0, seed=0):
	# identical content in every cell: f(frac_u, frac_v), optionally + white noise
	ys, xs = np.indices((H, W)).astype(float)
	det = a[0] * b[1] - a[1] * b[0]
	u = (b[1] * (xs - o[0]) - b[0] * (ys - o[1])) / det
	v = (-a[1] * (xs - o[0]) + a[0] * (ys - o[1])) / det
	img = np.sin(2 * np.pi * (u - np.floor(u))) + 0.5 * np.cos(2 * np.pi * (v - np.floor(v))) + 0.3
	if noise:
		img = img + np.random.default_rng(seed).normal(0, noise, img.shape)
	return img


def test_resample_uniform_count_zero_floor():
	# identical cells -> resample reads the same (u,v) in every cell: ~0 std, uniform count
	mean, sd, count = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="resample")
	assert np.nanmin(count) == np.nanmax(count)
	assert np.all(np.isfinite(mean))
	assert np.nanmax(sd) < 1e-6


def test_raw_tracks_noise_resample_attenuates():
	img = _synth(A, B, ORIG, noise=0.20, seed=1)
	_, sd_raw, _ = average_unit_cell(img, A, B, ORIG, method="raw")
	_, sd_res, _ = average_unit_cell(img, A, B, ORIG, method="resample")
	assert 0.17 < np.nanmedian(sd_raw) < 0.23          # raw ~ input white noise
	assert np.nanmedian(sd_res) < np.nanmedian(sd_raw)  # bilinear read damps white noise


def test_full_cells_only_trims_border():
	full = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="raw", full_cells_only=True)[2].sum()
	allc = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="raw", full_cells_only=False)[2].sum()
	assert full < allc


def test_lattice_px_from_fit_axis_aligned():
	# phi=0, gamma=90 -> a along x, b along y, both scaled by 1/calib
	lat = {"abg": [0.40, 0.50, 90.0], "base": [0.30, 0.15, 0.0]}
	calib = 0.02
	a_px, b_px, o_px = lattice_px_from_fit(lat, calib)
	assert np.allclose(a_px, (0.40 / calib, 0.0))
	assert np.allclose(b_px, (0.0, 0.50 / calib))
	assert np.allclose(o_px, (0.30 / calib, 0.15 / calib))


def test_degenerate_lattice_raises():
	try:
		average_unit_cell(np.zeros((50, 50)), (10.0, 0.0), (20.0, 0.0), (0.0, 0.0))
	except ValueError:
		return
	raise AssertionError("expected ValueError for colinear a, b")


def test_raw_rejects_shape():
	try:
		average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="raw", shape=(10, 10))
	except ValueError:
		return
	raise AssertionError("raw should reject a custom shape (resample-only)")


def test_sub_area_restricts_to_roi():
	img = _synth(A, B, ORIG)
	full = average_unit_cell(img, A, B, ORIG)[2].sum()
	roi = average_unit_cell(img, A, B, ORIG, sub_area=[60, 140, 60, 140])[2].sum()
	assert 0 < roi < full                              # ROI folds strictly fewer cells


def test_raw_native_pixel_footprint():
	# raw output is the cell's real-px bounding box (true sheared shape), distinct
	# from resample's fractional N x M, with NaN at the bbox corners outside the cell.
	rm = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="raw")[0]
	em = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="resample")[0]
	assert rm.shape != em.shape
	assert np.isnan(rm).any()


def test_uc_figure_smoke():
	# the combined schematic | mean | std figure renders to a PNG without error
	try:
		import matplotlib
		matplotlib.use("Agg")
	except ImportError:
		return
	import os
	import tempfile
	from unit_cell_average import _uc_figure
	mean, std, _ = average_unit_cell(_synth(A, B, ORIG), A, B, ORIG, method="raw")
	lat = {"abg": [0.40, 0.50, 100.0], "base": [0.0, 0.0, 12.0]}
	motif = {
		"A_1": {"atom": "Ta", "coord": (0.0, 0.0), "use": True},
		"B_1": {"atom": "Te", "coord": (0.0, 0.30), "use": True},
	}
	out = os.path.join(tempfile.gettempdir(), "uc_fig_smoke.png")
	_uc_figure(out, lat, 0.02, motif, mean, std)
	assert os.path.getsize(out) > 0
	os.remove(out)


def test_to_tiffs_resolves_tiff_extension():
	# unit_cell_average_to_tiffs must find a .tiff frame even when handed a .tif path
	import cv2
	import os
	import tempfile
	from unit_cell_average import unit_cell_average_to_tiffs
	d = tempfile.mkdtemp()
	cv2.imwrite(os.path.join(d, "frame.tiff"), _synth(A, B, ORIG).astype(np.float32))
	lat = {"abg": [2.2, 2.2, 90.0], "base": [0.7, 0.5, 0.0]}
	out = os.path.join(d, "out")
	unit_cell_average_to_tiffs(os.path.join(d, "frame.tif"), lat, calib=0.1, out_stem=out)
	assert os.path.exists(out + "_uc_mean.tif")


if __name__ == "__main__":
	for _name, _fn in sorted(globals().items()):
		if _name.startswith("test_") and callable(_fn):
			_fn()
			print(f"  ok  {_name}")
	print("all tests passed")
