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


def test_resample_floor_below_raw_uniform_count():
	img = _synth(A, B, ORIG)                              # identical content in every cell
	mean, sd_res, count = average_unit_cell(img, A, B, ORIG, method="resample")
	_, sd_raw, _ = average_unit_cell(img, A, B, ORIG, method="raw")
	assert np.nanmedian(sd_res) < 0.1 * np.nanmedian(sd_raw)   # resample kills the gradient floor
	assert np.nanmin(count) == np.nanmax(count)               # uniform count
	assert np.all(np.isfinite(mean))


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


if __name__ == "__main__":
	for _name, _fn in sorted(globals().items()):
		if _name.startswith("test_") and callable(_fn):
			_fn()
			print(f"  ok  {_name}")
	print("all tests passed")
