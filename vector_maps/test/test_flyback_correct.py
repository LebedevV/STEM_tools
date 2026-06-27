#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the flyback warp map (the model's forward distortion).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routines.routines import flyback_warp, slow_axis_warp


def test_warp_identity_when_zero_amplitude():
	x = np.linspace(0.0, 50.0, 100)
	assert np.allclose(flyback_warp(x, 0.0, 2.0), x)


def test_warp_full_amplitude_at_line_start_and_decays():
	x = np.array([0.0, 100.0])
	d = flyback_warp(x, 0.3, 2.0) - x
	assert abs(d[0] - 0.3) < 1e-12          # full edge amplitude at x = 0
	assert d[1] < 1e-6                       # negligible far down the line (x >> exp_b)


def test_warp_matches_formula():
	x = np.linspace(0.0, 10.0, 50)
	assert np.allclose(flyback_warp(x, 0.25, 1.5), x + 0.25 * np.exp(-x / 1.5))


def test_extra_pars_unpack_feeds_warp():
	# get_coords reads exp_a/exp_b as extr['exp_a'][0]; confirm that interface round-trips
	from routines.dicts_handling import dicts_to_vector, unpack_to_dicts
	lat = {"abg": [0.3, 0.3, 90.0], "fit_abg": [False, False, False],
	       "base": [0.0, 0.0, 0.0], "fit_base": [False, False, False]}
	motif = {"A_1": {"atom": "X", "coord": (0.0, 0.0), "use": True, "fit": [False, False]}}
	extra = {"exp_a": (0.12, True), "exp_b": (1.7, True)}
	p, _, _, _ = dicts_to_vector(lat, motif, extra)
	_, _, extr = unpack_to_dicts(p, lat, motif, extra)
	assert extr["exp_a"][0] == 0.12 and extr["exp_b"][0] == 1.7
	x = np.linspace(0.0, 5.0, 20)
	assert np.allclose(flyback_warp(x, extr["exp_a"][0], extr["exp_b"][0]), x + 0.12 * np.exp(-x / 1.7))


def test_slow_axis_warp_quad_cubic():
	y = np.linspace(0.0, 16.0, 40)
	x = np.zeros_like(y)
	xx, yy = slow_axis_warp(x, y, {"sx2": (0.01, True), "sy3": (0.001, True)})
	assert np.allclose(xx, 0.01 * y ** 2)
	assert np.allclose(yy, y + 0.001 * y ** 3)


def test_slow_axis_warp_linear_applied_const_and_absent_noop():
	y = np.linspace(0.0, 16.0, 10)
	x = np.ones_like(y)
	xx, _ = slow_axis_warp(x, y, {"sx1": (0.5, True)})        # linear is supported
	assert np.allclose(xx, x + 0.5 * y)
	# constant (sx0, left to the lattice origin) and unrelated keys are ignored -> identity
	xx2, yy2 = slow_axis_warp(x, y, {"exp_a": (0.1, True), "sx0": (9.0, True)})
	assert np.allclose(xx2, x) and np.allclose(yy2, y)


if __name__ == "__main__":
	for _name, _fn in sorted(globals().items()):
		if _name.startswith("test_") and callable(_fn):
			_fn()
			print(f"  ok  {_name}")
	print("all tests passed")
