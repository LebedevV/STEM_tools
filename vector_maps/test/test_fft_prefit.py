#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for fft_prefit: recover a planted lattice's rotation (align) and a,b,gamma (prefit).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routines.fft_prefit import fft_prefit, _real_basis_px

A, B, GAMMA, PHI, CALIB = 0.30, 0.42, 90.0, 12.0, 0.01   # planted lattice


def _synth(a, b, gamma, phi, calib, H=512, W=512, sigma=2.0):
	# Gaussian blobs at the lattice points of a known (a,b,gamma,phi); local windows for speed
	a_px, b_px = _real_basis_px({"abg": [a, b, gamma], "base": [0.0, 0.0, phi]}, calib)
	img = np.zeros((H, W))
	ox, oy = W / 2.0, H / 2.0
	rad = int(3 * sigma)
	for i in range(-40, 41):
		for j in range(-40, 41):
			x = ox + i * a_px[0] + j * b_px[0]
			y = oy + i * a_px[1] + j * b_px[1]
			c, r = int(round(x)), int(round(y))
			c0, c1 = max(0, c - rad), min(W, c + rad + 1)
			r0, r1 = max(0, r - rad), min(H, r + rad + 1)
			if c1 <= c0 or r1 <= r0:
				continue
			ys, xs = np.mgrid[r0:r1, c0:c1]
			img[r0:r1, c0:c1] += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
	return img


def _phi_err(phi, ref):
	return abs(((phi - ref + 90.0) % 180.0) - 90.0)      # a*/-a* gives a 180-deg ambiguity


def test_align_recovers_rotation():
	img = _synth(A, B, GAMMA, PHI, CALIB)
	lat = {"abg": [A, B, GAMMA], "base": [0.0, 0.0, PHI - 6.0],
	       "fit_abg": [True] * 3, "fit_base": [True] * 3}
	out = fft_prefit(img, lat, CALIB, refine_abg=False)
	assert _phi_err(out["base"][2], PHI) < 1.0
	assert out["abg"] == [A, B, GAMMA]                   # align refines phi only


def test_prefit_refines_abg():
	img = _synth(A, B, GAMMA, PHI, CALIB)
	lat = {"abg": [A * 1.06, B * 0.94, GAMMA - 3.0], "base": [0.0, 0.0, PHI - 5.0],
	       "fit_abg": [True] * 3, "fit_base": [True] * 3}
	out = fft_prefit(img, lat, CALIB, refine_abg=True)
	assert abs(out["abg"][0] - A) / A < 0.03
	assert abs(out["abg"][1] - B) / B < 0.03
	assert abs(out["abg"][2] - GAMMA) < 2.0
	assert _phi_err(out["base"][2], PHI) < 1.0


def test_prefit_recovers_oblique_gamma():
	# gamma=90 is the one value where a metric/sign error hides; plant an oblique cell
	g = 75.0
	img = _synth(A, B, g, PHI, CALIB)
	lat = {"abg": [A * 1.03, B * 0.97, g + 3.0], "base": [0.0, 0.0, PHI - 4.0],
	       "fit_abg": [True] * 3, "fit_base": [True] * 3}
	out = fft_prefit(img, lat, CALIB, refine_abg=True)
	assert abs(out["abg"][0] - A) / A < 0.03
	assert abs(out["abg"][1] - B) / B < 0.03
	assert abs(out["abg"][2] - g) < 2.0
	assert _phi_err(out["base"][2], PHI) < 1.0


def test_finds_phi_without_guess():
	# phi is read from the FFT (a -90..90 sweep), not refined from the input -- plant it
	# far from the seed (base phi=0) and across the +-90 range, including negative
	for planted in (-45.0, 30.0, 70.0):
		img = _synth(A, B, GAMMA, planted, CALIB)
		lat = {"abg": [A, B, GAMMA], "base": [0.0, 0.0, 0.0],
		       "fit_abg": [True] * 3, "fit_base": [True] * 3}
		assert _phi_err(fft_prefit(img, lat, CALIB)["base"][2], planted) < 1.5


def test_fit_flags_preserved():
	img = _synth(A, B, GAMMA, PHI, CALIB)
	lat = {"abg": [A, B, GAMMA], "base": [0.0, 0.0, PHI - 4.0],
	       "fit_abg": [True, False, True], "fit_base": [False, True, False]}
	out = fft_prefit(img, lat, CALIB)
	assert out["fit_abg"] == [True, False, True]
	assert out["fit_base"] == [False, True, False]


def test_degenerate_basis_raises():
	try:
		fft_prefit(np.zeros((64, 64)), {"abg": [0.3, 0.3, 0.0], "base": [0.0, 0.0, 0.0]}, CALIB)
	except ValueError:
		return
	raise AssertionError("expected ValueError for gamma=0 (colinear a, b)")


if __name__ == "__main__":
	for _name, _fn in sorted(globals().items()):
		if _name.startswith("test_") and callable(_fn):
			_fn()
			print(f"  ok  {_name}")
	print("all tests passed")
