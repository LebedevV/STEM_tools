#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Flyback-hysteresis warp (Mullarkey et al., Microsc. Microanal. 28, 2022).
# Short scan-line flyback compresses the start of each line; modelled as an exponential
# displacement along the fast-scan axis (image-x, in nm): a true position x appears
# measured at x + exp_a*exp(-x/exp_b). exp_a (edge amplitude, nm) and exp_b (decay
# length, nm) are fit as extra_pars by the refinement; this is the forward map the model
# applies to its theoretical positions. Direct nm space -- no timing constants needed.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np


def flyback_warp(x, exp_a, exp_b):
	"""Forward flyback map on the fast-scan coordinate x (nm): x + exp_a*exp(-x/exp_b)."""
	x = np.asarray(x, dtype=float)
	return x + exp_a * np.exp(-x / exp_b)


def slow_axis_warp(x, y, extr):
	"""Optional low-order slow-axis (y) distortion: add sx<k>*y**k to x and sy<k>*y**k
	to y for the coeffs (nm) present in extra_pars, k in 1..3. No-op if none set.
	The linear term (sx1/sy1) is degenerate with the lattice shear/scale, so enable it
	only with the lattice pinned (the two-stage centre->edge workflow); the constant is
	left to the lattice origin (shx/shy)."""
	dx = sum(extr[f"sx{k}"][0] * y ** k for k in (1, 2, 3) if f"sx{k}" in extr)
	dy = sum(extr[f"sy{k}"][0] * y ** k for k in (1, 2, 3) if f"sy{k}" in extr)
	return x + dx, y + dy
