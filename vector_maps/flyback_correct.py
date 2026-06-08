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
