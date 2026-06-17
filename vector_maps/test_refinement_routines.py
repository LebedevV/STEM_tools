#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for refinement_routines helpers.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refinement_routines as rr
from dicts_handling import dicts_to_vector


def test_empty_subarea_raises_clear_error():
	# a sub_area excluding every observed point -> actionable message, not IndexError;
	# pass a complete lat_params so the raise is for the right reason (not a KeyError).
	lat = {"abg": [1.0, 1.0, 90.0], "fit_abg": [False, False, False],
	       "base": [0.0, 0.0, 0.0], "fit_base": [False, False, False]}
	df = pd.DataFrame({"x_obs0": [10., 20., 30., 40.], "y_obs0": [10., 20., 30., 40.]})
	with pytest.raises(ValueError, match="no observed points"):
		rr.preprocess_dataset(lat, {}, {}, df, 0.008,
				      recall_zero=True, sub_area=[100, 200, 100, 200])


def test_gen_ij_shape_and_content():
	ij = rr.gen_ij((0, 3))
	assert ij.shape == (9, 2)
	assert {tuple(r) for r in ij} == {(i, j) for i in range(3) for j in range(3)}


def test_cost_function_mean_squared_distance():
	obs = np.array([[0.0, 0.0], [1.0, 1.0]])
	theor = np.array([[0.0, 0.0], [1.0, 0.0]])
	# per-point squared distances 0 and 1 -> mean 0.5
	assert np.isclose(rr.cost_function(obs, theor), 0.5)
	assert rr.cost_function(obs, obs) == 0.0


def test_mask_close_points_drops_later_duplicate():
	pts = np.array([[0.0, 0.0], [0.0, 0.1], [5.0, 5.0]])
	assert list(rr.mask_close_points(pts, threshold=0.5)) == [True, False, True]
	far = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
	assert rr.mask_close_points(far, threshold=0.5).all()


def _ortho(motif):
	# orthorhombic a=2, b=3, gamma=90, phi=0, no shift -> exact, hand-checkable placement
	lat = {"abg": [2.0, 3.0, 90.0], "fit_abg": [False, False, False],
	       "base": [0.0, 0.0, 0.0], "fit_base": [False, False, False]}
	return dicts_to_vector(lat, motif, {})[0], lat


def test_get_coords_single_atom_lands_on_lattice():
	motif = {"A_1": {"atom": "X", "coord": (0.0, 0.0), "use": True, "fit": [False, False]}}
	p, lat = _ortho(motif)
	ij = np.array([[0, 0], [1, 0], [0, 1]])
	cr_lat, cr_ij, _ij_ref = rr.get_coords_from_ij(ij, p, None, lat, motif, {}, crop=False)
	assert np.allclose(cr_lat, [[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]], atol=1e-6)
	assert np.array_equal(cr_ij, ij)


def test_get_coords_multi_atom_motif_loop():
	# two-atom motif -> each lattice point yields both atoms, exercising the per-atom
	# loop in get_coords_from_ij that a single-atom motif never reaches.
	motif = {"A_1": {"atom": "X", "coord": (0.0, 0.0), "use": True, "fit": [False, False]},
	         "B_1": {"atom": "Y", "coord": (0.0, 0.5), "use": True, "fit": [False, False]}}
	p, lat = _ortho(motif)
	ij = np.array([[0, 0], [1, 0]])
	cr_lat, _cr_ij, _ij_ref = rr.get_coords_from_ij(ij, p, None, lat, motif, {}, crop=False)
	# 2 lattice points x 2 atoms; A_1 on the grid, B_1 offset by 0.5*b = 1.5 in y
	assert cr_lat.shape == (4, 2)
	assert np.allclose(cr_lat, [[0.0, 0.0], [2.0, 0.0], [0.0, 1.5], [2.0, 1.5]], atol=1e-6)
