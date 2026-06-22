#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for dicts_handling: the parameter-vector <-> dicts engine + equation system.
# The fixture is the real 4-atom Si<110> motif from examples/fit_si.toml, so the
# chained equation (B_1c depends on the equation-driven B_1) is exercised.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dicts_handling as dh


def _si_config():
	# the maintained Si<110> config (examples/fit_si.toml), four motif atoms:
	#   A_1   base, fixed at the origin
	#   A_1c  centering          = A_1 + (centering_a, centering_b)
	#   B_1   dumbbell partner    = A_1 + polar(db_dist, db_angle)
	#   B_1c  dumbbell centering  = B_1 + (centering_a, centering_b)   <- chained on B_1
	lat = {"abg": [0.3867, 0.5469, 89.75], "fit_abg": [True, True, True],
	       "base": [0.0, 0.0, 1.5], "fit_base": [True, True, True]}
	motif = {
		"A_1": {"atom": "Si", "coord": (0.0, 0.0), "I": 1, "use": True,
		        "fit": [False, False]},
		"A_1c": {"atom": "Si", "coord": (0.0, 0.0), "I": 1, "use": True,
		         "fit": [True, True],
		         "eq": ["= motif['A_1'][0] + extra_pars['centering_a']",
		                "= motif['A_1'][1] + extra_pars['centering_b']"]},
		"B_1": {"atom": "Si", "coord": (0.0, 0.2), "I": 1, "use": True,
		        "fit": [True, True],
		        "eq": ["= motif['A_1'][0] + extra_pars['db_dist']*np.sin(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][0]",
		               "= motif['A_1'][1] + extra_pars['db_dist']*np.cos(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][1]"]},
		"B_1c": {"atom": "Si", "coord": (0.5, 0.7), "I": 1, "use": True,
		         "fit": [True, True],
		         "eq": ["= motif['B_1'][0] + extra_pars['centering_a']",
		                "= motif['B_1'][1] + extra_pars['centering_b']"]},
	}
	extra = {"db_dist": (0.136, True), "db_angle": (0.0, True),
	         "centering_a": (0.5, True), "centering_b": (0.5, True)}
	return lat, motif, extra


def test_layout_indices_and_size():
	lat, motif, extra = _si_config()
	layout = dh.build_layout(lat, motif, extra)
	# 3 abg + 3 base, then 2 slots per used motif atom (4 atoms), then 4 extra = 18
	assert layout["lat"]["abg"] == [0, 1, 2]
	assert layout["lat"]["base"] == [3, 4, 5]
	assert layout["motif"]["A_1"] == (6, 7)
	assert layout["size"] == 6 + 2 * 4 + 4
	assert set(layout["extra"]) == {"db_dist", "db_angle", "centering_a", "centering_b"}


def test_equations_evaluate_including_chained_b1c():
	lat, motif, extra = _si_config()
	p, _fit, _eq_mask, _eq_funcs = dh.dicts_to_vector(lat, motif, extra)
	layout = dh.build_layout(lat, motif, extra)
	ca, cb = 0.5, 0.5
	# A_1c = A_1 + centering = (0.5, 0.5)
	ix, iy = layout["motif"]["A_1c"]
	assert np.isclose(p[ix], ca) and np.isclose(p[iy], cb)
	# B_1 = A_1 + dumbbell; db_angle 0 -> x = 0 (sin 0), y = db_dist / b
	jx, jy = layout["motif"]["B_1"]
	assert np.isclose(p[jx], 0.0)
	assert np.isclose(p[jy], 0.136 / 0.5469)
	# B_1c = B_1 + centering -- chained on the equation-driven B_1, so this only
	# holds if inflate_params evaluates equation-on-equation in the right order
	kx, ky = layout["motif"]["B_1c"]
	assert np.isclose(p[kx], p[jx] + ca)
	assert np.isclose(p[ky], p[jy] + cb)


def test_independent_index_excludes_fixed_and_all_equation_atoms():
	lat, motif, extra = _si_config()
	_p, fit, eq_mask, _eq_funcs = dh.dicts_to_vector(lat, motif, extra)
	indep = dh.build_independent_index(fit, eq_mask)
	layout = dh.build_layout(lat, motif, extra)
	# independent = 6 lattice (all fitted) + 4 extra (all fitted, none eq) = 10
	assert len(indep) == 10
	# A_1 is fixed; A_1c / B_1 / B_1c are equation-driven -> none are independent
	for label in ("A_1", "A_1c", "B_1", "B_1c"):
		ix, iy = layout["motif"][label]
		assert ix not in indep and iy not in indep


def test_roundtrip_unpack_then_revectorize_is_idempotent():
	lat, motif, extra = _si_config()
	p1, *_ = dh.dicts_to_vector(lat, motif, extra)
	dh.unpack_to_dicts(p1, lat, motif, extra)   # mutate the dicts to reflect p1
	p2, *_ = dh.dicts_to_vector(lat, motif, extra)
	assert np.allclose(p1, p2)


def test_compile_eq_accepts_arithmetic_and_indexing():
	code = dh._compile_eq("lat_params['abg'][0] + extra_pars['centering_a']*2")
	assert code is not None


def test_compile_eq_rejects_disallowed_syntax():
	# nodes outside ALLOWED_NODES: comprehension, lambda, ternary, comparison
	for bad in ("[i for i in range(3)]", "lambda x: x", "a if b else c", "1 < 2"):
		with pytest.raises(ValueError, match="Disallowed syntax"):
			dh._compile_eq(bad)


def test_equation_sandbox_blocks_builtins():
	# __import__('os') compiles (Call/Name are allowed) but the eval env strips
	# __builtins__, so evaluating the equation raises NameError.
	lat = {"abg": [1.0, 1.0, 90.0], "fit_abg": [False, False, False],
	       "base": [0.0, 0.0, 0.0], "fit_base": [False, False, False]}
	motif = {"A_1": {"atom": "X", "coord": (0.0, 0.0), "use": True,
	                 "fit": [False, False], "eq": ["= __import__('os')", None]}}
	with pytest.raises(NameError):
		dh.dicts_to_vector(lat, motif, {})


if __name__ == "__main__":
	for _name, _fn in sorted(globals().items()):
		if _name.startswith("test_") and callable(_fn):
			_fn()
			print(f"  ok  {_name}")
	print("all tests passed")
