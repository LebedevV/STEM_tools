#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the config-driven runner's fit-mask + refinement_run passthrough.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmap_run as vr
from vmap_config import Pass, load_config

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "fit_si.toml")


def _state():
	lp = {"fit_abg": [True] * 3, "fit_base": [True] * 3}
	mo = {"A_1": {"fit": [False, False]}}
	ep = {"db_dist": (0.136, True), "centering_a": (0.5, False)}
	return lp, mo, ep


def test_fit_mask_toggles_all_three_categories():
	# lattice, motif label, and extra_par are homogeneous keys in one mask
	lp, mo, ep = _state()
	vr._apply_fit({"abg": [False, False, False], "A_1": [True, True], "db_dist": [False]}, lp, mo, ep)
	assert lp["fit_abg"] == [False, False, False]
	assert mo["A_1"]["fit"] == [True, True]
	assert ep["db_dist"] == (0.136, False)          # value kept, flag flipped


def test_fit_mask_rejects_eq_coupled_extra_par():
	lp, mo, _ = _state()
	with pytest.raises(KeyError, match="eq-coupled"):
		vr._apply_fit({"x": [True]}, lp, mo, {"x": (1.0, "= foo")})


def test_fit_mask_rejects_unknown_key():
	lp, mo, ep = _state()
	with pytest.raises(KeyError, match="unknown param"):
		vr._apply_fit({"nope": [True]}, lp, mo, ep)


def test_passthrough_forwards_known_kwarg():
	p = Pass(name="t", recall_zero=True, export_sublattice_xy=True)
	assert vr._passthrough(p) == {"recall_zero": True, "export_sublattice_xy": True}


def test_passthrough_rejects_non_kwarg():
	with pytest.raises(KeyError, match="not a refinement_run kwarg"):
		vr._passthrough(Pass(name="t", bogus_kw=1))


def test_passthrough_rejects_runner_owned():
	with pytest.raises(KeyError, match="set by the runner"):
		vr._passthrough(Pass(name="t", do_fit=False))


def test_example_si_config_round_trips():
	cfg = load_config(EXAMPLE)
	names = [p.name for p in cfg.run.passes]
	assert names == ["lattice", "dumbbell", "free"]
	# stages toggle the dumbbell extra_pars through the fit mask
	assert cfg.run.passes[0].fit["db_dist"] == [False]
	assert cfg.run.passes[1].fit["db_dist"] == [True]
	# behavioural flags land as passthrough, not schema fields
	assert vr._passthrough(cfg.run.passes[0]) == {"recall_zero": True}
	assert vr._passthrough(cfg.run.passes[2]) == {"export_sublattice_xy": True}
