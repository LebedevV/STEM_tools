#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the config-driven runner's fit-mask + refinement_run passthrough.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmap_run as vr
from vmap_config import Detect, Pass, load_config

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


def test_save_folder_includes_fname(monkeypatch):
	# the saved-pass folder uses run.save_stem -> "<fname>_<pass>" by default, so frames
	# don't collide in a bare "<pass>/" dir; refinement_run is stubbed to capture sf.
	captured = {}

	def fake_run(folder, sf, fname, calib, *a, **k):
		captured["sf"] = sf
		return {}, None
	monkeypatch.setattr(vr, "refinement_run", fake_run)
	monkeypatch.setattr(vr, "unpack_to_dicts", lambda *a, **k: None)

	vr._run_pass(Pass(name="free", save=True), "fld/", "myframe", 0.01, {}, {}, {},
		     False, True, save_stem="{fname}_{name}")
	assert captured["sf"] == "myframe_free"
	# the old bare-pass naming stays available via the template
	vr._run_pass(Pass(name="free", save=True), "fld/", "myframe", 0.01, {}, {}, {},
		     False, True, save_stem="{name}")
	assert captured["sf"] == "free"
	# a non-saving pass writes nowhere
	captured["sf"] = "untouched"
	vr._run_pass(Pass(name="prefit", save=False), "fld/", "myframe", 0.01, {}, {}, {},
		     False, True)
	assert captured["sf"] is None


def test_run_unit_cell_flag_defaults_off_and_example_parses():
	# the averaged-unit-cell save is opt-in; the re-atomap example turns it on and
	# carries a re-detect step (PZT-style chained detection).
	from vmap_config import Run
	assert Run(passes=[Pass(name="p")]).unit_cell is False
	cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)),
				       "examples", "fit_reatomap.toml"))
	assert cfg.run.unit_cell is True
	det = [p.detect for p in cfg.run.passes if p.detect is not None]
	assert len(det) == 1 and det[0].ptonn == [0.6, 0.4] and det[0].accrete is True


def test_rotate_backup_rotates_and_caps(tmp_path):
	# path -> .bckp1, older ones shift down, only `keep` (3) survive
	p = os.path.join(str(tmp_path), "m.csv")
	for tag in ["A", "B", "C", "D"]:
		with open(p, "w") as f:
			f.write(tag)
		vr._rotate_backup(p)
	assert not os.path.exists(p)                       # last rotate moved it to .bckp1
	assert open(p + ".bckp1").read() == "D"
	assert open(p + ".bckp2").read() == "C"
	assert open(p + ".bckp3").read() == "B"
	assert not os.path.exists(p + ".bckp4")            # "A" dropped off


def test_run_detect_reset_replaces_and_backs_up(tmp_path, monkeypatch):
	# reset (default): fresh detection overwrites <fname>_xyI.csv, old -> .bckp1, returns None
	import sys as _sys
	import types
	folder = os.path.join(str(tmp_path), "")
	with open(os.path.join(folder, "frame_xyI.csv"), "w") as f:
		f.write("OLD")

	def fake_detect(**kw):
		assert kw["out_suffix"] == ""                  # reset writes the canonical name
		pd.DataFrame({"x_obs0": [1.0, 2.0], "y_obs0": [0.0, 0.0]}).to_csv(
			os.path.join(kw["folder"], os.path.splitext(kw["fname"])[0] + "_xyI.csv"), index=False)
	fake_mod = types.ModuleType("detect_columns")
	fake_mod.detect_columns = fake_detect
	monkeypatch.setitem(_sys.modules, "detect_columns", fake_mod)

	out = vr._run_detect(Detect(ptonn=[0.6], imsize=[10.0, 10.0]), folder, "frame")

	assert out is None                                 # fit reads the canonical csv
	assert open(os.path.join(folder, "frame_xyI.csv.bckp1")).read() == "OLD"
	assert len(pd.read_csv(os.path.join(folder, "frame_xyI.csv"))) == 2


def test_run_detect_accrete_concats_no_dedup(tmp_path, monkeypatch):
	# accrete: chained passes -> _sub_A/_sub_B -> concat into _sub_AB, NO dedup
	# (a near-coincident cross-pass pair is kept); <fname>_xyI.csv untouched
	import sys as _sys
	import types
	folder = os.path.join(str(tmp_path), "")
	with open(os.path.join(folder, "frame_xyI.csv"), "w") as f:
		f.write("OLD")
	rows = {"_sub_A": [(0.0, 0.0), (20.0, 0.0)],
		"_sub_B": [(0.1, 0.0), (40.0, 0.0)]}           # (0.1,0) ~ A's (0,0): kept anyway

	def fake_detect(**kw):
		pts = rows[kw["out_suffix"]]
		pd.DataFrame({"x_obs0": [p[0] for p in pts], "y_obs0": [p[1] for p in pts]}).to_csv(
			os.path.join(kw["folder"], os.path.splitext(kw["fname"])[0] + kw["out_suffix"] + "_xyI.csv"),
			index=False)
	fake_mod = types.ModuleType("detect_columns")
	fake_mod.detect_columns = fake_detect
	monkeypatch.setitem(_sys.modules, "detect_columns", fake_mod)

	stem = vr._run_detect(Detect(ptonn=[0.6, 0.4], imsize=[10.0, 10.0], accrete=True), folder, "frame")

	assert stem == "frame_sub_AB"
	out = pd.read_csv(os.path.join(folder, "frame_sub_AB_xyI.csv"))
	assert len(out) == 4                               # all kept -- no dedup
	assert sorted(out["sub_id"].tolist()) == ["A", "A", "B", "B"]
	assert open(os.path.join(folder, "frame_xyI.csv")).read() == "OLD"  # canonical untouched
