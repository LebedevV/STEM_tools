#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the batch sweep's per-row calibration handling.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vmap_sweep

TMPL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "fit_pm3m.toml")


def test_run_row_skips_without_descriptive_toml(capsys):
	# the sweep self-calibrates from each frame's descriptive toml; a row without its scan_s
	# is skipped + logged, not silently fit with a fixed template calibration
	out = vmap_sweep._run_row({"tiff_path": "/d/frame.tif", "scan_s": ""}, TMPL, 1)
	assert out == (None, None, None, {})
	assert "no descriptive toml" in capsys.readouterr().out


def test_run_row_calibrates_from_manifest_scan_s(monkeypatch):
	# with scan_s in the manifest row, calibration is frame_size (the manifest value); the
	# toml is NOT reopened. the fit is stubbed.
	captured = {}
	def fake_run(cfg, gui=False):
		captured["cal"] = cfg.calibration.model_dump()
		return {"abg": [0.4, 0.4, 90.0]}, {}, {}, {}
	monkeypatch.setattr(vmap_sweep.vmap_run, "run", fake_run)
	vmap_sweep._run_row({"tiff_path": "/d/Pm3m_(0.0, 0.0)_110_haadf.tif", "scan_s": 50.0}, TMPL, 1)
	assert captured["cal"]["source"] == "frame_size"
	assert captured["cal"]["frame_size"] == 50.0 and captured["cal"]["toml_path"] is None
