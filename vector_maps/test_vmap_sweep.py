#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the batch sweep's per-row calibration handling.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vmap_sweep

TMPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", "fit_pm3m.toml")


def test_run_row_skips_without_descriptive_toml(capsys):
	# the sweep self-calibrates from each frame's descriptive toml; a row without one
	# is skipped + logged, not silently fit with a fixed template calibration
	out = vmap_sweep._run_row({"tiff_path": "/d/frame.tif", "toml_path": ""}, TMPL, 1)
	assert out == (None, None, None, {})
	assert "no descriptive toml" in capsys.readouterr().out
