#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the experimental-batch orchestration (_select + _run_row, align/detect stubbed).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_batch_sweep as sw
from exp_batch_config import ExtrasBatchConfig


def _cfg(**over):
    d = {"manifest": {"root": "data"}}
    d.update(over)
    return ExtrasBatchConfig.model_validate(d)


def _row(tmp_path, name="s.emd"):
    raw = tmp_path / name
    raw.write_bytes(b"x")
    return {"raw_path": str(raw), "pixel_size_nm": 0.02, "nx": 100, "ny": 100}


def test_select_filters_by_type():
    df = pd.DataFrame({"ext": [".emd", ".dm3", ".emd"], "n_frames": [8, 16, 8]})
    assert len(sw._select(df, {"ext": ".emd"})) == 2
    assert len(sw._select(df, {"n_frames": 16})) == 1
    assert len(sw._select(df, {})) == 3


def test_imsize_derived_from_pixel_size():
    row = {"pixel_size_nm": 0.02, "nx": 100, "ny": 50}
    assert sw._imsize(row, _cfg().detect) == [2.0, 1.0]
    c = _cfg(detect={"imsize": [16.0, 16.0]})
    assert sw._imsize(row, c.detect) == [16.0, 16.0]


def test_run_row_align_detect_then_csv(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(sw, "_align", lambda raw, folder, stem, cfg: seen.update(stem=stem))

    def fake_detect(aligned, folder, imsize, cfg):
        seen["imsize"] = imsize
        open(os.path.join(folder, aligned + "_xyI.csv"), "w").close()
    monkeypatch.setattr(sw, "_detect", fake_detect)

    rec = sw._run_row(_row(tmp_path), _cfg())
    assert rec["status"] == "ok"
    assert rec["xyI_path"].endswith("s_RA_xyI.csv") and os.path.exists(rec["xyI_path"])
    assert seen["stem"] == "s" and seen["imsize"] == [2.0, 2.0]


def test_run_row_skip_existing(tmp_path, monkeypatch):
    called = {"align": 0}
    monkeypatch.setattr(sw, "_align", lambda *a, **k: called.update(align=called["align"] + 1))
    monkeypatch.setattr(sw, "_detect", lambda *a, **k: None)
    open(os.path.join(str(tmp_path), "s_RA_xyI.csv"), "w").close()   # pre-existing output
    rec = sw._run_row(_row(tmp_path), _cfg())
    assert rec["status"] == "skipped" and called["align"] == 0


def test_run_row_retries_then_error(tmp_path, monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("align failed")
    monkeypatch.setattr(sw, "_align", boom)
    monkeypatch.setattr(sw, "_detect", lambda *a, **k: None)
    rec = sw._run_row(_row(tmp_path), _cfg(run={"retries": 2}))
    assert rec["status"].startswith("error") and calls["n"] == 2 and rec["xyI_path"] == ""
