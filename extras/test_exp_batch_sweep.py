#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the experimental-batch orchestration: _select / _pixel_imsize /
# _propagate_sidecar / _run_row, with the heavy align+detect steps stubbed.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np
import pandas as pd
import tifffile

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
    return {"raw_path": str(raw)}


def test_select_filters_by_type():
    df = pd.DataFrame({"ext": [".emd", ".dm3", ".emd"], "n_frames": [8, 16, 8]})
    assert len(sw._select(df, {"ext": ".emd"})) == 2
    assert len(sw._select(df, {"n_frames": 16})) == 1
    assert len(sw._select(df, {})) == 3


def test_pixel_imsize_matches_frame_dims(tmp_path):
    # detect_columns reads imsize_px = (cols, rows) = tiff.shape[::-1], so _pixel_imsize
    # must return the same -> detection scale is exactly 1 (x_obs0/y_obs0 in pixels).
    p = tmp_path / "s_RA.tiff"
    tifffile.imwrite(str(p), np.zeros((3, 5), dtype=np.float32))   # (rows=3, cols=5)
    assert sw._pixel_imsize(str(p)) == (5, 3)


def test_propagate_sidecar_copies_verbatim(tmp_path):
    folder = str(tmp_path) + os.sep
    body = "xres_px\t1024\nyres_px\t1024\nxreal_nm\t8.0\nyreal_nm\t8.0\n"
    (tmp_path / "s_frame.txt").write_text(body)
    assert sw._propagate_sidecar(folder, "s", "s_RA") is True
    out = tmp_path / "s_RA_frame.txt"
    assert out.exists() and out.read_text() == body          # nm/px copied verbatim
    # absent sidecar -> no copy, returns False
    assert sw._propagate_sidecar(folder, "nope", "nope_RA") is False
    assert not (tmp_path / "nope_RA_frame.txt").exists()


def test_run_row_align_detect_sidecar_then_csv(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(sw, "_align", lambda raw, folder, stem, cfg: seen.update(stem=stem))

    def fake_detect(aligned, folder, cfg):
        open(os.path.join(folder, aligned + "_xyI.csv"), "w").close()
    monkeypatch.setattr(sw, "_detect", fake_detect)

    (tmp_path / "s_frame.txt").write_text("xres_px\t1\nyres_px\t1\nxreal_nm\t0.5\nyreal_nm\t0.5\n")
    rec = sw._run_row(_row(tmp_path), _cfg())
    assert rec["status"] == "ok" and seen["stem"] == "s"
    assert rec["xyI_path"].endswith("s_RA_xyI.csv") and os.path.exists(rec["xyI_path"])
    assert (tmp_path / "s_RA_frame.txt").exists()            # sidecar propagated to aligned stem


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
