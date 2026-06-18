#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the experimental-batch catalog builder (synthetic tree + stub meta_reader).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_batch_manifest as m


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def _stub_reader(nf=8, ny=512, nx=512, px=0.01):
    # same metadata for any path; nf<2 marks a non-stack, px=None a calibration miss
    return lambda path: (nf, ny, nx, px, "stub")


def test_walks_tree_one_row_per_stack(tmp_path):
    _touch(str(tmp_path / "a" / "s1.emd"))
    _touch(str(tmp_path / "a" / "b" / "s2.dm3"))
    _touch(str(tmp_path / "a" / "notes.txt"))          # non-raw -> ignored
    rows, _skipped = m.build_manifest(str(tmp_path), meta_reader=_stub_reader())
    assert len(rows) == 2
    assert sorted(r["source"] for r in rows) == [os.path.join("a"), os.path.join("a", "b")]
    r0 = next(r for r in rows if r["ext"] == ".emd")
    assert r0["scan_size_nm"] == 0.01 * 512 and r0["meta_ok"] is True


def test_non_stack_2d_is_skipped(tmp_path):
    _touch(str(tmp_path / "img.emd"))
    rows, skipped = m.build_manifest(str(tmp_path), meta_reader=_stub_reader(nf=1))
    assert rows == [] and len(skipped) == 1


def test_unreadable_file_skipped(tmp_path):
    _touch(str(tmp_path / "bad.emd"))
    rows, skipped = m.build_manifest(str(tmp_path), meta_reader=lambda p: None)
    assert rows == [] and len(skipped) == 1


def test_missing_pixel_size_kept_but_flagged(tmp_path):
    _touch(str(tmp_path / "s.dm4"))
    rows, _ = m.build_manifest(str(tmp_path), meta_reader=_stub_reader(px=None))
    assert len(rows) == 1 and rows[0]["meta_ok"] is False and rows[0]["scan_size_nm"] == ""


def test_write_manifest_round_trip(tmp_path):
    _touch(str(tmp_path / "s.emd"))
    rows, _ = m.build_manifest(str(tmp_path), meta_reader=_stub_reader())
    out = tmp_path / "catalog.csv"
    m.write_manifest(rows, str(out))
    with open(out) as f:
        got = list(csv.DictReader(f))
    assert len(got) == 1 and got[0]["ext"] == ".emd" and set(got[0]) == set(m._COLS)
