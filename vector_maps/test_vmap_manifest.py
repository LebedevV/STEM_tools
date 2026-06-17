#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for vmap_manifest: recursive tree walk + path-to-source column.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vmap_manifest import build_manifest


def _touch(path):
    open(path, "a").close()


def test_walks_tree_and_tags_source():
    # frames in two sibling run folders under one root -> one manifest, tagged by source
    root = tempfile.mkdtemp()
    a = os.path.join(root, "runA")
    b = os.path.join(root, "runB")
    os.makedirs(a)
    os.makedirs(b)
    _touch(os.path.join(a, "Pm3m_(25.0, 10.0)_110_haadf_0-25.tif"))
    _touch(os.path.join(b, "fph_Pm3m_(25.0, 5.0)_110_abf.tif"))
    _touch(os.path.join(a, "random_note.tif"))      # tiff, non-matching -> skipped
    _touch(os.path.join(b, "log.txt"))              # non-tiff -> ignored silently

    rows, skipped = build_manifest(root)

    assert len(rows) == 2
    by_src = {r["source"]: r for r in rows}
    assert set(by_src) == {"runA", "runB"}
    assert by_src["runA"]["hkl"] == "110" and by_src["runA"]["detector"] == "haadf"
    assert by_src["runA"]["blur_sigma"] == 0.25
    assert by_src["runB"]["is_fph"] is True and by_src["runB"]["detector"] == "abf"
    assert os.path.dirname(by_src["runA"]["tiff_path"]) == a   # absolute, into its subfolder
    assert any("random_note.tif" in s for s in skipped)        # skipped reported with path
    assert not any("log.txt" in s for s in skipped)


def test_toml_sidecar_resolved_per_subfolder():
    # the sidecar must be read from the frame's own (nested) dir, not the walk root
    root = tempfile.mkdtemp()
    sub = os.path.join(root, "deep", "runC")
    os.makedirs(sub)
    _touch(os.path.join(sub, "Pm3m_(0.0, 0.0)_010_haadf.tif"))
    with open(os.path.join(sub, "Pm3m_010_(0.0, 0.0).toml"), "w") as f:
        f.write("[lamella_settings]\nscan_s = 50.0\nthickness = 40.0\nborders = 5.0\n")
        f.write("[simulations]\nfrozen_phonons = 8\nfph_sigma = 0.1\n")

    rows, _ = build_manifest(root)

    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == os.path.join("deep", "runC")
    assert r["scan_s"] == 50.0 and r["phonons"] == 8 and r["toml_path"]


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  ok  {_name}")
    print("all tests passed")
