#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for vmap_manifest: recursive tree walk + path-to-source column.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vmap_manifest import build_manifest, write_manifest


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
        f.write('[paths]\nsample_name = "PZO"\n')
        f.write("[microscope]\nHT_value = 200000\nhaadfinner = 99.0\n")
        f.write('detectors = ["haadf", "abf"]\naberrations = {C30 = 1.0}\n')
        f.write('[lamella_settings]\nscan_s = 50.0\nthickness = 40.0\nborders = 5.0\natom_to_zero = "Zr"\n')
        f.write("[simulations]\nfrozen_phonons = 8\nfph_sigma = 0.1\nblur_sigmas = [0.1, 0.25]\n")

    rows, _ = build_manifest(root)

    assert len(rows) == 1
    r = rows[0]
    assert r["source"] == os.path.join("deep", "runC")
    # scalar physics params harvested; frozen_phonons exposed as the load-bearing `phonons`
    assert r["scan_s"] == 50.0 and r["phonons"] == 8 and r["fph_sigma"] == 0.1 and r["toml_path"]
    assert r["HT_value"] == 200000 and r["sample_name"] == "PZO" and r["atom_to_zero"] == "Zr"
    assert "frozen_phonons" not in r
    # list/dict fields don't fit a column -> skipped
    assert "detectors" not in r and "aberrations" not in r and "blur_sigmas" not in r


def test_harvest_does_not_clobber_structural_columns(tmp_path):
    # a (future) toml field colliding with a structural column name must not overwrite it
    sub = os.path.join(str(tmp_path), "runX")
    os.makedirs(sub)
    _touch(os.path.join(sub, "Pm3m_(0.0, 0.0)_010_haadf.tif"))
    with open(os.path.join(sub, "Pm3m_010_(0.0, 0.0).toml"), "w") as f:
        f.write('[lamella_settings]\nscan_s = 50.0\ndetector = "FAKE"\nhkl = "999"\n')

    rows, _ = build_manifest(str(tmp_path))

    r = rows[0]
    assert r["detector"] == "haadf" and r["hkl"] == "010"   # from the filename, not the toml
    assert r["scan_s"] == 50.0                              # non-colliding harvest still lands


def test_write_manifest_appends_harvested_columns(tmp_path):
    # write_manifest keeps the known _COLS, then appends harvested params (sorted)
    import csv
    rows = [{"tiff_path": "/d/f.tif", "scan_s": 50.0, "HT_value": 200000, "sample_name": "PZO"}]
    out = os.path.join(str(tmp_path), "manifest.csv")
    write_manifest(rows, out)
    with open(out) as f:
        header = next(csv.reader(f))
    assert "scan_s" in header                                  # known col kept
    assert "HT_value" in header and "sample_name" in header    # harvested cols appended


def test_manifest_requires_exactly_one_source():
    # batch.toml [manifest]: exactly one of root (walk a tree) / path (pre-built CSV)
    import pydantic
    from vmap_config import Manifest
    assert Manifest(root="some/tree").root == "some/tree"
    assert Manifest(path="manifest.csv").path == "manifest.csv"
    for bad in ({}, {"root": "t", "path": "m.csv"}):
        try:
            Manifest(**bad)
        except pydantic.ValidationError:
            continue
        raise AssertionError(f"expected ValidationError for {bad}")


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  ok  {_name}")
    print("all tests passed")
