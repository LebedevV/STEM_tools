#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the experimental-batch config schema (pure pydantic; no microscope deps).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_batch_config as cfgmod


def test_manifest_root_xor_path():
    cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"root": "data"}})
    cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"path": "catalog.csv"}})
    for bad in ({}, {"root": "d", "path": "c.csv"}):
        with pytest.raises(ValidationError):
            cfgmod.ExtrasBatchConfig.model_validate({"manifest": bad})


def test_defaults_round_trip():
    c = cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"root": "data"}})
    assert c.align.nra is False and c.align.bin_factor == 1 and c.align.use == "RA"
    assert c.detect.imsize is None and c.run.retries == 1 and c.run.skip_existing is True
    assert c.run.chain_fit is None


def test_align_bin_factor_must_be_positive():
    with pytest.raises(ValidationError):
        cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"root": "d"}, "align": {"bin_factor": 0}})


def test_align_nra_use_consistency():
    with pytest.raises(ValidationError):
        cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"root": "d"}, "align": {"use": "NRA"}})
    c = cfgmod.ExtrasBatchConfig.model_validate(
        {"manifest": {"root": "d"}, "align": {"use": "NRA", "nra": True}})
    assert c.align.use == "NRA"


def test_extra_keys_forbidden():
    with pytest.raises(ValidationError):
        cfgmod.ExtrasBatchConfig.model_validate({"manifest": {"root": "d"}, "bogus": 1})


def test_load_batch_extras_round_trip(tmp_path):
    p = tmp_path / "b.toml"
    p.write_text('[manifest]\nroot = "data"\n[detect]\nsep = 6.0\n[run]\nretries = 3\n')
    c = cfgmod.load_batch_extras(str(p))
    assert c.manifest.root == "data" and c.detect.sep == 6.0 and c.run.retries == 3


if __name__ == "__main__":
    import inspect
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn) and not inspect.signature(_fn).parameters:
            _fn()
            print(f"  ok  {_name}")
    print("config tests passed (fixture-using tests run under pytest)")
