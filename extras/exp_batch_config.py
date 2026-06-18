#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Config schema for the experimental batch: align raw stacks -> detect columns.
# Mirrors vector_maps/vmap_config (Manifest root-XOR-path), for the front-end.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import tomllib
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Optional[str] = None    # recursively walk this tree of raw stacks
    path: Optional[str] = None    # ...or point at a pre-built catalog CSV

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.root) == bool(self.path):
            raise ValueError("manifest: set exactly one of 'root' (walk a tree) or 'path' (pre-built CSV)")
        return self


class Align(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nra: bool = False             # also write _NRA.tiff (demon non-rigid registration)
    bin_factor: int = 1
    use: str = "RA"               # which averaged frame feeds detection: "RA" | "NRA"

    @field_validator("bin_factor")
    @classmethod
    def _bin_positive(cls, v):
        if v < 1:
            raise ValueError("align.bin_factor must be >= 1")
        return v

    @field_validator("use")
    @classmethod
    def _use_known(cls, v):
        if v not in ("RA", "NRA"):
            raise ValueError("align.use must be 'RA' or 'NRA'")
        return v

    @model_validator(mode="after")
    def _use_needs_nra(self):
        if self.use == "NRA" and not self.nra:
            raise ValueError("align.use='NRA' requires align.nra=true")
        return self


class Detect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sep: float = 8.0
    sigma1: float = 1.0
    thr: float = 0.1
    ptonn: float = 0.6
    pca: bool = True
    subtract_background: bool = True


class ExtrasRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retries: int = 1
    skip_existing: bool = True             # skip a frame whose _xyI.csv already exists
    chain_fit: Optional[str] = None        # optional vmap fit toml; None -> stop at the csv


class ExtrasBatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest: Manifest
    filter: dict[str, Any] = {}
    align: Align = Align()
    detect: Detect = Detect()
    run: ExtrasRun = ExtrasRun()


def load_batch_extras(path) -> ExtrasBatchConfig:
    with open(path, "rb") as f:
        return ExtrasBatchConfig.model_validate(tomllib.load(f))
