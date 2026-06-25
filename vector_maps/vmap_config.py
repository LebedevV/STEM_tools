#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Pydantic schema for the config-driven refinement runner (see DESIGN.md).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import tomllib
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Io(BaseModel):
    model_config = ConfigDict(extra="forbid")
    folder: str = "./"
    fname: str


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["sidecar", "value", "frame_size"] = "sidecar"  # _frame.txt | inline | toml frame_size
    value: Optional[float] = None
    frame_size: Optional[float] = None                             # source="frame_size": scan_s (Angstrom), /n_px
    toml_path: Optional[str] = None                                # source="frame_size": read scan_s from this toml

    @model_validator(mode="after")
    def _frame_size_source(self):
        if self.source == "frame_size" and (self.frame_size is None) == (self.toml_path is None):
            raise ValueError("calibration: source='frame_size' needs exactly one of frame_size or toml_path")
        return self


class Lattice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abg: list[float]
    base: list[float]
    fit_abg: list[bool] = [True, True, True]
    fit_base: list[bool] = [True, True, True]


class MotifAtom(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    label: str
    el: str
    coord: list[float]
    fit: list[bool] = [False, False]
    intensity: float = Field(default=1.0, alias="I")
    use: bool = True
    eq: Optional[list[str]] = None

    @field_validator("eq")
    @classmethod
    def _eq_prefixed(cls, v):
        return v if v is None else [e if e.lstrip().startswith("=") else "= " + e for e in v]


class ExtraPar(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    fit: Optional[bool] = None
    eq: Optional[str] = None

    @field_validator("eq")
    @classmethod
    def _eq_prefixed(cls, v):
        return v if v is None or v.lstrip().startswith("=") else "= " + v


class Expand(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    start: list[float] = Field(alias="from")
    to: list[float]
    step: float

    @field_validator("step")
    @classmethod
    def _step_positive(cls, v):
        if v <= 0:
            raise ValueError("expand.step must be > 0")
        return v


class Detect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ptonn: float = 0.6                       # percent_to_nn fit window (scalar; one detection per step)
    sep: float = 2.0
    sigma1: float = 1.0
    thr: float = 0.1
    imsize: list[float]                      # nm; required by detect_columns
    pca: bool = False
    subtract_background: bool = True
    source: Optional[str] = None             # detect on this image instead of <fname>.tif, e.g. a prior
                                             # detect step's "{fname}_2DG_ptnn_<ptonn>_diff2.tif" residual
    accrete: bool = False                    # False = reset (replace <fname>_xyI.csv, old -> .bckp1/2/3);
                                             # True  = accrete (concat this detection onto the working set)
    save_as: Optional[str] = None            # accrete output stem ({fname}/{name} templated); required iff accrete

    @model_validator(mode="after")
    def _save_as_iff_accrete(self):
        if self.accrete and not self.save_as:
            raise ValueError("detect: accrete=true needs save_as (the merged-output stem)")
        if self.save_as and not self.accrete:
            raise ValueError("detect: save_as is for accrete only; reset writes <fname>_xyI.csv")
        return self


class Pass(BaseModel):
    # extra="allow": any unrecognised key is treated as a refinement_run kwarg and
    # passed through (validated against its signature in vmap_run._run_pass).
    model_config = ConfigDict(extra="allow")
    name: str = "pass"
    sub_area: Optional[list[float]] = None
    vec_scale: float = 0.05
    max_dist: float = 0.0
    save: bool = False
    gui: bool = False
    refine: bool = True
    fit: dict[str, list[bool]] = {}
    add: list[MotifAtom] = []
    expand: Optional[Expand] = None
    body: list["Pass"] = []
    detect: Optional[Detect] = None


class Run(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gui: bool = True                  # master allow; per-pass gui decides, --no-gui forces off
    seed: bool = False
    seed_file: str = "{fname}.start.toml"
    save_stem: str = "{fname}_{name}"  # saved-pass output folder; {fname}=frame stem, {name}=pass
    unit_cell: bool = False            # after the fit, save the averaged unit cell (<fname>_uc_*.tif + _uc_figure.png)
    passes: list[Pass]


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    io: Io
    calibration: Calibration = Calibration()
    lattice: Lattice
    motif: list[MotifAtom]
    extra_pars: dict[str, ExtraPar] = {}
    run: Run


Pass.model_rebuild()


def load_config(path) -> AppConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return AppConfig.model_validate(data)


# ---- batch sweep schema ----------------------------------------------------

class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Optional[str] = None    # recursively walk this tree -> build the manifest in-process
    path: Optional[str] = None    # ...or point at a pre-built manifest CSV

    @model_validator(mode="after")
    def _one_source(self):
        if bool(self.root) == bool(self.path):
            raise ValueError("manifest: set exactly one of 'root' (walk a tree) or 'path' (pre-built CSV)")
        return self


class FitRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    config: str


class SweepRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retries: int = 1


class MapSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field: str
    title: str = ""
    significant: Optional[str] = None
    scale: float = 1.0


class BatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest: Manifest
    filter: dict[str, Any] = {}
    fit: FitRef
    run: SweepRun = SweepRun()
    maps: list[MapSpec] = []


def load_batch(path) -> BatchConfig:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return BatchConfig.model_validate(data)
