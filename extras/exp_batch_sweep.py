#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Experimental batch: register raw stacks, then detect atom columns per frame.
#   python exp_batch_sweep.py --config examples/batch_exp.toml
# Reuses extras/batch.py (alignment) + vector_maps/detect_columns (the _xyI.csv
# producer). Detection runs in pixels (calibration-free); the per-.dm3 sidecar
# <name>_frame.txt is copied onto the aligned stem so a source="sidecar" fit gets
# the nm/px (invariant under the alignment crop). The align + detect steps need
# real stacks; the orchestration here is unit-tested with _align/_detect stubbed.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import tifffile

from exp_batch_config import load_batch_extras
from exp_batch_manifest import build_manifest


def _ensure_vector_maps_on_path():
    # detect_columns / vmap_run live in the sibling vector_maps package
    vm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vector_maps")
    if vm not in sys.path:
        sys.path.insert(0, vm)


def _select(df, flt):
    # generic catalog filter (same semantics as vector_maps/vmap_sweep._select)
    mask = pd.Series(True, index=df.index)
    for k, v in flt.items():
        if k not in df.columns:
            raise KeyError(f"filter column '{k}' not in catalog {list(df.columns)}")
        col = df[k]
        if isinstance(v, bool):
            mask &= col.astype(str).str.lower().isin(["true", "1", "yes"]) == v
        elif isinstance(v, (int, float)):
            mask &= np.isclose(pd.to_numeric(col, errors="coerce"), float(v))
        else:
            mask &= col.astype(str) == str(v)
    return df.loc[mask].copy()


def _pixel_imsize(aligned_path):
    # the aligned frame's own pixel dims, in detect_columns' axis order
    # (hyperspy axes_manager[0]=x=cols, [1]=y=rows == tiff shape reversed); passing
    # these back as imsize makes the detection scale exactly 1 -> x_obs0 in pixels,
    # which is what the fit expects (preprocess_dataset applies calib to pixels).
    return tuple(int(n) for n in tifffile.imread(aligned_path).shape[::-1][:2])


def _propagate_sidecar(folder, raw_stem, aligned_stem):
    # nm/px survives the alignment crop, so copy the per-.dm3 <raw>_frame.txt sidecar
    # onto the aligned stem; a source="sidecar" fit then reads the same calibration.
    src = os.path.join(folder, f"{raw_stem}_frame.txt")
    if not os.path.exists(src):
        return False
    shutil.copyfile(src, os.path.join(folder, f"{aligned_stem}_frame.txt"))
    return True


# --- heavy steps, factored out so the sweep is testable with these stubbed -------

def _align(raw, folder, stem, cfg):
    import hyperspy.api as hs
    import batch
    s = hs.load(raw, lazy=True)
    batch.alignment(s, folder, stem, NRA=cfg.align.nra, bin_factor=cfg.align.bin_factor)


def _detect(aligned, folder, cfg):
    _ensure_vector_maps_on_path()
    from detect_columns import detect_columns
    detect_columns(
        fname=aligned + ".tiff", folder=folder,
        imsize=_pixel_imsize(os.path.join(folder, aligned + ".tiff")),
        sep=cfg.detect.sep, sigma1=cfg.detect.sigma1, thr=cfg.detect.thr,
        ptonn=cfg.detect.ptonn, pca=cfg.detect.pca,
        subtract_background=cfg.detect.subtract_background, interactive=False,
    )


def _chain_fit(fit_cfg_path, folder, aligned):
    import tomllib
    _ensure_vector_maps_on_path()
    import vmap_run
    from vmap_config import AppConfig
    with open(fit_cfg_path, "rb") as f:
        data = tomllib.load(f)
    data.setdefault("io", {})["folder"] = folder
    data["io"]["fname"] = aligned
    vmap_run.run(AppConfig.model_validate(data), gui=False)


def _run_row(row, cfg):
    raw = str(row["raw_path"])
    folder = os.path.join(os.path.dirname(raw), "")
    stem = os.path.splitext(os.path.basename(raw))[0]
    aligned = f"{stem}_{cfg.align.use}"
    csv_path = os.path.join(folder, f"{aligned}_xyI.csv")
    if cfg.run.skip_existing and os.path.exists(csv_path):
        return {"raw_path": raw, "xyI_path": csv_path, "status": "skipped"}
    last = None
    for _ in range(max(1, cfg.run.retries)):
        try:
            _align(raw, folder, stem, cfg)
            if not _propagate_sidecar(folder, stem, aligned):
                print(f"  WARN {stem}: no {stem}_frame.txt sidecar; a source='sidecar' "
                      "fit will have no calibration for this frame")
            _detect(aligned, folder, cfg)
            if cfg.run.chain_fit:
                _chain_fit(cfg.run.chain_fit, folder, aligned)
            return {"raw_path": raw, "xyI_path": csv_path, "status": "ok"}
        except Exception as exc:
            last = exc
    print(f"  SKIP {stem}: {type(last).__name__}: {last}")
    return {"raw_path": raw, "xyI_path": "", "status": f"error: {type(last).__name__}"}


def sweep(bc):
    if bc.manifest.root:
        rows, skipped = build_manifest(bc.manifest.root)
        if not rows:
            print(f"no raw stacks under {bc.manifest.root} (skipped {len(skipped)})")
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        outdir = os.path.abspath(bc.manifest.root)
        print(f"catalog: {len(df)} stack(s) under {bc.manifest.root}"
              + (f"; skipped {len(skipped)}" if skipped else ""))
    else:
        df = pd.read_csv(bc.manifest.path)
        outdir = os.path.dirname(os.path.abspath(bc.manifest.path))
    sel = _select(df, bc.filter)
    print(f"selected: {len(sel)}")
    recs = [_run_row(row, bc) for _, row in sel.iterrows()]
    out = pd.concat([sel.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    out_csv = os.path.join(outdir, "catalog_processed.csv")
    out.to_csv(out_csv, index=False)
    n_ok = sum(1 for r in recs if r["status"] == "ok")
    print(f"processed {n_ok}/{len(recs)} stack(s) -> {out_csv}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="experimental batch: align + detect over a catalog")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    sweep(load_batch_extras(args.config))


if __name__ == "__main__":
    main()
