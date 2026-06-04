#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Batch sweep: run the single-frame runner across a manifest, map results vs tilt.
#   python vmap_sweep.py --config batch.toml
# v1 is serial (run.workers / run.preprocess not yet wired); fields are harvested
# from the per-frame run return (abg / residual / std / sublattice shift).
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import math
import os
import tomllib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmap_config import AppConfig, load_batch
import vmap_run


def _select(df, flt):
    mask = pd.Series(True, index=df.index)
    for k, v in flt.items():
        if k not in df.columns:
            raise KeyError(f"filter column '{k}' not in manifest {list(df.columns)}")
        col = df[k]
        if isinstance(v, bool):
            mask &= col.astype(str).str.lower().isin(["true", "1", "yes"]) == v
        elif isinstance(v, (int, float)):
            mask &= np.isclose(pd.to_numeric(col, errors="coerce"), float(v))
        else:
            mask &= col.astype(str) == str(v)
    return df.loc[mask].copy()


def _run_row(row, fit_cfg_path, retries):
    folder = os.path.join(os.path.dirname(str(row["tiff_path"])), "")
    stem = os.path.basename(str(row["tiff_path"]))
    if stem.endswith(".tif"):
        stem = stem[:-4]
    with open(fit_cfg_path, "rb") as f:
        data = tomllib.load(f)
    data.setdefault("io", {})["folder"] = folder
    data["io"]["fname"] = stem
    cfg = AppConfig.model_validate(data)
    nominal = {m.label: tuple(m.coord) for m in cfg.motif}
    last = None
    for _ in range(max(1, retries)):
        try:
            lat, motif, _extra, meta = vmap_run.run(cfg, gui=False)
            return lat, motif, meta, nominal
        except Exception as exc:
            last = exc
    print(f"  SKIP {stem}: {type(last).__name__}: {last}")
    return None, None, None, nominal


def _fields(lat, motif, meta, nominal):
    out = {k: math.nan for k in
           ("a_fit", "b_fit", "g_fit", "residual_in_pm", "std", "atoms_used", "motif_dist")}
    if lat is None:
        return out
    abg = lat.get("abg") or []
    if len(abg) >= 3:
        out["a_fit"], out["b_fit"], out["g_fit"] = float(abg[0]), float(abg[1]), float(abg[2])
    if meta:
        if meta.get("residual_in_pm") is not None:
            out["residual_in_pm"] = float(meta["residual_in_pm"])
        if meta.get("atoms_used") is not None:
            out["atoms_used"] = float(meta["atoms_used"])
        std = meta.get("std")
        if std is not None:
            out["std"] = float(np.sqrt(np.sum(np.asarray(std, dtype=float) ** 2)))
    # motif_dist: real-space deviation of the fitted A-B vector from the nominal
    # (config) A-B vector, between the first two used sublattices.
    labels = [lbl for lbl, m in (motif or {}).items() if m.get("use", True)]
    if len(labels) >= 2 and not math.isnan(out["a_fit"]):
        a, b = labels[0], labels[1]
        (fax, fay), (fbx, fby) = motif[a]["coord"], motif[b]["coord"]
        (nax, nay), (nbx, nby) = nominal[a], nominal[b]
        ddx = (fax - fbx) - (nax - nbx)
        ddy = (fay - fby) - (nay - nby)
        g = out["g_fit"] / 180.0 * np.pi
        dx = ddx * out["a_fit"] + ddy * out["b_fit"] * np.cos(g)
        dy = ddy * out["b_fit"] * np.sin(g)
        out["motif_dist"] = float(np.sqrt(dx ** 2 + dy ** 2))
    return out


def _scatter(ax, x, y, title):
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("tilt a")
    ax.set_ylabel("tilt b")
    ax.set_title(title)


def _coverage(sel, outdir):
    x = pd.to_numeric(sel["tilt_a"], errors="coerce").to_numpy()
    y = pd.to_numeric(sel["tilt_b"], errors="coerce").to_numpy()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=50)
    _scatter(ax, x, y, "coverage")
    fig.savefig(os.path.join(outdir, "coverage.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_map(aug, spec, outdir):
    x = pd.to_numeric(aug["tilt_a"], errors="coerce").to_numpy()
    y = pd.to_numeric(aug["tilt_b"], errors="coerce").to_numpy()
    z = pd.to_numeric(aug[spec.field], errors="coerce").to_numpy() * spec.scale
    fig, ax = plt.subplots(figsize=(6, 6))
    if spec.significant and spec.significant in aug.columns:
        zs = pd.to_numeric(aug[spec.significant], errors="coerce").to_numpy() * spec.scale
        sig = np.abs(z) > zs
        ax.scatter(x[~sig], y[~sig], s=75, facecolors="none", edgecolors="k")
        sc = ax.scatter(x[sig], y[sig], c=z[sig], cmap="viridis", s=75)
    else:
        sc = ax.scatter(x, y, c=z, cmap="viridis", s=75)
    if sc.get_array() is not None and len(sc.get_array()):
        fig.colorbar(sc, ax=ax).set_label(spec.title or spec.field)
    _scatter(ax, x, y, spec.title or spec.field)
    fig.savefig(os.path.join(outdir, f"{spec.field}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def sweep(bc):
    df = pd.read_csv(bc.manifest.path)
    sel = _select(df, bc.filter)
    outdir = os.path.dirname(os.path.abspath(bc.manifest.path))
    print(f"Matches: {len(sel)}")
    if len(sel) == 0:
        for c in bc.filter:
            if c in df.columns:
                print(f"  {c}: {sorted(df[c].dropna().unique())[:12]}")
        return sel
    _coverage(sel, outdir)
    recs = []
    for _, row in sel.iterrows():
        lat, motif, meta, nominal = _run_row(row, bc.fit.config, bc.run.retries)
        recs.append(_fields(lat, motif, meta, nominal))
    aug = pd.concat([sel.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
    aug.to_csv(os.path.join(outdir, "lookup_augmented.csv"), index=False)
    for spec in bc.maps:
        if spec.field in aug.columns:
            _plot_map(aug, spec, outdir)
    print(f"wrote coverage.png, lookup_augmented.csv, {len(bc.maps)} map(s) -> {outdir}")
    return aug


def main(argv=None):
    ap = argparse.ArgumentParser(description="batch sweep over a manifest")
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    sweep(load_batch(args.config))


if __name__ == "__main__":
    main()
