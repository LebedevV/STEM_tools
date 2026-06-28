#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Config-driven refinement runner (see DESIGN.md). Run from the vector_maps dir:
#   python vmap_run.py --config fit.toml [--set k=v ...] [--no-gui] [--no-fit] [--calib X]
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import os
from pathlib import Path
import tomllib

import matplotlib.pyplot as plt
import pandas as pd

from routines.routines import *
from routines.refinement_routines import *
from routines.plot_routines import *
from routines.dicts_handling import unpack_to_dicts

from vmap_config import AppConfig

import inspect

# A pass forwards any non-schema key to refinement_run (see DESIGN.md). _RUN_SETS are
# the kwargs the runner owns, so they can't be set per-pass; the rest must name a real
# refinement_run parameter.
_REFINE_PARAMS = set(inspect.signature(refinement_run).parameters)
_RUN_SETS = {"folder", "sf", "fname", "calib", "lat_params", "motif", "extra_pars",
             "show_initial_spots", "vec_scale", "sub_area", "max_dist", "do_fit", "dataset_fname"}


# ---- config models -> the dicts refinement_run expects ---------------------

def _atom_entry(m):
    e = {"atom": m.el, "coord": tuple(m.coord), "I": m.intensity,
         "use": m.use, "fit": list(m.fit)}
    if m.eq is not None:
        e["eq"] = m.eq
    return e


def _to_dicts(cfg):
    lat_params = {"abg": list(cfg.lattice.abg), "fit_abg": list(cfg.lattice.fit_abg),
                  "base": list(cfg.lattice.base), "fit_base": list(cfg.lattice.fit_base)}
    motif = {m.label: _atom_entry(m) for m in cfg.motif}
    extra_pars = {n: (ep.value, ep.eq if ep.eq is not None else bool(ep.fit))
                  for n, ep in cfg.extra_pars.items()}
    return lat_params, motif, extra_pars


def _resolve_calib(cfg, folder, override):
    if override is not None:
        return float(override)
    if cfg.calibration.source == "value" and cfg.calibration.value is not None:
        return float(cfg.calibration.value)
    if cfg.calibration.source == "sidecar":
        return read_frame_calib(folder, cfg.io.fname, fallback=cfg.calibration.value)
    if cfg.calibration.source == "frame_size":
        if cfg.calibration.frame_size is not None:
            return calib_from_frame_size(folder, cfg.io.fname, cfg.calibration.frame_size)
        return read_toml_calib(folder, cfg.io.fname, cfg.calibration.toml_path)
    if cfg.calibration.source == "toml":
        return read_toml_calib(folder, cfg.io.fname, cfg.calibration.toml_path)
    if cfg.calibration.value is not None:
        return float(cfg.calibration.value)
    raise ValueError("no calibration: set [calibration].value, a sidecar/toml, or --calib")


# ---- seed sidecar ----------------------------------------------------------

def _seed_path(cfg, folder):
    return os.path.join(folder, cfg.run.seed_file.replace("{fname}", cfg.io.fname))


def _load_seed(path, lat_params, motif):
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        s = tomllib.load(f)
    if "abg" in s:
        lat_params["abg"] = list(s["abg"])
    if "base" in s:
        lat_params["base"] = list(s["base"])
    for m in s.get("added", []):
        motif[m["label"]] = {"atom": m["el"], "coord": tuple(m["coord"]),
                             "I": m.get("I", 1), "use": m.get("use", True), "fit": list(m["fit"])}


def _save_seed(path, lat_params):
    abg = [float(x) for x in lat_params["abg"]]
    base = [float(x) for x in lat_params["base"]]
    with open(path, "w") as f:
        f.write("# auto-written after a gui pass\n")
        f.write(f"abg  = {abg}\n")
        f.write(f"base = {base}\n")


# ---- pass execution --------------------------------------------------------

def _apply_use(mask, motif):
    for key, flag in mask.items():
        if key not in motif:
            raise KeyError(f"use mask references unknown motif '{key}'")
        motif[key]["use"] = bool(flag)


def _apply_fit(mask, lat_params, motif, extra_pars):
    for key, flags in mask.items():
        if key == "abg":
            lat_params["fit_abg"] = list(flags)
        elif key == "base":
            lat_params["fit_base"] = list(flags)
        elif key in motif:
            motif[key]["fit"] = list(flags)
        elif key in extra_pars:
            val, spec = extra_pars[key]
            if isinstance(spec, str):
                raise KeyError(f"fit mask cannot toggle eq-coupled extra_par '{key}'")
            extra_pars[key] = (val, bool(flags[0]))
        else:
            raise KeyError(f"fit mask references unknown param '{key}'")


def _expand_areas(exp):
    frm, to, step = list(exp.start), list(exp.to), exp.step
    nstep = int(round(max(abs(to[i] - frm[i]) for i in range(len(frm))) / step)) if step else 0
    areas = []
    for k in range(nstep + 1):
        area = []
        for i in range(len(frm)):
            if to[i] == frm[i]:
                area.append(frm[i])
            else:
                d = step if to[i] > frm[i] else -step
                v = frm[i] + d * k
                area.append(min(v, to[i]) if d > 0 else max(v, to[i]))
        areas.append(area)
    return areas


def _passthrough(p):
    # non-schema pass keys -> refinement_run kwargs (recall_zero, export_sublattice_xy,
    # kernel, relative_to, ...). Reject runner-owned kwargs and non-parameters.
    out = {}
    for k, v in (p.__pydantic_extra__ or {}).items():
        if k in _RUN_SETS:
            raise KeyError(f"pass '{p.name}': '{k}' is set by the runner, not per-pass")
        if k not in _REFINE_PARAMS:
            raise KeyError(f"pass '{p.name}': '{k}' is not a refinement_run kwarg")
        out[k] = v
    return out


def _run_pass(p, folder, fname, calib, lat_params, motif, extra_pars,
              gui_master, refine_master, sub_area=None, dataset_fname=None,
              save_stem="{fname}_{name}"):
    if p.fit:
        _apply_fit(p.fit, lat_params, motif, extra_pars)
    for m in p.add:
        motif[m.label] = _atom_entry(m)
    sa = sub_area if sub_area is not None else p.sub_area
    # saved-pass folder carries the source frame name (configurable via run.save_stem),
    # so different frames don't collide in a bare "<pass>/" dir.
    sf = save_stem.format(fname=fname, name=p.name) if p.save else None
    meta, vec = refinement_run(
        folder, sf, fname, calib, lat_params, motif, extra_pars=extra_pars,
        show_initial_spots=(gui_master and p.gui), vec_scale=p.vec_scale,
        sub_area=sa, max_dist=p.max_dist, do_fit=(p.refine and refine_master),
        dataset_fname=dataset_fname, **_passthrough(p),
    )
    unpack_to_dicts(vec, lat_params, motif, extra_pars)
    return meta


def _merge_body(parent, body):
    # body sub-pass inherits the parent expand pass; overrides only what it sets
    overrides = {f: getattr(body, f) for f in body.model_fields_set if f not in ("expand", "body")}
    overrides.update(body.__pydantic_extra__ or {})
    return parent.model_copy(update={**overrides, "expand": None, "body": []})


def _rotate_backup(path, keep=3):
    # Bump an existing measurement out of the way before a reset overwrites it:
    # path -> .bckp1, .bckp1 -> .bckp2, ...; .bckp{keep} drops off ("gone").
    if not os.path.exists(path):
        return
    oldest = f"{path}.bckp{keep}"
    if os.path.exists(oldest):
        os.remove(oldest)
    for i in range(keep - 1, 0, -1):
        src = f"{path}.bckp{i}"
        if os.path.exists(src):
            os.replace(src, f"{path}.bckp{i + 1}")
    os.replace(path, f"{path}.bckp1")


def _read_xyi_csv(path, *, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} did not write {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise RuntimeError(f"{label} wrote an empty CSV: {path}") from exc
    if len(df) == 0:
        raise RuntimeError(f"{label} found zero atom positions: {path}")
    return df


def _xyi_csv_path(folder, stem):
    return os.path.join(folder, f"{stem}_xyI.csv")

def _run_detect(d, folder, fname, current, name, lat_params, motif, extra_pars, calib):
    # One detection step, composed at the schedule level (see DESIGN.md). The two
    # modes are never crossed:
    #   reset (default): a fresh detection REPLACES <fname>_xyI.csv; the prior one
    #     rotates to .bckp1/2/3 (seed="fit" seeds it from the current lattice instead of
    #     finding peaks from scratch). Returns None -> the next fit reads the canonical csv.
    #   accrete: detect (typically on d.source, a prior step's _diff2.tif residual),
    #     then concat that detection onto the current working set -- `current` (None ->
    #     canonical) -- with NO dedup (fitted positions must not be merged), into
    #     <save_as>_xyI.csv. The working set is left untouched, so a reset measurement
    #     is never folded back into an accreted set. Mirrors fit_lattice_PZT's A+B concat.
    from routines.detect_columns import detect_columns
    src = d.source.format(fname=fname, name=name) if d.source else None

    def detect(out_suffix):
        detect_columns(
            fname=fname + ".tif", folder=folder, imsize=tuple(d.imsize),
            sep=d.sep, sigma1=d.sigma1, thr=d.thr, ptonn=d.ptonn,
            pca=d.pca, subtract_background=d.subtract_background,
            interactive=False, source_fname=src, out_suffix=out_suffix,
        )

    if not d.accrete:
        _rotate_backup(_xyi_csv_path(folder, fname))
        if d.seed == "fit":
            # only the peak-finding is skipped -- detect_columns still 2D-gaussian-refines the seeds
            redetect_from_lattice(folder, fname, calib, lat_params, motif, extra_pars,
                                  ptonn=d.ptonn, out_suffix="")
        else:
            detect(out_suffix="")
        out_path = _xyi_csv_path(folder, fname)
        out_df = _read_xyi_csv(out_path, label=f"detect pass '{name}'")
        print(f"[{name}] detect count: {len(out_df)} -> {out_path}")
        return None

    detect(out_suffix=f"_{name}")
    base = current if current is not None else fname
    target = d.save_as.format(fname=fname, name=name)
    base_path = _xyi_csv_path(folder, base)
    new_path = _xyi_csv_path(folder, f"{fname}_{name}")
    target_path = _xyi_csv_path(folder, target)

    base_df = _read_xyi_csv(base_path, label=f"detect pass '{name}' base dataset")
    new_df = _read_xyi_csv(new_path, label=f"detect pass '{name}' accreted detection")
    merged = pd.concat([base_df, new_df], ignore_index=True, sort=False)
    merged.to_csv(target_path, index=False, float_format="%.8g")
    print(f"[{name}] detect count: base={len(base_df)}, new={len(new_df)}, "
          f"merged={len(merged)} -> {target_path}")
    return target


def run(cfg: AppConfig, *, gui=None, refine=None, calib=None):
    folder = os.path.join(cfg.io.folder, "")
    fname = cfg.io.fname
    cal = _resolve_calib(cfg, folder, calib)
    lat_params, motif, extra_pars = _to_dicts(cfg)

    gui_master = cfg.run.gui if gui is None else gui
    refine_master = True if refine is None else refine

    seed_path = _seed_path(cfg, folder) if cfg.run.seed else None
    if seed_path:
        _load_seed(seed_path, lat_params, motif)

    meta = None
    gui_opened = False
    dataset = None                      # the working point set the next fit reads (None -> canonical
                                        # <fname>_xyI.csv); a reset detect resets it to None, an accrete
                                        # detect concats onto it and sets its save_as stem
    for p in cfg.run.passes:
        if p.detect is not None:
            if p.use:
                _apply_use(p.use, motif)
            dataset = _run_detect(p.detect, folder, fname, dataset, p.name,
                                  lat_params, motif, extra_pars, cal)
            print(f"[{p.name}] detect -> dataset = {dataset}")
            continue
        if p.expand is not None:
            bodies = p.body if p.body else [None]
            areas = _expand_areas(p.expand)
            for ai, area in enumerate(areas):
                for b in bodies:
                    eff = p if b is None else _merge_body(p, b)
                    if eff.use:
                        _apply_use(eff.use, motif)
                    gui_opened = gui_opened or (gui_master and eff.gui)
                    meta = _run_pass(eff, folder, fname, cal, lat_params, motif,
                                     extra_pars, gui_master, refine_master,
                                     sub_area=area, dataset_fname=dataset,
                                     save_stem=cfg.run.save_stem)
                    print(f"[{p.name} {ai + 1}/{len(areas)}] residual_in_pm = "
                          f"{meta.get('residual_in_pm') if meta else None}")
        else:
            if p.use:
                _apply_use(p.use, motif)
            gui_opened = gui_opened or (gui_master and p.gui)
            meta = _run_pass(p, folder, fname, cal, lat_params, motif,
                             extra_pars, gui_master, refine_master, dataset_fname=dataset,
                             save_stem=cfg.run.save_stem)
            print(f"[{p.name}] residual_in_pm = {meta.get('residual_in_pm') if meta else None}")

    if seed_path and gui_opened:
        _save_seed(seed_path, lat_params)

    if cfg.run.unit_cell:
        # fold the frame onto the refined cell -> <fname>_uc_{mean,std,count}.tif
        # + <fname>_uc_figure.png (cell schematic | mean | std), as the legacy
        # --unit-cell driver did (motif drives the schematic).
        from routines.unit_cell_average import unit_cell_average_to_tiffs
        unit_cell_average_to_tiffs(os.path.join(folder, fname), lat_params, cal, motif)

    out = lat_params, motif, extra_pars, meta
    plt.close('all')
    return out


# ---- CLI -------------------------------------------------------------------

def _parse_scalar(val):
    try:
        return tomllib.loads(f"x = {val}")["x"]
    except tomllib.TOMLDecodeError:
        return val


def _apply_overrides(data, sets):
    for s in sets or []:
        key, _, val = s.partition("=")
        node = data
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = _parse_scalar(val)
    return data


def _folder_slash(path):
    return os.path.join(str(path), "")


def _apply_cli_paths(data, args):
    if args.frame is not None:
        frame = Path(args.frame)
        data.setdefault("io", {})["folder"] = _folder_slash(frame.parent)
        data["io"]["fname"] = frame_stem(frame.name)
    if args.folder is not None:
        data.setdefault("io", {})["folder"] = _folder_slash(args.folder)
    if args.fname is not None:
        data.setdefault("io", {})["fname"] = frame_stem(args.fname)
    if args.toml_path is not None:
        data.setdefault("calibration", {})["toml_path"] = str(args.toml_path)
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="config-driven vector_maps refinement runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--frame", default=None, help="frame TIFF path; sets io.folder and io.fname")
    ap.add_argument("--folder", default=None, help="override io.folder")
    ap.add_argument("--fname", default=None, help="override io.fname; TIFF suffix is stripped safely")
    ap.add_argument("--toml-path", default=None, help="override calibration.toml_path")
    ap.add_argument("--set", action="append", default=[], metavar="k=v")
    ap.add_argument("--calib", type=float, default=None)
    ap.add_argument("--no-gui", dest="gui", action="store_false", default=None)
    ap.add_argument("--no-fit", dest="refine", action="store_false", default=None)
    args = ap.parse_args(argv)

    with open(args.config, "rb") as f:
        data = tomllib.load(f)
    if "manifest" in data or "maps" in data:
        raise SystemExit(f"{args.config} is a batch sweep config; "
                         f"run: python vmap_sweep.py --config {args.config}")
    _apply_overrides(data, args.set)
    _apply_cli_paths(data, args)
    cfg = AppConfig.model_validate(data)

    run(cfg, gui=args.gui, refine=args.refine, calib=args.calib)


if __name__ == "__main__":
    main()
