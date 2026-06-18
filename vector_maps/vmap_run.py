#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Config-driven refinement runner (see DESIGN.md). Run from the vector_maps dir:
#   python vmap_run.py --config fit.toml [--set k=v ...] [--no-gui] [--no-fit] [--calib X]
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import os
import tomllib

from routines import *
from refinement_routines import *
from plot_routines import *
from dicts_handling import unpack_to_dicts

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
    if cfg.calibration.value is not None:
        return float(cfg.calibration.value)
    raise ValueError("no calibration: set [calibration].value, a sidecar, or --calib")


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


def _run_detect(d, folder, fname):
    # PZT-style re-detection: one chained detect pass per ptonn entry (each on the
    # prior residual; ptonn = percent_to_nn fit window), merged into one
    # <fname>_sub_AB dataset. Mirrors index_all3 / fit_lattice_PZT.
    # UNVERIFIED: needs detect_columns on the path + image data to exercise.
    from detect_columns import detect_columns
    parts, source = [], None
    for i, pt in enumerate(d.ptonn):
        suffix = f"_sub{i}_rerun"
        detect_columns(
            fname=fname + ".tif", folder=folder,
            imsize=tuple(d.imsize),
            sep=d.sep, sigma1=d.sigma1, thr=d.thr, ptonn=pt,
            pca=d.pca, subtract_background=d.subtract_background,
            interactive=False, source_fname=source, out_suffix=suffix,
        )
        parts.append(os.path.join(folder, f"{fname}{suffix}_xyI.csv"))
        source = f"{fname}{suffix}_2DG_ptnn_{pt}_diff2.tif"
    if not d.merge:
        return None
    frames = []
    for j, path in enumerate(parts):
        df = pd.read_csv(path)
        df["sub_id"] = chr(ord("A") + j)
        frames.append(df)
    merged = f"{fname}_sub_AB"
    pd.concat(frames, ignore_index=True, sort=False).to_csv(
        os.path.join(folder, f"{merged}_xyI.csv"), index=False, float_format="%.8g")
    return merged


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
    dataset = None                      # None -> use <fname>; set after a detect+merge
    for p in cfg.run.passes:
        if p.detect is not None:
            dataset = _run_detect(p.detect, folder, fname)
            print(f"[{p.name}] detect -> dataset = {dataset}")
            continue
        if p.expand is not None:
            bodies = p.body if p.body else [None]
            areas = _expand_areas(p.expand)
            for ai, area in enumerate(areas):
                for b in bodies:
                    eff = p if b is None else _merge_body(p, b)
                    gui_opened = gui_opened or (gui_master and eff.gui)
                    meta = _run_pass(eff, folder, fname, cal, lat_params, motif,
                                     extra_pars, gui_master, refine_master,
                                     sub_area=area, dataset_fname=dataset,
                                     save_stem=cfg.run.save_stem)
                    print(f"[{p.name} {ai + 1}/{len(areas)}] residual_in_pm = "
                          f"{meta.get('residual_in_pm') if meta else None}")
        else:
            gui_opened = gui_opened or (gui_master and p.gui)
            meta = _run_pass(p, folder, fname, cal, lat_params, motif,
                             extra_pars, gui_master, refine_master, dataset_fname=dataset,
                             save_stem=cfg.run.save_stem)
            print(f"[{p.name}] residual_in_pm = {meta.get('residual_in_pm') if meta else None}")

    if seed_path and gui_opened:
        _save_seed(seed_path, lat_params)

    return lat_params, motif, extra_pars, meta


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


def main(argv=None):
    ap = argparse.ArgumentParser(description="config-driven vector_maps refinement runner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="k=v")
    ap.add_argument("--calib", type=float, default=None)
    ap.add_argument("--no-gui", dest="gui", action="store_false", default=None)
    ap.add_argument("--no-fit", dest="refine", action="store_false", default=None)
    args = ap.parse_args(argv)

    with open(args.config, "rb") as f:
        data = tomllib.load(f)
    _apply_overrides(data, args.set)
    cfg = AppConfig.model_validate(data)

    run(cfg, gui=args.gui, refine=args.refine, calib=args.calib)


if __name__ == "__main__":
    main()
