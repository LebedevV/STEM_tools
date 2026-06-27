#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Build a manifest.csv from an abtem out_full directory for the batch sweep.
#   python vmap_manifest.py <out_full_dir> [-o manifest.csv]
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import csv
import os
import re
import tomllib

# e.g. Pm3m_(25.0, 10.0)_110_haadf_0-25.tif  |  fph_Pm3m_(25.0, 5.0)_110_abf.tif
_TIFF = re.compile(
    r"^(?P<fph>fph_)?(?P<sg>[^_]+)_\((?P<ta>[-\d.]+),\s*(?P<tb>[-\d.]+)\)_"
    r"(?P<hkl>[^_]+)_(?P<det>haadf|abf|bf)(?P<blur>(?:_[\d-]+)?)\.tiff?$",
    re.IGNORECASE,
)

_COLS = ["tiff_path", "source", "toml_path", "sg", "hkl", "tilt_a", "tilt_b", "detector",
         "scan_s", "thickness", "borders", "phonons", "fph_sigma", "blur_sigma",
         "is_fph", "matched_naming"]


def _blur(s):
    return float(s.lstrip("_").replace("-", ".")) if s else 0.0


def _toml_meta(path):
    # harvest every column-friendly value from the physics sections (+ sample_name) into the
    # manifest, so downstream analysis never reopens a toml; list/dict fields (detectors,
    # aberrations, blur_sigmas) don't fit a column and are skipped. frozen_phonons -> phonons
    # (the name the filter/maps key on).
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        t = tomllib.load(f)
    meta = {}
    for sect in ("microscope", "lamella_settings", "simulations"):
        meta.update({k: v for k, v in t.get(sect, {}).items()
                     if isinstance(v, (int, float, str, bool))})
    sample = t.get("paths", {}).get("sample_name")
    if sample is not None:
        meta["sample_name"] = sample
    if "frozen_phonons" in meta:
        meta["phonons"] = meta.pop("frozen_phonons")
    return meta


def build_manifest(folder):
    # walk the whole tree so one root covering several run folders yields one manifest;
    # source = the frame's dir relative to the root (groups the otherwise-identical frames).
    # Store file paths as absolute paths so the manifest remains usable from any cwd.
    folder = os.path.abspath(folder)
    rows, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith((".tif", ".tiff")):
                continue
            m = _TIFF.match(name)
            if not m:
                skipped.append(os.path.join(dirpath, name))
                continue
            toml_path = os.path.join(dirpath, f"{m['sg']}_{m['hkl']}_({m['ta']}, {m['tb']}).toml")
            rows.append({
                **_toml_meta(toml_path),                 # harvest first; structural keys below win on any name clash
                "tiff_path": os.path.join(dirpath, name),
                "source": os.path.relpath(dirpath, folder),
                "toml_path": toml_path if os.path.exists(toml_path) else "",
                "sg": m["sg"], "hkl": m["hkl"],
                "tilt_a": float(m["ta"]), "tilt_b": float(m["tb"]),
                "detector": m["det"].lower(), "blur_sigma": _blur(m["blur"]),
                "is_fph": bool(m["fph"]), "matched_naming": True,
            })
    return rows, skipped


def write_manifest(rows, out):
    # _COLS first (familiar order), then any harvested params present, sorted
    extra = sorted({k for r in rows for k in r if k not in _COLS})
    fieldnames = _COLS + extra
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in fieldnames})


def main(argv=None):
    ap = argparse.ArgumentParser(description="build a batch manifest from an abtem out_full dir")
    ap.add_argument("folder")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    rows, skipped = build_manifest(args.folder)
    out = args.out or os.path.join(args.folder, "manifest.csv")
    write_manifest(rows, out)
    print(f"{len(rows)} rows -> {out}")
    if skipped:
        shown = ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else "")
        print(f"skipped {len(skipped)} tiff(s) with non-frame naming: {shown}")


if __name__ == "__main__":
    main()
