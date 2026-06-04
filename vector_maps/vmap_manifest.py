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
    r"(?P<hkl>[^_]+)_(?P<det>haadf|abf|bf)(?P<blur>(?:_[\d-]+)?)\.tif$"
)

_COLS = ["tiff_path", "toml_path", "sg", "hkl", "tilt_a", "tilt_b", "detector",
         "scan_s", "thickness", "borders", "phonons", "fph_sigma", "blur_sigma",
         "is_fph", "matched_naming"]


def _blur(s):
    return float(s.lstrip("_").replace("-", ".")) if s else 0.0


def _toml_meta(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        t = tomllib.load(f)
    lam, sim = t.get("lamella_settings", {}), t.get("simulations", {})
    return {"scan_s": lam.get("scan_s"), "thickness": lam.get("thickness"),
            "borders": lam.get("borders"), "phonons": sim.get("frozen_phonons"),
            "fph_sigma": sim.get("fph_sigma")}


def build_manifest(folder):
    rows, skipped = [], []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        m = _TIFF.match(name)
        if not m:
            skipped.append(name)
            continue
        toml_path = os.path.join(folder, f"{m['sg']}_{m['hkl']}_({m['ta']}, {m['tb']}).toml")
        rows.append({
            "tiff_path": os.path.join(folder, name),
            "toml_path": toml_path if os.path.exists(toml_path) else "",
            "sg": m["sg"], "hkl": m["hkl"],
            "tilt_a": float(m["ta"]), "tilt_b": float(m["tb"]),
            "detector": m["det"], "blur_sigma": _blur(m["blur"]),
            "is_fph": bool(m["fph"]), "matched_naming": True,
            **_toml_meta(toml_path),
        })
    return rows, skipped


def write_manifest(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})


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
