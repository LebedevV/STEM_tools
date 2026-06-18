#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Catalog raw multi-frame STEM stacks (.emd/.dm3/.dm4) for the experimental batch.
# Mirrors vector_maps/vmap_manifest, but raw files carry no sg/hkl/tilt naming, so
# this is a catalog of stacks + shape (one row per stack). The pixel-size columns are
# best-effort / informational only -- experimental calibration is NOT taken from here;
# it comes from the per-.dm3 <name>_frame.txt sidecar at fit time (read_frame_calib).
#   python exp_batch_manifest.py <root> [-o catalog.csv]
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import argparse
import csv
import os

_EXTS = (".emd", ".dm3", ".dm4")
_COLS = ["raw_path", "source", "ext", "n_frames", "ny", "nx",
         "pixel_size_nm", "scan_size_nm", "title", "meta_ok"]


def _read_meta(path):
    # (n_frames, ny, nx, pixel_size_nm|None, title) or None on failure. Heavy deps are
    # imported lazily so the walk + tests run without pyTEMlib/hyperspy. pyTEMlib first
    # (calibrated pixel size for .emd), hyperspy fallback.
    # UNVERIFIED against real files: exercised end to end only on a real stack.
    try:
        import pyTEMlib.file_tools as ft
        ds = ft.open_file(path)
        if isinstance(ds, dict):
            ds = max(ds.values(), key=lambda d: getattr(d, "ndim", 0))
        shape = tuple(int(s) for s in ds.shape)
        nf = shape[0] if len(shape) >= 3 else 1
        px = float(getattr(getattr(ds, "x", None), "slope", 0.0)) or None
        return nf, shape[-2], shape[-1], px, str(getattr(ds, "title", "") or "")
    except Exception:
        pass
    try:
        import hyperspy.api as hs
        s = hs.load(path, lazy=True)
        shape = tuple(int(s) for s in s.data.shape)
        nf = shape[0] if len(shape) >= 3 else 1
        sc = s.axes_manager[-1].scale
        return nf, shape[-2], shape[-1], float(sc) if sc else None, str(getattr(s, "title", "") or "")
    except Exception:
        return None


def build_manifest(root, meta_reader=_read_meta):
    # one row per raw stack; source = the file's dir relative to root (groups sessions).
    # meta_reader is injectable so the walk is unit-testable without microscope libs.
    rows, skipped = [], []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith(_EXTS):
                continue
            path = os.path.join(dirpath, name)
            meta = meta_reader(path)
            if meta is None:
                skipped.append(path)
                continue
            nf, ny, nx, px, title = meta
            if nf < 2:                    # a single image cannot be registered -> skip
                skipped.append(path)
                continue
            rows.append({
                "raw_path": path,
                "source": os.path.relpath(dirpath, root),
                "ext": os.path.splitext(name)[1].lower(),
                "n_frames": nf, "ny": ny, "nx": nx,
                "pixel_size_nm": px if px else "",
                "scan_size_nm": px * nx if px else "",
                "title": title, "meta_ok": px is not None,
            })
    return rows, skipped


def write_manifest(rows, out):
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in _COLS})


def main(argv=None):
    ap = argparse.ArgumentParser(description="catalog raw STEM stacks for the experimental batch")
    ap.add_argument("root")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args(argv)
    rows, skipped = build_manifest(args.root)
    out = args.out or os.path.join(args.root, "catalog.csv")
    write_manifest(rows, out)
    print(f"{len(rows)} stack(s) -> {out}")
    if skipped:
        shown = ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else "")
        print(f"skipped {len(skipped)} file(s) (unreadable / not a stack): {shown}")


if __name__ == "__main__":
    main()
