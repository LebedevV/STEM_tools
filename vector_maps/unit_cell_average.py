#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Fold a periodic lattice image onto one unit cell: per-bin mean / std / count.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np


def _setup(a_px, b_px, origin_px, shape):
	ax, ay = float(a_px[0]), float(a_px[1])
	bx, by = float(b_px[0]), float(b_px[1])
	x0, y0 = float(origin_px[0]), float(origin_px[1])
	det = ax * by - ay * bx
	if abs(det) < 1e-9:
		raise ValueError("degenerate lattice: a_px and b_px are colinear")
	if shape is None:
		N = max(1, int(round(np.hypot(ax, ay))))
		M = max(1, int(round(np.hypot(bx, by))))
	else:
		N, M = int(shape[0]), int(shape[1])
	return ax, ay, bx, by, x0, y0, det, N, M


def _full_mask(i, j, geom, W, H):
	# cell (i,j) wholly inside the image iff its 4 corners are in-bounds
	# (the cell is their convex hull). Works for 1-D or 2-D i,j.
	ax, ay, bx, by, x0, y0 = geom
	ok = np.ones(np.shape(i), dtype=bool)
	for di in (0.0, 1.0):
		for dj in (0.0, 1.0):
			cx = x0 + (i + di) * ax + (j + dj) * bx
			cy = y0 + (i + di) * ay + (j + dj) * by
			ok &= (cx >= 0) & (cx <= W - 1) & (cy >= 0) & (cy <= H - 1)
	return ok


def _bilinear(img, x, y):
	# sample img at (x, y); NaN outside [0, W-1] x [0, H-1] (or near a NaN pixel)
	H, W = img.shape
	xi = np.floor(x).astype(np.int64)
	yi = np.floor(y).astype(np.int64)
	ok = (xi >= 0) & (xi < W - 1) & (yi >= 0) & (yi < H - 1)
	xc = np.clip(xi, 0, W - 2)
	yc = np.clip(yi, 0, H - 2)
	fx, fy = x - xc, y - yc
	v = ((1 - fx) * (1 - fy) * img[yc, xc] + fx * (1 - fy) * img[yc, xc + 1]
	     + (1 - fx) * fy * img[yc + 1, xc] + fx * fy * img[yc + 1, xc + 1])
	return np.where(ok, v, np.nan)


def _finalize(s1, s2, cnt):
	with np.errstate(invalid="ignore", divide="ignore"):
		safe = np.where(cnt > 0, cnt, 1)
		mean = np.where(cnt > 0, s1 / safe, np.nan)
		ssd = s2 - s1 * s1 / safe                       # sum of squared deviations
		var = np.where(cnt > 1, ssd / np.where(cnt > 1, cnt - 1, 1), np.nan)
	std = np.sqrt(np.clip(var, 0.0, None))              # sample std (ddof=1)
	return mean, std, cnt.astype(np.float64)


def _accum_raw(img, geom, det, N, M, full_cells_only):
	# scatter: each pixel counted once in its (u,v) bin — raw measured values,
	# no interpolation. Bin counts are uneven (orthogonal grid vs oblique cell).
	ax, ay, bx, by, x0, y0 = geom
	H, W = img.shape
	ys, xs = np.indices((H, W), dtype=np.float64)
	u = (by * (xs - x0) - bx * (ys - y0)) / det
	v = (-ay * (xs - x0) + ax * (ys - y0)) / det
	i, j = np.floor(u), np.floor(v)
	keep = np.isfinite(img)
	if full_cells_only:
		keep &= _full_mask(i, j, geom, W, H)
	bi = np.clip(((u - i) * N).astype(np.int64), 0, N - 1)
	bj = np.clip(((v - j) * M).astype(np.int64), 0, M - 1)
	flat = (bi * M + bj)[keep]
	vals = img[keep]
	cnt = np.bincount(flat, minlength=N * M).astype(np.float64)
	s1 = np.bincount(flat, weights=vals, minlength=N * M)
	s2 = np.bincount(flat, weights=vals * vals, minlength=N * M)
	return s1.reshape(N, M), s2.reshape(N, M), cnt.reshape(N, M)


def _accum_resample(img, geom, det, N, M, full_cells_only):
	# gather: read every full cell at the SAME regular (u,v) grid (bilinear) and
	# stack. Uniform count (= number of cells), std is pure cell-to-cell variation.
	ax, ay, bx, by, x0, y0 = geom
	H, W = img.shape
	cx = np.array([0, W - 1, 0, W - 1], dtype=np.float64)
	cy = np.array([0, 0, H - 1, H - 1], dtype=np.float64)
	cu = (by * (cx - x0) - bx * (cy - y0)) / det
	cv = (-ay * (cx - x0) + ax * (cy - y0)) / det
	ii, jj = np.meshgrid(np.arange(int(np.floor(cu.min())), int(np.ceil(cu.max())) + 1),
			     np.arange(int(np.floor(cv.min())), int(np.ceil(cv.max())) + 1),
			     indexing="ij")
	ii, jj = ii.ravel().astype(np.float64), jj.ravel().astype(np.float64)
	if full_cells_only:
		m = _full_mask(ii, jj, geom, W, H)
		ii, jj = ii[m], jj[m]

	uu, vv = np.meshgrid((np.arange(N) + 0.5) / N, (np.arange(M) + 0.5) / M, indexing="ij")
	s1 = np.zeros((N, M))
	s2 = np.zeros((N, M))
	cnt = np.zeros((N, M))
	for ci, cj in zip(ii, jj):                          # O(N*M) memory, loop over cells
		x = x0 + (ci + uu) * ax + (cj + vv) * bx
		y = y0 + (ci + uu) * ay + (cj + vv) * by
		samp = _bilinear(img, x, y)
		fin = np.isfinite(samp)
		s1[fin] += samp[fin]
		s2[fin] += samp[fin] ** 2
		cnt[fin] += 1
	return s1, s2, cnt


def average_unit_cell(image, a_px, b_px, origin_px, shape=None,
		      full_cells_only=True, method="resample"):
	"""Average every unit cell of a periodic lattice onto one.

	image            2D array (H, W) — STEM image or an atomap residual tiff.
	a_px, b_px       lattice basis vectors in pixels, (dx, dy) each (rotation /
	                 shear included — e.g. from the fit's abg/base via calib).
	origin_px        (x0, y0), lattice origin in pixels.
	shape            (N, M) output grid over fractional (u along a, v along b);
	                 defaults to (round|a|, round|b|).
	full_cells_only  use only cells wholly inside the image — drops the
	                 poorly-fit border cells (default True).
	method           "resample" (default): read each full cell at the same regular
	                 (u,v) grid (bilinear) and average across cells -> uniform
	                 count, std = pure cell-to-cell variation. "raw": scatter each
	                 pixel into its bin (no interpolation) -> raw values, uneven count.

	Returns (mean, std, count), each (N, M), indexed [u_bin, v_bin]. Empty bins
	are NaN; std (sample, ddof=1) is NaN where count < 2. The cell is a torus, so
	there is no seam to special-case.
	"""
	img = np.asarray(image, dtype=np.float64)
	ax, ay, bx, by, x0, y0, det, N, M = _setup(a_px, b_px, origin_px, shape)
	geom = (ax, ay, bx, by, x0, y0)
	if method == "resample":
		s1, s2, cnt = _accum_resample(img, geom, det, N, M, full_cells_only)
	elif method == "raw":
		s1, s2, cnt = _accum_raw(img, geom, det, N, M, full_cells_only)
	else:
		raise ValueError(f"method must be 'resample' or 'raw', got {method!r}")
	return _finalize(s1, s2, cnt)


def lattice_px_from_fit(lat_params, calib):
	"""Pixel-space lattice vectors + origin from a fit's lat_params.

	Mirrors get_coords_from_ij: abg = a, b (nm), gamma (deg) build the cell;
	base = shx, shy (nm), phi (deg) set origin + frame rotation. Model nm =
	pixel * calib, so px = nm / calib. Returns (a_px, b_px, origin_px), (dx, dy) each.
	"""
	a, b, gamma = lat_params["abg"]
	shx, shy, phi = lat_params["base"]
	g, p = np.radians(gamma), np.radians(phi)
	ax, ay = a * np.cos(p), -a * np.sin(p)
	bx0, by0 = b * np.cos(g), b * np.sin(g)
	bx = bx0 * np.cos(p) + by0 * np.sin(p)
	by = by0 * np.cos(p) - bx0 * np.sin(p)
	return ((ax / calib, ay / calib), (bx / calib, by / calib), (shx / calib, shy / calib))


def average_unit_cell_from_fit(image, lat_params, calib, **kwargs):
	"""average_unit_cell with lattice vectors derived from a fit's lat_params.

	Run after a do_fit pass on the refined params, or on any provided params.
	kwargs pass through (shape, full_cells_only, method).
	"""
	a_px, b_px, origin_px = lattice_px_from_fit(lat_params, calib)
	return average_unit_cell(image, a_px, b_px, origin_px, **kwargs)


def _strip_tif(path):
	for ext in (".tiff", ".tif"):
		if path.lower().endswith(ext):
			return path[:-len(ext)]
	return path


def unit_cell_average_to_tiffs(image_path, lat_params, calib, out_stem=None, **kwargs):
	"""Read image_path, fold via a fit's lattice, write <stem>_uc_{mean,std,count}.tif.

	Convenience for a driver's post-fit step: pass the refined lat_params + calib.
	"""
	import cv2
	img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise FileNotFoundError(image_path)
	mean, std, count = average_unit_cell_from_fit(img, lat_params, calib, **kwargs)
	stem = out_stem if out_stem is not None else _strip_tif(image_path)
	for nm, arr in (("mean", mean), ("std", std), ("count", count)):
		cv2.imwrite(f"{stem}_uc_{nm}.tif", np.asarray(arr, dtype=np.float32))
	return mean, std, count


# ---- CLI: eat a tiff (image or atomap residual) -> mean / std / count tiffs ----

def _xy(s):
	parts = [float(t) for t in s.split(",")]
	if len(parts) != 2:
		raise ValueError(f"expected 'x,y', got {s!r}")
	return parts


def _read_fit_lattice(path):
	import pandas as pd
	df = pd.read_csv(path, sep=None, engine="python", index_col=0)
	cols = list(df.columns)[:3]
	return {"abg": [float(df.loc["abg", c]) for c in cols],
		"base": [float(df.loc["base", c]) for c in cols]}


def main(argv=None):
	import argparse
	import cv2
	p = argparse.ArgumentParser(description="average a periodic lattice tiff onto one unit cell")
	p.add_argument("tiff")
	p.add_argument("--a", type=_xy, help="a vector 'dx,dy' in px")
	p.add_argument("--b", type=_xy, help="b vector 'dx,dy' in px")
	p.add_argument("--origin", type=_xy, help="lattice origin 'x0,y0' in px")
	p.add_argument("--from-fit", dest="from_fit", help="fit lattice.csv (abg/base rows)")
	p.add_argument("--calib", type=float, help="nm/pixel, with --from-fit")
	p.add_argument("--shape", type=lambda s: [int(t) for t in s.split(",")], help="N,M output grid")
	p.add_argument("--method", choices=("resample", "raw"), default="resample")
	p.add_argument("--no-full-cells", dest="full_cells", action="store_false")
	p.add_argument("-o", "--out", help="output prefix (default: input path minus .tif)")
	args = p.parse_args(argv)

	img = cv2.imread(args.tiff, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise SystemExit(f"could not read {args.tiff}")

	if args.from_fit:
		if args.calib is None:
			raise SystemExit("--from-fit needs --calib")
		a_px, b_px, origin_px = lattice_px_from_fit(_read_fit_lattice(args.from_fit), args.calib)
	elif args.a and args.b and args.origin:
		a_px, b_px, origin_px = args.a, args.b, args.origin
	else:
		raise SystemExit("give --from-fit LATTICE.csv --calib X, or --a --b --origin (px)")

	mean, std, count = average_unit_cell(img, a_px, b_px, origin_px, shape=args.shape,
					     full_cells_only=args.full_cells, method=args.method)

	stem = args.out if args.out is not None else _strip_tif(args.tiff)
	for name, arr in (("mean", mean), ("std", std), ("count", count)):
		out = f"{stem}_uc_{name}.tif"
		cv2.imwrite(out, arr.astype(np.float32))
		print(f"  wrote {out}")
	print(f"cell {mean.shape}, cells averaged ~{int(np.nanmax(count))}, method={args.method}")


if __name__ == "__main__":
	main()
