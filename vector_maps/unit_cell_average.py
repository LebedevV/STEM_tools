#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Fold a periodic lattice image onto one unit cell: per-cell mean / std / count.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np


def _geom(a_px, b_px, origin_px):
	ax, ay = float(a_px[0]), float(a_px[1])
	bx, by = float(b_px[0]), float(b_px[1])
	x0, y0 = float(origin_px[0]), float(origin_px[1])
	det = ax * by - ay * bx
	if abs(det) < 1e-9:
		raise ValueError("degenerate lattice: a_px and b_px are colinear")
	return ax, ay, bx, by, x0, y0, det


def _native_shape(ax, ay, bx, by):
	# the cell's own pixel extent along a and b
	return max(1, int(round(np.hypot(ax, ay)))), max(1, int(round(np.hypot(bx, by))))


def _box(W, H, sub_area):
	# region of interest in px, clipped to the image; sub_area = [x0, x1, y0, y1]
	if sub_area is None:
		return 0.0, W - 1.0, 0.0, H - 1.0
	x0, x1, y0, y1 = (float(v) for v in sub_area)
	return max(x0, 0.0), min(x1, W - 1.0), max(y0, 0.0), min(y1, H - 1.0)


def _full_mask(i, j, geom, box):
	# cell (i,j) wholly inside the box iff its 4 corners are (convex hull). 1-D/2-D.
	ax, ay, bx, by, x0, y0 = geom
	xlo, xhi, ylo, yhi = box
	ok = np.ones(np.shape(i), dtype=bool)
	for di in (0.0, 1.0):
		for dj in (0.0, 1.0):
			cx = x0 + (i + di) * ax + (j + dj) * bx
			cy = y0 + (i + di) * ay + (j + dj) * by
			ok &= (cx >= xlo) & (cx <= xhi) & (cy >= ylo) & (cy <= yhi)
	return ok


def _bilinear(img, x, y):
	# sample img at (x, y); NaN outside the grid (or near a NaN pixel)
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


def _accum_raw(img, geom, det, full_cells_only, box):
	# Fold actual pixels onto one cell's native pixel footprint: each pixel -> its
	# cell (i,j) -> its integer px offset within that cell, placed once (no tiling,
	# no interpolation). The cell is shown once in its parallelogram bbox; rounding
	# the oblique edge makes a staircase (pixels protrude past it on one side, gaps
	# on the other), and bins no pixel centre lands in -- gaps + the bbox corners
	# outside the parallelogram -- stay empty (NaN in _finalize).
	ax, ay, bx, by, x0, y0 = geom
	H, W = img.shape
	xlo, xhi, ylo, yhi = box
	ys, xs = np.indices((H, W), dtype=np.float64)
	u = (by * (xs - x0) - bx * (ys - y0)) / det
	v = (-ay * (xs - x0) + ax * (ys - y0)) / det
	i, j = np.floor(u), np.floor(v)
	dx = (u - i) * ax + (v - j) * bx                # in-cell offset, real px
	dy = (u - i) * ay + (v - j) * by
	cxs = np.array([0.0, ax, bx, ax + bx])          # parallelogram corners -> bbox
	cys = np.array([0.0, ay, by, ay + by])
	x0c, y0c = cxs.min(), cys.min()
	ncol = int(np.ceil(cxs.max() - x0c)) + 1
	nrow = int(np.ceil(cys.max() - y0c)) + 1
	keep = np.isfinite(img) & (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
	if full_cells_only:
		keep &= _full_mask(i, j, geom, box)
	bcol = np.clip(np.round(dx - x0c).astype(np.int64), 0, ncol - 1)
	brow = np.clip(np.round(dy - y0c).astype(np.int64), 0, nrow - 1)
	flat = (brow * ncol + bcol)[keep]
	vals = img[keep]
	cnt = np.bincount(flat, minlength=nrow * ncol).astype(np.float64)
	s1 = np.bincount(flat, weights=vals, minlength=nrow * ncol)
	s2 = np.bincount(flat, weights=vals * vals, minlength=nrow * ncol)
	return s1.reshape(nrow, ncol), s2.reshape(nrow, ncol), cnt.reshape(nrow, ncol)


def _accum_resample(img, geom, det, N, M, full_cells_only, box):
	# read every cell at the same regular N x M (u,v) grid (bilinear) and average
	# across cells. Uniform count, std = pure cell-to-cell variation.
	ax, ay, bx, by, x0, y0 = geom
	xlo, xhi, ylo, yhi = box
	cx = np.array([xlo, xhi, xlo, xhi])
	cy = np.array([ylo, ylo, yhi, yhi])
	cu = (by * (cx - x0) - bx * (cy - y0)) / det
	cv = (-ay * (cx - x0) + ax * (cy - y0)) / det
	ii, jj = np.meshgrid(np.arange(int(np.floor(cu.min())), int(np.ceil(cu.max())) + 1),
			     np.arange(int(np.floor(cv.min())), int(np.ceil(cv.max())) + 1),
			     indexing="ij")
	ii, jj = ii.ravel().astype(np.float64), jj.ravel().astype(np.float64)
	if full_cells_only:
		m = _full_mask(ii, jj, geom, box)
		ii, jj = ii[m], jj[m]

	uu, vv = np.meshgrid((np.arange(N) + 0.5) / N, (np.arange(M) + 0.5) / M, indexing="ij")
	s1 = np.zeros((N, M))
	s2 = np.zeros((N, M))
	cnt = np.zeros((N, M))
	for ci, cj in zip(ii, jj):                          # O(N*M) memory, loop over cells
		x = x0 + (ci + uu) * ax + (cj + vv) * bx
		y = y0 + (ci + uu) * ay + (cj + vv) * by
		samp = _bilinear(img, x, y)
		if not full_cells_only:                     # drop reads outside the ROI box
			inb = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
			samp = np.where(inb, samp, np.nan)
		fin = np.isfinite(samp)
		s1[fin] += samp[fin]
		s2[fin] += samp[fin] ** 2
		cnt[fin] += 1
	return s1, s2, cnt


def average_unit_cell(image, a_px, b_px, origin_px, sub_area=None, shape=None,
		      full_cells_only=True, method="resample"):
	"""Average every unit cell of a periodic lattice onto one.

	image            2D array (H, W) — STEM image or an atomap residual tiff.
	a_px, b_px       lattice basis vectors in pixels, (dx, dy) each (rotation /
	                 shear included — e.g. from the fit's abg/base via calib).
	origin_px        (x0, y0), lattice origin in pixels.
	sub_area         ROI in px [x0, x1, y0, y1]; fold only cells inside it (∩ image).
	shape            (N, M) interpolation grid — **resample only**. Defaults to the
	                 cell's native pixel extent (round|a|, round|b|).
	full_cells_only  use only cells wholly inside the ROI — drops poorly-fit border
	                 cells (default True).
	method           "resample" (default): bilinear-read each cell at the same
	                 fractional (u,v) grid and average across cells -> an N x M
	                 fractional cell, uniform count, std = cell-to-cell variation.
	                 "raw": fold the actual pixels onto the cell's native pixel
	                 footprint (each pixel -> its cell -> its integer px offset),
	                 placed once, no interpolation -> the cell in real space (true
	                 sheared shape) with staircased edges; bins no pixel centre lands
	                 in (gaps + bbox corners outside the cell) are NaN. `shape` is
	                 rejected for "raw".

	Returns (mean, std, count). For "resample" the grid is (N, M) in fractional
	coords (u along a, v along b); for "raw" it is the cell's px bounding box in real
	space. Empty / uncovered bins are NaN; std (sample, ddof=1) is NaN where count < 2.
	"""
	img = np.asarray(image, dtype=np.float64)
	H, W = img.shape
	ax, ay, bx, by, x0, y0, det = _geom(a_px, b_px, origin_px)
	geom = (ax, ay, bx, by, x0, y0)
	box = _box(W, H, sub_area)
	if method == "raw":
		if shape is not None:
			raise ValueError("shape is resample-only; 'raw' uses the native pixel footprint")
		s1, s2, cnt = _accum_raw(img, geom, det, full_cells_only, box)
	elif method == "resample":
		N, M = (int(shape[0]), int(shape[1])) if shape is not None else _native_shape(ax, ay, bx, by)
		s1, s2, cnt = _accum_resample(img, geom, det, N, M, full_cells_only, box)
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
	kwargs pass through (sub_area, shape, full_cells_only, method).
	"""
	a_px, b_px, origin_px = lattice_px_from_fit(lat_params, calib)
	return average_unit_cell(image, a_px, b_px, origin_px, **kwargs)


def _strip_tif(path):
	for ext in (".tiff", ".tif"):
		if path.lower().endswith(ext):
			return path[:-len(ext)]
	return path


def _draw_cell_schematic(ax, lat_params, calib, motif):
	"""Unit-cell parallelogram with the a-axis horizontal (phi removed) and the motif
	atoms at their fractional positions, coloured per element; legend bottom-left."""
	lat0 = dict(lat_params)
	lat0["base"] = [0.0, 0.0, 0.0]                                   # a horizontal, b at gamma (no frame rotation)
	a_px, b_px, _ = lattice_px_from_fit(lat0, calib)
	a, b = np.asarray(a_px, float), np.asarray(b_px, float)
	box = np.array([[0, 0], a, a + b, b, [0, 0]])
	ax.plot(box[:, 0], box[:, 1], "k-", lw=1.5)
	atoms = [(m["atom"], np.asarray(m["coord"], float)) for m in motif.values() if m.get("use", True)]
	for i, el in enumerate(dict.fromkeys(e for e, _ in atoms)):       # unique elements, in order
		xy = np.array([u * a + v * b for e, (u, v) in atoms if e == el])
		ax.scatter(xy[:, 0], xy[:, 1], s=140, color=f"C{i}", edgecolors="k", linewidths=1.2, label=el, zorder=3)
	ax.set_aspect("equal", adjustable="datalim")                    # full-slot box so panels align (titles + middles)
	ax.axis("off")
	ax.legend(loc="center right", bbox_to_anchor=(0.0, 0.5), frameon=False)   # right edge at the plot's left, at cell level


def _uc_figure(out_path, lat_params, calib, motif, mean, std):
	"""Combined PNG: [unit-cell schematic | mean | std], all with the a-axis horizontal
	(the raw maps are rotated by phi to match the schematic). TIFFs stay un-rotated."""
	import matplotlib.pyplot as plt
	import scipy.ndimage as ndi
	phi = lat_params["base"][2]
	fig, ax = plt.subplots(1, 3, figsize=(12, 4))
	ty = 0.75                                                            # titles low, near the centred cells
	_draw_cell_schematic(ax[0], lat_params, calib, motif)
	ax[0].set_title("unit cell", y=ty)
	for axi, arr, title, cmap in ((ax[1], mean, "mean", "gray"), (ax[2], std, "std", "viridis")):
		up = ndi.zoom(arr, 8, order=1)                                    # aggressively upsample before rotating
		rot = ndi.rotate(up, -phi, reshape=True, order=1, cval=np.nan)    # undo phi -> a horizontal
		axi.imshow(rot, origin="lower", cmap=cmap)                        # no colorbar: scale is relative
		axi.set_aspect("equal", adjustable="datalim")                    # full-slot box so panels align
		axi.axis("off")
		axi.set_title(title, y=ty)
	fig.subplots_adjust(wspace=0.05)                                     # tight gap between the panels
	fig.savefig(out_path, dpi=150, bbox_inches="tight")
	plt.close(fig)


def unit_cell_average_to_tiffs(image_path, lat_params, calib, motif=None, out_stem=None, **kwargs):
	"""Read image_path, fold via a fit's lattice, write <stem>_uc_{mean,std,count}.tif.
	With motif given, also write <stem>_uc_figure.png (cell schematic | mean | std,
	in one orientation, element legend bottom-left).

	Convenience for a driver's post-fit step: pass the refined lat_params + motif + calib.
	"""
	import cv2
	img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise FileNotFoundError(image_path)
	kwargs.setdefault("method", "raw")           # real-cell footprint, sharing the schematic's orientation
	mean, std, count = average_unit_cell_from_fit(img, lat_params, calib, **kwargs)
	stem = out_stem if out_stem is not None else _strip_tif(image_path)
	for nm, arr in (("mean", mean), ("std", std), ("count", count)):
		cv2.imwrite(f"{stem}_uc_{nm}.tif", np.asarray(arr, dtype=np.float32))
	if motif is not None:
		_uc_figure(f"{stem}_uc_figure.png", lat_params, calib, motif, mean, std)
	return mean, std, count
