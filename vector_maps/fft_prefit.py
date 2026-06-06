#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Seed a lattice fit from the image FFT: match the first-order Bragg peaks to the
# abg/phi prediction and refine. Rotation (phi) is the primary target; a,b,gamma
# are refined too in prefit mode. The actual fit stays in direct space
# (refinement_run) -- this only produces a better starting lat.
#
# Geometry is done in nm / nm^-1 via the diffpy.structure metric; calib (nm/px) is
# the only pixel-facing scale, applied at the FFT-peak boundary.
# MEMO -- non-square pixels: a per-axis calib (cx, cy) would enter there as a
# diagonal scaling, and since rotation and non-uniform scaling don't commute, phi
# (already in nm-space here) must precede it. Square pixels are the current case
# and use the scalar path below; the anisotropic case is deferred.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np
from diffpy.structure import Lattice

_ORDERS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)], dtype=float)


def _rot(phi_deg):
	p = np.radians(phi_deg)
	return np.array([[np.cos(p), np.sin(p)], [-np.sin(p), np.cos(p)]])    # get_coords' R(phi)


def _real_basis_px(lat_params, calib):
	# real lattice basis (a, b) in pixels, get_coords convention; used by the tests
	a, b, gamma = lat_params['abg']
	g = np.radians(gamma)
	L0 = np.array([[a, b * np.cos(g)], [0.0, b * np.sin(g)]])
	L = _rot(lat_params['base'][2]) @ L0
	return L[:, 0] / calib, L[:, 1] / calib


def _recip_basis_nm(lat_params):
	# image-oriented reciprocal basis (nm^-1), columns a*, b*. diffpy supplies the
	# metric G; reciprocal = (L0^-1)^T = L0 @ G^-1, then rotated by phi.
	a, b, gamma = lat_params['abg']
	g = np.radians(gamma)
	if abs(np.sin(g)) < 1e-9:
		raise ValueError("degenerate lattice: gamma is 0 or 180 deg")
	L0 = np.array([[a, b * np.cos(g)], [0.0, b * np.sin(g)]])
	Ginv = np.linalg.inv(np.array(Lattice(a, b, 1.0, 90, 90, gamma).metrics)[:2, :2])
	return _rot(lat_params['base'][2]) @ (L0 @ Ginv)


def _fft_mag(image, sub_area):
	# centered magnitude spectrum of the (ROI-cropped, mean-removed, Hann-windowed) image
	img = np.asarray(image, dtype=np.float64)
	if sub_area is not None:                        # ROI in px [x0, x1, y0, y1]
		x0, x1, y0, y1 = (int(round(v)) for v in sub_area)
		img = img[y0:y1, x0:x1]
	img = np.nan_to_num(img - np.nanmean(img))
	H, W = img.shape
	win = np.hanning(H)[:, None] * np.hanning(W)[None, :]
	return np.abs(np.fft.fftshift(np.fft.fft2(img * win))), H, W


def _snap_peak(mag, row, col, radius):
	# brightest pixel within +-radius of (row, col), refined by a 3x3 centroid
	H, W = mag.shape
	r0, r1 = max(0, int(row - radius)), min(H, int(row + radius) + 1)
	c0, c1 = max(0, int(col - radius)), min(W, int(col + radius) + 1)
	if r1 <= r0 or c1 <= c0:
		return None
	win = mag[r0:r1, c0:c1]
	pr, pc = np.unravel_index(int(np.argmax(win)), win.shape)
	pr, pc = pr + r0, pc + c0
	rr0, rr1 = max(0, pr - 1), min(H, pr + 2)
	cc0, cc1 = max(0, pc - 1), min(W, pc + 2)
	sub = mag[rr0:rr1, cc0:cc1]
	s = sub.sum()
	if s <= 0:
		return float(pr), float(pc)
	ys, xs = np.mgrid[rr0:rr1, cc0:cc1]
	return float((ys * sub).sum() / s), float((xs * sub).sum() / s)


def _lat_from_recip_nm(recip, lat_params, refine_abg):
	# fitted reciprocal basis (nm^-1, columns a*, b*) -> updated lat_params (nm).
	# Real basis L = (recip^-1)^T; a,b,gamma from its metric G = L^T L, phi from a.
	L = np.linalg.inv(recip).T
	out = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in lat_params.items()}
	out['base'] = [lat_params['base'][0], lat_params['base'][1],
		       float(np.degrees(np.arctan2(-L[1, 0], L[0, 0])))]
	if refine_abg:
		G = L.T @ L
		a, b = np.sqrt(G[0, 0]), np.sqrt(G[1, 1])
		gamma = np.degrees(np.arccos(np.clip(G[0, 1] / (a * b), -1.0, 1.0)))
		out['abg'] = [float(a), float(b), float(gamma)]
	return out


def fft_prefit(image, lat_params, calib, refine_abg=False, sub_area=None, search_frac=0.3):
	"""Seed lat_params from the image FFT by matching the 6 first-order Bragg peaks
	to the abg/phi prediction and refining the reciprocal basis.

	refine_abg=False ("align"): refine the frame rotation phi only.
	refine_abg=True  ("prefit"): also refine a, b, gamma.
	Geometry in nm/nm^-1 (diffpy metric); calib (nm/px, isotropic -- see top MEMO
	for non-square) maps to FFT bins. Returns an updated lat_params; the actual fit
	stays in direct space.
	"""
	recip_nm = _recip_basis_nm(lat_params)                # columns a*, b* (nm^-1, oriented)
	mag, H, W = _fft_mag(image, sub_area)
	cx, cy = W / 2.0, H / 2.0

	pred = (_ORDERS @ recip_nm.T) * calib                 # (6,2) cycles/px (fx, fy)
	radius = max(2.0, search_frac * np.hypot(pred[:, 0] * W, pred[:, 1] * H).min())

	orders_ok, freqs = [], []
	for (m, n), (fx, fy) in zip(_ORDERS, pred):
		snap = _snap_peak(mag, cy + fy * H, cx + fx * W, radius)
		if snap is None:
			continue
		sr, sc = snap
		orders_ok.append([m, n])
		freqs.append([(sc - cx) / W / calib, (sr - cy) / H / calib])   # bin -> cycles/px -> nm^-1
	if len(orders_ok) < 2:
		raise ValueError("fft_prefit: fewer than 2 first-order peaks matched")

	Gfit, *_ = np.linalg.lstsq(np.array(orders_ok), np.array(freqs), rcond=None)
	recip_fit = np.array([[Gfit[0, 0], Gfit[1, 0]], [Gfit[0, 1], Gfit[1, 1]]])   # columns a*, b*
	return _lat_from_recip_nm(recip_fit, lat_params, refine_abg)
