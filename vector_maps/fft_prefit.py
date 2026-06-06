#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Seed a lattice fit from the image FFT: match the first-order Bragg peaks to
# the abg/phi prediction and refine. Rotation (phi) is the primary target;
# a,b,gamma are refined too in prefit mode. The actual fit stays in direct space
# (refinement_run) -- this only produces a better starting lat.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np

# first-order reciprocal peaks: +-a*, +-b*, +-(a*+b*)
_ORDERS = np.array([(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)], dtype=float)


def _real_basis_px(lat_params, calib):
	# real lattice basis (a, b) in pixels; mirrors get_coords_from_ij / lattice_px_from_fit
	a, b, gamma = lat_params['abg']
	_, _, phi = lat_params['base']
	g, p = np.radians(gamma), np.radians(phi)
	ax, ay = a * np.cos(p), -a * np.sin(p)
	bx = b * np.cos(g) * np.cos(p) + b * np.sin(g) * np.sin(p)
	by = b * np.sin(g) * np.cos(p) - b * np.cos(g) * np.sin(p)
	return np.array([ax / calib, ay / calib]), np.array([bx / calib, by / calib])


def _recip_basis(a_px, b_px):
	# reciprocal basis (cycles/px) as rows a*, b*: G = (L^-1)^T, L columns a,b
	L = np.array([[a_px[0], b_px[0]], [a_px[1], b_px[1]]], dtype=float)
	if abs(np.linalg.det(L)) < 1e-9:
		raise ValueError("degenerate lattice basis")
	G = np.linalg.inv(L).T
	return G[:, 0], G[:, 1]


def _fft_mag(image, sub_area):
	# centered magnitude spectrum of the (ROI-cropped, mean-removed, Hann-windowed) image
	img = np.asarray(image, dtype=np.float64)
	if sub_area is not None:                       # ROI in px [x0, x1, y0, y1]
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


def _lat_from_basis(a_px, b_px, calib, lat_params, refine_abg):
	# refined real basis (px) -> updated lat_params (nm); align sets phi only
	out = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in lat_params.items()}
	phi = np.degrees(np.arctan2(-a_px[1], a_px[0]))      # ax = a cos(phi), ay = -a sin(phi)
	out['base'] = [lat_params['base'][0], lat_params['base'][1], float(phi)]
	if refine_abg:
		a_nm = float(np.hypot(*a_px) * calib)
		b_nm = float(np.hypot(*b_px) * calib)
		cross = a_px[0] * b_px[1] - a_px[1] * b_px[0]
		dot = a_px[0] * b_px[0] + a_px[1] * b_px[1]
		out['abg'] = [a_nm, b_nm, float(np.degrees(np.arctan2(cross, dot)))]
	return out


def fft_prefit(image, lat_params, calib, refine_abg=False, sub_area=None, search_frac=0.3):
	"""Seed lat_params from the image FFT by matching the 6 first-order Bragg peaks
	to the abg/phi prediction and refining the reciprocal basis.

	refine_abg=False ("align"): refine the frame rotation phi only.
	refine_abg=True  ("prefit"): also refine a, b, gamma.
	calib is nm/px (isotropic); H != W and a px ROI are handled. Returns an
	updated lat_params dict; the actual fit stays in direct space.
	"""
	a_px, b_px = _real_basis_px(lat_params, calib)
	astar, bstar = _recip_basis(a_px, b_px)
	mag, H, W = _fft_mag(image, sub_area)

	cx, cy = W / 2.0, H / 2.0
	pred = _ORDERS @ np.array([astar, bstar])            # (6, 2) predicted (fx, fy) cycles/px
	radius = max(2.0, search_frac * np.hypot(pred[:, 0] * W, pred[:, 1] * H).min())

	orders_ok, freqs = [], []
	for (m, n), (fx, fy) in zip(_ORDERS, pred):
		snap = _snap_peak(mag, cy + fy * H, cx + fx * W, radius)
		if snap is None:
			continue
		sr, sc = snap
		orders_ok.append([m, n])
		freqs.append([(sc - cx) / W, (sr - cy) / H])     # observed (fx, fy)
	if len(orders_ok) < 2:
		raise ValueError("fft_prefit: fewer than 2 first-order peaks matched")

	# observed_freq = orders @ [a*; b*]  ->  least-squares for the reciprocal basis
	Gfit, *_ = np.linalg.lstsq(np.array(orders_ok), np.array(freqs), rcond=None)
	Gmat = np.array([[Gfit[0, 0], Gfit[1, 0]], [Gfit[0, 1], Gfit[1, 1]]])   # columns a*, b*
	Lnew = np.linalg.inv(Gmat).T                          # columns a, b (px)
	return _lat_from_basis(Lnew[:, 0], Lnew[:, 1], calib, lat_params, refine_abg)
