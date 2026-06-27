#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import filecmp
import tomllib
import numpy as np
import hyperspy.api as hs
import pandas as pd
import cv2
from pathlib import Path

import matplotlib.pyplot as plt

from dicts_handling import unpack_to_dicts

def rotate_vec(v,an):
	'''
	Rotate 2-vector in plane
	inputs:
		v - list or nparray, 2-vector
		an - float, rotation angle in degrees
	outputs:
		(x,y) - tuple, 2-vector
	'''

	an = an/180*np.pi
	c = np.cos(an)
	s = np.sin(an)
	vx,vy = v
	#print(vx,vy)
	x = vx*c-vy*s
	y = vy*c+vx*s
	#print(x,y)
	return x,y

def resolve_frame_path(folder, fname):
	'''Path to the frame TIFF, accepting .tif/.tiff/.TIF/.TIFF (a trailing tiff
	extension on `fname` is ignored). If several variants are present they must be
	byte-identical, otherwise it raises.'''
	low = fname.lower()
	if low.endswith('.tiff'):
		stem = fname[:-5]
	elif low.endswith('.tif'):
		stem = fname[:-4]
	else:
		stem = fname
	cands = [os.path.join(folder, stem + ext) for ext in ('.tif', '.tiff', '.TIF', '.TIFF')]
	found = [p for p in cands if os.path.exists(p)]
	if not found:
		raise FileNotFoundError(os.path.join(folder, stem + '.{tif,tiff,TIF,TIFF}'))
	differ = [p for p in found[1:] if not filecmp.cmp(found[0], p, shallow=False)]
	if differ:
		raise ValueError('frame TIFF variants differ byte-wise: ' + ', '.join([found[0]] + differ))
	return found[0]


def load_frame(folder,fname,calib_size_by_px): #TODO - we do not really have to have a tiff
	'''
	Loads a tiff file provided as a hyperspy object
	inputs:
		folder - str, path to the workfolder
		fname - str, basename of the tif file
	output:
		s - hyperspy 2Dsignal with pixels enforced to be square 
	'''
	s = hs.load(resolve_frame_path(folder, fname))
	metadata = {}
	#'''
	metadata['fname'] = fname#TODO should we return this mdata?

	#xy directions not checked! has to be verified
	#d0,d1 = imsize[0]/imsize_px[0],imsize[1]/imsize_px[1]
	#print(d0,d1)
	
	d0 = calib_size_by_px#calib_size/calib_px
	metadata['nm_per_pix'] = d0

	#TODO Flaw!!! atomap apparently does not support non-sqare pixels!
	s.axes_manager[0].scale = d0
	s.axes_manager[1].scale = d0
	s.axes_manager[0].units = 'nm'
	s.axes_manager[1].units = 'nm'
	
	return s
	
def read_frame_calib(folder,fname,fallback=None,atol=1e-6):
	'''
	Calibration (nm/pixel) from a <fname>_frame.txt sidecar (keys
	xres_px/yres_px/xreal_nm/yreal_nm). Returns `fallback` if the sidecar is
	absent; raises on anisotropic pixels.
	'''
	frame_path = os.path.join(folder, fname + '_frame.txt')
	if not os.path.exists(frame_path):
		if fallback is None:
			raise FileNotFoundError(frame_path)
		return fallback
	vals = {}
	with open(frame_path) as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			key, value = line.split('\t', 1)
			vals[key] = value
	calib_x = float(vals['xreal_nm']) / float(vals['xres_px'])
	calib_y = float(vals['yreal_nm']) / float(vals['yres_px'])
	if not np.isclose(calib_x, calib_y, atol=atol):
		raise ValueError(f'Anisotropic calibration in {frame_path}: {calib_x} vs {calib_y}')
	return calib_x


def calib_from_frame_size(folder,fname,scan_s):
	'''
	Calibration (nm/pixel) = frame size (scan_s, Angstrom) / the frame's own pixel
	count. Recomputed against the real grid, so it tracks the actual image even if
	it was rebinned.
	'''
	img = cv2.imread(resolve_frame_path(folder, fname), cv2.IMREAD_UNCHANGED)
	if img.shape[0] != img.shape[1]:
		raise ValueError(f'non-square frame {img.shape[:2]} for {fname}; calibration assumes square pixels')
	return (scan_s / 10.0) / img.shape[0]


def read_toml_calib(folder,fname,toml_path):
	'''
	Calibration (nm/pixel) for a synthetic frame: reads scan_s (frame size, Angstrom)
	from the descriptive toml, divided by the frame's own pixel count.  toml_path is
	used exactly as supplied; manifest-generated paths are absolute.
	'''
	if toml_path is None:
		raise ValueError('toml_path is required for toml-derived calibration')
	with open(toml_path, 'rb') as f:
		scan_s = tomllib.load(f)['lamella_settings']['scan_s']
	return calib_from_frame_size(folder, fname, scan_s)


def export_data(folder,sf,fname,lat_params_vec,raw_lat_params,raw_motif,raw_extra_pars,metadata):
	'''
	Save variables as csv
	
	'''

	#check the existance of the output folder
	entries = os.listdir(folder)
	if sf not in entries:
		os.mkdir(folder+sf)
		print('Folder %s created' % sf)

	lat_params_fin, motif_fin, extra_pars_fin = unpack_to_dicts(lat_params_vec, raw_lat_params,raw_motif, raw_extra_pars)
	#lat_params_fin,motif_fin = unpack_vector(lat_params_vec,raw_lat_params,raw_motif)
	
	export_name = folder+sf+'/'+sf
	
	mdata = pd.DataFrame.from_dict(metadata,orient='index')
	mdata.to_csv(export_name +'_metadata.csv',sep='\t')
	
	par = pd.DataFrame.from_dict(lat_params_fin,orient='index')
	par.to_csv(export_name + '_lattice.csv',sep='\t')
	
	motif = pd.DataFrame.from_dict(motif_fin,orient='index')
	motif.to_csv(export_name + '_motif.csv',sep='\t')

	extra = pd.DataFrame.from_dict(extra_pars_fin,orient='index')
	extra.to_csv(export_name + '_extra.csv',sep='\t')

def vector_map_calc(phi_deg,df):
	"""Compute observed-minus-fitted vector-map quantities.

	``phi_deg`` is the fitted lattice rotation ``base[2]`` in degrees, matching
	``get_coords_from_ij`` and the config/schema convention.  ``vproj`` stores
	scalar components of each displacement projected onto the fitted a-axis and
	the corresponding normal-to-a direction in image coordinates.
	"""
	phi_rad = np.deg2rad(phi_deg)
	a_hat = np.array([np.cos(phi_rad), -np.sin(phi_rad)])
	a90_hat = np.array([np.sin(phi_rad), np.cos(phi_rad)])
	#fin_lat = get_coords_from_ij(f_ij,param,no_modulation,only_ortho,max_lim)[0]
	
	obs = np.array(df[['x_obs','y_obs']].values)
	calc = np.array(df[['x_theor_new','y_theor_new']].values)
	
	vdiff_xy = obs - calc
	df['vdiff_xy'] = vdiff_xy.tolist()
	 
	vdiff_ref = np.nanmean(vdiff_xy,axis=0)
	
	vproj = np.column_stack((vdiff_xy @ a_hat, vdiff_xy @ a90_hat))
	df['vproj'] = vproj.tolist()
	
	#print(np.std(vproj,axis=0))
	#print(np.std(vdiff_xy,axis=0))
	#print('ref',vdiff_ref)
	#print('Std rot',np.sqrt(np.sum(np.std(vproj,axis=0)**2)),'len',len(vproj))
	#print('Std raw',np.sqrt(np.sum(np.std(vdiff_xy,axis=0)**2)),'len',len(vdiff_xy))
	
	#std_to_report = np.std(vproj,axis=0)
	
	
	vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	df['vdist'] = vdist
	
	vdiff_xy_corr = vdiff_xy - vdiff_ref
	df['vdiff_xy_corr'] = vdiff_xy_corr.tolist()
	print('Test dist',sum(vdist)/len(vdist))
	#print(np.mean(vdiff_xy_corr,axis=0))
	
	
	
	#print(np.std(abs(vdiff_xy_corr),axis=0))
	ang = [np.arctan2(j,i) for i,j in vdiff_xy]
	ang_corr = [np.arctan2(j,i) for i,j in vdiff_xy_corr] #np.angle(vdiff_xy, deg=True)
	df['ang'] = ang
	df['ang_corr'] = ang_corr
	
	std_to_report = np.std(abs(vdiff_xy),axis=0)
	#vdist = np.sqrt(np.sum(vdiff_xy**2,axis=1))
	#str_mean = plot_stats_rep(vdist,fname_save)
	
	return std_to_report,df




def load_and_trim_cv2(path, white_threshold=245):
	"""
	Load an image with OpenCV and trim (near-)white borders.
	Keeps all non-white pixels intact.
	Returns an array in RGB/RGBA for matplotlib.
	Code has been created with AI assistance (OpenAI GPT-5) and manually reviewed
	"""
	path = str(Path(path))
	img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
	if img is None:
		raise FileNotFoundError(f"Could not read image: {path}")

	# Grayscale
	if img.ndim == 2:
		nonwhite = img < white_threshold
		if not np.any(nonwhite):
			return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
		ys, xs = np.where(nonwhite)
		crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1]
		return cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

	# Color with alpha (BGRA)
	if img.shape[2] == 4:
		b, g, r, a = cv2.split(img)
		# A pixel is considered "content" if it’s visible (a>0) and not pure white
		nonwhite = (a > 0) & ((r < white_threshold) | (g < white_threshold) | (b < white_threshold))
		if not np.any(nonwhite):
			return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
		ys, xs = np.where(nonwhite)
		crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
		return cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA)

	# Color (BGR)
	b, g, r = cv2.split(img)
	nonwhite = (r < white_threshold) | (g < white_threshold) | (b < white_threshold)
	if not np.any(nonwhite):
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		
	ys, xs = np.where(nonwhite)
	crop = img[ys.min():ys.max()+1, xs.min():xs.max()+1, :]
	return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def imshow_no_axes(ax, im):
	ax.imshow(im)
	ax.set_axis_off()


def _panel_image(folder, fname):
	# panel slot "a": the frame preview <fname>.png (one level up from the save
	# folder) if present, else the processing TIFF normalized for display, so the
	# panel still renders when no png was made.
	png = os.path.join(folder, '..', fname + '.png')
	if os.path.exists(png):
		return load_and_trim_cv2(png)
	s = hs.load(resolve_frame_path(str(Path(folder).parent), fname))
	arr = np.asarray(s.data, dtype=float)
	lo, hi = np.nanpercentile(arr, [1, 99])
	arr8 = (np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
	return cv2.cvtColor(arr8, cv2.COLOR_GRAY2RGB)

def plot_output_page(fname,folder,full_df=None):
	'''
	Code has been created with AI assistance (OpenAI GPT-5) and manually reviewed
	'''
	sf = Path(folder.rstrip('/')).name
	pngs = {
		"b": folder+sf + '_vmap_rotated.png',
		"c": folder+sf + '_diff_hist.png',
		"d": folder+sf + '_angles_hist.png',
		"e": folder+sf + '_vmap_rotated_fr0.png',
	}

	# Load & trim; slot "a" is the frame preview <fname>.png, or the processing
	# TIFF when no png was made (see _panel_image).
	imgs = {k: load_and_trim_cv2(Path(v)) for k, v in pngs.items()}
	imgs["a"] = _panel_image(folder, fname)

	# -------- figure layout --------
	# 2 rows, 3 columns; first row col3 is text panel
	# First row slightly taller to make those images "larger"
	fig = plt.figure(figsize=(12, 7))

	gs_parent = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[2, 1], hspace=0.15)

	# Row 1 (top): 1x3 with [1, 1, 0.3] widths
	gs_top = gs_parent[0].subgridspec(nrows=1, ncols=3, width_ratios=[1, 1, 0.3], wspace=0.05)
	ax_a   = fig.add_subplot(gs_top[0, 0])
	ax_b   = fig.add_subplot(gs_top[0, 1])
	ax_txt = fig.add_subplot(gs_top[0, 2])

	# Row 2 (bottom): 1x3 with equal widths
	gs_bot = gs_parent[1].subgridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1], wspace=0.05)
	ax_c   = fig.add_subplot(gs_bot[0, 0])
	ax_d   = fig.add_subplot(gs_bot[0, 1])
	ax_e   = fig.add_subplot(gs_bot[0, 2])


	# Show images
	imshow_no_axes(ax_a, imgs["a"])
	imshow_no_axes(ax_b, imgs["b"])
	imshow_no_axes(ax_c, imgs["c"])
	imshow_no_axes(ax_d, imgs["d"])
	imshow_no_axes(ax_e, imgs["e"])


	#load text vals
	
	df = pd.read_csv(folder+sf +'_metadata.csv', sep="\t", index_col=0)
	print(df)

	try:
		av_dist = df.loc['residual_in_pm','0']
		correct_dist = True
	except KeyError:
		av_dist = df.loc['std'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).to_numpy()[0]
		av_dist = np.sqrt(sum(av_dist**2))*1000
		correct_dist = False
		
	at_num = df.loc['atoms_used','0']
	lat_par = df.loc['param'].apply(lambda x: np.array(x.strip('[]').split(), dtype=float)).to_numpy()[0]
	lat_a = np.round(lat_par[0]*10,2)
	lat_b = np.round(lat_par[1]*10,2)
	lat_g = np.round(lat_par[2],1)


	txt_label = "N = " + str(at_num) +"\n"
	txt_label += "a = " + str(lat_a) +"$\\AA$ \n"
	txt_label += "b = " + str(lat_b) +"$\\AA$ \n"
	txt_label += "$\\gamma $ = " + str(lat_g) +"$^{\\circ}$ \n"
	
	parent_dir = Path(folder).parent

	# -------------------------
	# global calib: only once
	# -------------------------
	stats0_path = parent_dir / f"{fname}_stats.txt"
	if stats0_path.exists():
		with open(stats0_path, "r") as f:
			stats0 = dict(
				line.strip().split(" = ", 1)
				for line in f
				if " = " in line
			)

		img_x = pd.to_numeric(stats0.get("img_x", np.nan), errors="coerce")
		pix_x = pd.to_numeric(stats0.get("pix_x", np.nan), errors="coerce")

		calib = img_x / pix_x * 1000 if pd.notna(img_x) and pd.notna(pix_x) and pix_x != 0 else np.nan

		if pd.notna(calib):
			txt_label += f"Pixel size = {calib:.2g} pm/px\n"

	# -------------------------
	# per-dataset stats
	# -------------------------
	if full_df is not None and "sub_id" in full_df.columns:
		sub_ids = pd.Series(full_df["sub_id"]).dropna().astype(str).unique()

		for sid in sub_ids:
			stats_path = parent_dir / f"{fname}_{sid}_rerun_stats.txt"
			if stats_path.exists():
				with open(stats_path, "r") as f:
					stats_sub = dict(
						line.strip().split(" = ", 1)
						for line in f
						if " = " in line
					)

				SnR = pd.to_numeric(stats_sub.get("SnR", np.nan), errors="coerce")
				n_pix = pd.to_numeric(stats_sub.get("n_pix_1sigma_mean_linear", np.nan), errors="coerce")

				txt_label += f"\n{sid}:\n"
				if pd.notna(SnR):
					txt_label += f"SnR = {SnR:.3g}\n"
				if pd.notna(n_pix):
					txt_label += f"σ = {n_pix:.3g}pix\n"

	else:
		if stats0_path.exists():
			with open(stats0_path, "r") as f:
				stats0 = dict(
					line.strip().split(" = ", 1)
					for line in f
					if " = " in line
				)

			SnR = pd.to_numeric(stats0.get("SnR", np.nan), errors="coerce")
			n_pix = pd.to_numeric(stats0.get("n_pix_1sigma_mean_linear", np.nan), errors="coerce")

			if pd.notna(SnR):
				txt_label += f"SnR = {SnR:.3g}\n"
			if pd.notna(n_pix):
				txt_label += f"σ = {n_pix:.3g}pix\n"
	
	
	# Text area (right side of first row)
	ax_txt.set_axis_off()
	ax_txt.text(
		0.0, 0.80, txt_label,
		transform=ax_txt.transAxes,
		va="top", ha="left",
		fontsize=12
	)

	
	# Overlay text on one image (here, on 'e')
	if correct_dist:
		ttt = "$| \\delta | = $"+str(np.round(float(av_dist),1))+'pm'
	else:
		ttt = "$| \\Delta d | = $"+str(np.round(float(av_dist),1))+'pm'
	ax_e.text(
		0.6, 0.9, ttt,
		transform=ax_e.transAxes,
		va="top", ha="left",
		fontsize=11#, bbox=dict(facecolor="white", alpha=0.7, boxstyle="round,pad=0.2")
	)

	# Optional: tight layout and save
	#plt.tight_layout()
	plt.savefig(folder+"_panel_1.png", dpi=400, bbox_inches="tight")
	plt.close('all')
	#plt.show()

def plot_output_page_diff(fname,folder):
	'''
	Code has been created with AI assistance (OpenAI GPT-5) and manually reviewed
	'''
	sf = Path(folder.rstrip('/')).name
	files = [folder+sf + '_vmap_rotated.png',
		folder+sf + '_vmap_proj_a.png',
		folder+sf + '_vmap_proj_a90.png']
	titles = ["Vector map", "Components $\\parallel ~ a$", "Components $\\perp ~ a$"]

	images = [load_and_trim_cv2(f) for f in files]

	# ---------- Plot in 1×3 grid ----------
	fig, axes = plt.subplots(1, 3, figsize=(12, 4))

	for ax, img, title in zip(axes, images, titles):
		ax.imshow(img)
		ax.set_title(title, fontsize=13, pad=10)
		ax.axis("off")

	plt.tight_layout()
	plt.savefig(folder+"_panel_2.png", dpi=400, bbox_inches="tight")
	plt.close('all')
	#plt.show()


# --- scan-distortion model terms applied by get_coords_from_ij ---
# Flyback hysteresis (Mullarkey et al., Microsc. Microanal. 28, 2022): short scan-line
# flyback compresses the line start, modelled as an exponential displacement along the
# fast axis (image-x, nm). exp_a/exp_b and the slow-axis sx<k>/sy<k> are fit as extra_pars.
def flyback_warp(x, exp_a, exp_b):
	"""Forward flyback map on the fast-scan coordinate x (nm): x + exp_a*exp(-x/exp_b)."""
	x = np.asarray(x, dtype=float)
	return x + exp_a * np.exp(-x / exp_b)


def slow_axis_warp(x, y, extr):
	"""Optional low-order slow-axis (y) distortion: add sx<k>*y**k to x and sy<k>*y**k
	to y for the coeffs (nm) present in extra_pars, k in 1..3. No-op if none set.
	The linear term (sx1/sy1) is degenerate with the lattice shear/scale, so enable it
	only with the lattice pinned (the two-stage centre->edge workflow); the constant is
	left to the lattice origin (shx/shy)."""
	dx = sum(extr[f"sx{k}"][0] * y ** k for k in (1, 2, 3) if f"sx{k}" in extr)
	dy = sum(extr[f"sy{k}"][0] * y ** k for k in (1, 2, 3) if f"sy{k}" in extr)
	return x + dx, y + dy

