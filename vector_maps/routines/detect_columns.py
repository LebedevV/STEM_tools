#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import hyperspy.api as hs
import atomap.api as am

import scipy
import scipy.ndimage

from .routines import frame_stem, resolve_frame_path



def detect_columns(
	fname,folder,imsize,sep,
	sigma1=1.0,pca=True,
	subtract_background=True,
	thr=0.1,ptonn=0.6,
	interactive=False,
	source_fname=None,
	start_csv=None,
	out_suffix=''
):
	full_stats = {}
	metadata = {}

	# to be provided by workflow
	#fname = 'fph_Pm3m_(25.0, 15.0)_110_haadf_0-25.tif'
	#folder = '/mnt/data2/sci/PZT/tilts/10/out_full/'
	folder = str(folder)
	if not folder.endswith(os.sep):
		folder += os.sep
		
	stem = frame_stem(fname) + out_suffix

	#fpath = os.path.join(folder, fname)
	
	img_name = source_fname if source_fname is not None else fname
	fpath = resolve_frame_path(folder, img_name)

	s = hs.load(fpath)
	if not hasattr(s, "data"):
		raise ValueError("Loaded object is not a 2D signal: %s" % fpath)
		


	metadata['fname'] = fname
	metadata['imsize'] = imsize
	imsize_px = (s.axes_manager[0].size,s.axes_manager[1].size)
	#xy directions not checked! has to be verified
	d0,d1 = imsize[0]/imsize_px[0],imsize[1]/imsize_px[1]
	print(d0,d1)
	#Flaw!!! atomap apparently does not support non-sqare pixels!

	s.axes_manager[0].scale = d0
	s.axes_manager[1].scale = d1
	s.axes_manager[0].units = 'nm'
	s.axes_manager[1].units = 'nm'
	#'''

	#'''
	#s.map(scipy.ndimage.gaussian_filter, sigma=1)


	full_stats['img_x'] = imsize[0]
	full_stats['pix_x'] = s.axes_manager[0].size

	s1 = s.copy()
	s1.map(scipy.ndimage.gaussian_filter, sigma=sigma1)
	s1.plot()

	full_stats['smooth_1st'] = sigma1
	plt.close('all')

	if interactive and start_csv is None:
		s_pks = am.get_feature_separation(s, separation_range=(2, 20),subtract_background=subtract_background,
									  threshold_rel=thr, pca=pca, show_progressbar=False)
		s_pks.plot()
		plt.show()
		sep = int(round(s_pks.axes_manager.coordinates[0]))

	full_stats['separation'] = sep
	full_stats['subtract_background'] = subtract_background
	full_stats['threshold_rel'] = thr
	full_stats['PCA'] = pca

	plt.close('all')
	#atom_positions = am.get_atom_positions(s,subtract_background=subtract_background,
	#								  threshold_rel=thr, pca=pca, separation=sep)
	#plt.plot()
	#atom_positions2 = atom_positions
	#atom_positions2 = ipf.add_atoms_with_gui(s,atom_positions=atom_positions) #doube-check
	
	if start_csv is None:
		atom_positions = am.get_atom_positions(
			s,
			subtract_background=subtract_background,
			threshold_rel=thr,
			pca=pca,
			separation=sep
		)
		atom_positions2 = atom_positions
	else:
		seed_df = pd.read_csv(start_csv)
		atom_positions2 = seed_df[["x_obs0", "y_obs0"]].to_numpy(dtype=float)
	

	plt.close('all')
	sublattice = am.Sublattice(atom_positions2, image=s)

	if interactive:
		sublattice.plot()
		plt.show()
	plt.close('all')
	#
	sublattice.plot()

	png_path = os.path.join(folder, stem + "_am_sep_%s.png" % str(sep))
	plt.savefig(png_path)

	sublattice.find_nearest_neighbors()
	sublattice.refine_atom_positions_using_center_of_mass(s.data)
	
	plt.close('all')
	if interactive:
		sublattice.plot()
		plt.show()

	full_stats['ptonn'] = ptonn
	sublattice.refine_atom_positions_using_2d_gaussian(s.data, percent_to_nn=ptonn)#percent_to_nn=ptonn
	#metadata['percent_to_nn']=ptonn
	#sublattice.refine_atom_positions_using_center_of_mass()


	plt.close('all')
	if interactive:
		sublattice.plot()
		plt.show()
	plt.close('all')
	model_image = sublattice.get_model_image().data
	image_data_subtracted = sublattice.image - model_image

	# image size
	img = sublattice.image
	ny, nx = img.shape
	Y, X = np.indices((ny, nx))

	theor_img = sublattice.get_model_image().data
	crop_p = 0.1

	full_stats['Cropped % from edges (for stats)'] = crop_p
	# pixel limits (10%–90% of frame in each direction)
	roi_x_min = int(crop_p * nx)
	roi_x_max = int((1-crop_p) * nx)
	roi_y_min = int(crop_p * ny)
	roi_y_max = int((1-crop_p) * ny)

	N_pix_area = (roi_y_max - roi_y_min) * (roi_x_max - roi_x_min)
	full_stats['N_pix_area'] = N_pix_area



	x0 = sublattice.x_position #p["x0"]
	y0 = sublattice.y_position #p["y0"]

	I_gauss = sublattice.atom_amplitude_gaussian2d
	s_avg = sublattice.sigma_average
	sx = sublattice.sigma_x		  # σx
	sy = sublattice.sigma_y		  # σy
	phi = sublattice.rotation_ellipticity   # rotation of ellipse (radians)

	mask = (
		(x0 <= roi_x_max) & (y0 <= roi_y_max) &
		(x0 >= roi_x_min) & (y0 >= roi_y_min)
	)


	intensities = I_gauss[mask]
	sigmas = s_avg[mask]


	atom_idx = np.where(mask)[0]
	if len(atom_idx) == 0:
		raise ValueError("No atoms remained inside ROI after cropping.")


	I_theor_1sigma = np.zeros(atom_idx.size, dtype=float)
	n_pix_1sigma   = np.zeros(atom_idx.size, dtype=int)


	roi_mask_pixels = (
		(X >= roi_x_min) & (X <= roi_x_max) &
		(Y >= roi_y_min) & (Y <= roi_y_max)
	)

	for j, k in enumerate(atom_idx):
		cx   = x0[k]
		cy   = y0[k]
		sigx = sx[k]
		sigy = sy[k]
		th   = phi[k]

		# bounding box: ±1σ along principal axes
		rx = int(np.ceil(sigx))
		ry = int(np.ceil(sigy))

		bx_min = max(0, int(np.floor(cx - rx)))
		bx_max = min(nx, int(np.ceil (cx + rx)))
		by_min = max(0, int(np.floor(cy - ry)))
		by_max = min(ny, int(np.ceil (cy + ry)))

		# local coords relative to atom centre
		X_loc = X[by_min:by_max, bx_min:bx_max] - cx
		Y_loc = Y[by_min:by_max, bx_min:bx_max] - cy

		# rotate into ellipse principal frame
		c, s = np.cos(th), np.sin(th)
		xp =  c * X_loc + s * Y_loc
		yp = -s * X_loc + c * Y_loc

		# 1σ ellipse: (xp/σx)^2 + (yp/σy)^2 <= 1
		mask_1sigma = (xp / sigx)**2 + (yp / sigy)**2 <= 1.0

		# values from theoretical image inside this ellipse
		vals_theor = theor_img[by_min:by_max, bx_min:bx_max][mask_1sigma]

		I_theor_1sigma[j] = vals_theor.sum()
		n_pix_1sigma[j]   = mask_1sigma.sum()
		
		
	print(np.min(I_theor_1sigma),np.max(I_theor_1sigma),np.mean(I_theor_1sigma))
	print(np.min(n_pix_1sigma),np.max(n_pix_1sigma),np.mean(n_pix_1sigma))


	valid_sigma = n_pix_1sigma > 0
	if not np.any(valid_sigma):
		raise ValueError("No valid 1-sigma masks were produced.")
		
	signal_per_pix = I_theor_1sigma/n_pix_1sigma

	full_stats['signal_per_pix_min'] = np.min(signal_per_pix)
	full_stats['signal_per_pix_max'] = np.max(signal_per_pix)
	full_stats['signal_per_pix_mean'] = np.mean(signal_per_pix)


	full_stats['I_theor_1sigma_min'] = np.min(I_theor_1sigma)
	full_stats['I_theor_1sigma_max'] = np.max(I_theor_1sigma)
	full_stats['I_theor_1sigma_mean'] = np.mean(I_theor_1sigma)

	full_stats['n_pix_1sigma_min'] = np.min(n_pix_1sigma)
	full_stats['n_pix_1sigma_max'] = np.max(n_pix_1sigma)
	full_stats['n_pix_1sigma_mean'] = np.mean(n_pix_1sigma)

	full_stats['n_pix_1sigma_mean_linear'] = np.sqrt(np.mean(n_pix_1sigma)/np.pi)

	print('Mean sigma pitch',np.sqrt(np.mean(n_pix_1sigma)/np.pi))


	print(np.min(signal_per_pix),np.max(signal_per_pix),np.mean(signal_per_pix))

	print(np.min(sigmas),np.max(sigmas),np.mean(sigmas))

	print(np.min(intensities),np.max(intensities),np.mean(intensities))

	pixel_mask = np.zeros_like(img, dtype=bool)

	for k in range(len(x0)): #we need to mask also those which are centered outside of 10% area
		cx, cy = x0[k], y0[k]
		sigx, sigy = sx[k], sy[k]
		th = phi[k]

		# bounding box ±3σ
		rx = int(np.ceil(3 * sigx))
		ry = int(np.ceil(3 * sigy))

		bx_min = max(0, int(cx - rx))
		bx_max = min(nx, int(cx + rx))
		by_min = max(0, int(cy - ry))
		by_max = min(ny, int(cy + ry))

		# local coords
		X_loc = X[by_min:by_max, bx_min:bx_max] - cx
		Y_loc = Y[by_min:by_max, bx_min:bx_max] - cy

		# rotate
		c, s = np.cos(th), np.sin(th)
		xp =  c * X_loc + s * Y_loc
		yp = -s * X_loc + c * Y_loc

		# 3σ ellipse
		mask_3sigma = (xp / (3*sigx))**2 + (yp / (3*sigy))**2 <= 1.0

		# add this region to global pixel mask
		pixel_mask[by_min:by_max, bx_min:bx_max] |= mask_3sigma



	pixel_mask = ~pixel_mask
	pixel_mask &= roi_mask_pixels
	raw_masked = np.where(pixel_mask, img, np.nan)

	sdiff = hs.signals.Signal2D(raw_masked)
	sdiff.plot()

	N_bckgr_pix = np.count_nonzero(~np.isnan(raw_masked))
	print(N_bckgr_pix)

	N_atoms_in_use = len(atom_idx)
	full_stats['N_atoms'] = N_atoms_in_use
	full_stats['N_pix_bckgr'] = N_bckgr_pix

	full_stats['% bckg area'] = N_bckgr_pix/N_pix_area*100

	vec = raw_masked.flatten()
	vec = vec[~np.isnan(vec)]

	if vec.size == 0:
		#raise ValueError("Background mask is empty.")
		print("Background mask is empty.")
		vec = np.nan

	#plt.close('all')
	#plt.hist(vec,bins=100)
	#plt.show()

	mean_bckgr = np.mean(vec)
	std_bckgr = np.std(vec)
	full_stats['std_bkgr'] = std_bckgr
	full_stats['mean_bkgr'] = mean_bckgr

	SnR = np.mean(signal_per_pix)/std_bckgr
	print(SnR)

	full_stats['SnR'] = SnR


	text = "\n".join(f"{k} = {str(v)}" for k, v in full_stats.items())


	stats_path = os.path.join(folder, stem + "_stats.txt")
	with open(stats_path, "w") as f:
		f.write(text)
		


	diff_path = os.path.join(folder, stem + "_2DG_ptnn_%s_diff2.tif" % str(ptonn))
	csv_path = os.path.join(folder, stem + "_xyI.csv")


	model_image = sublattice.get_model_image().data
	image_data_subtracted = sublattice.image - model_image
	sdiff = hs.signals.Signal2D(image_data_subtracted)
	sdiff.save(diff_path, overwrite=True)


	i_points, i_record, p_record = sublattice.integrate_column_intensity()



	a_gauss = sublattice.atom_amplitude_gaussian2d
	x = sublattice.x_position
	y = sublattice.y_position
	#size = sublattice.pixel_size
	ellipticity = np.asarray(sublattice.ellipticity) - 1
	rot = -np.asarray(sublattice.rotation_ellipticity)
	
	a_std = [a.amplitude_gaussian_std for a in sublattice.atom_list]
	x_std = [a.pixel_x_std for a in sublattice.atom_list] 
	y_std = [a.pixel_y_std for a in sublattice.atom_list]
	r_std = [a.rotation_std for a in sublattice.atom_list]
	sx_std = [a.sigma_x_std for a in sublattice.atom_list]
	sy_std = [a.sigma_y_std for a in sublattice.atom_list]

	df = pd.DataFrame({
		"x_obs0": x,
		"y_obs0": y,
		"ell0": ellipticity,
		"rot0": rot,
		"I_gauss": a_gauss,
		"I0": i_points,
		"sigma_x": sublattice.sigma_x,
		"sigma_y": sublattice.sigma_y,
		'x0_std': x_std,
		'y0_std': y_std,
		'sx_std': sx_std,
		'sy_std': sy_std,
		'rot_std': r_std,
		'Ig_std': a_std
	})
	
	df.to_csv(csv_path, index=False, float_format="%.8g")
	
	
if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument("--fname", required=True)
	parser.add_argument("--folder", required=True)
	parser.add_argument("--imsize", nargs=2, type=float, required=True)

	parser.add_argument("--sep", type=int, default=8)

	parser.add_argument("--sigma1", type=float, default=1.0)
	parser.add_argument("--pca", action="store_true")
	parser.add_argument("--subtract_background", action="store_true")
	parser.add_argument("--thr", type=float, default=0.02)
	parser.add_argument("--ptonn", type=float, default=0.6)

	parser.add_argument("--interactive", action="store_true")

	args = parser.parse_args()

	detect_columns(
		fname=args.fname,
		folder=args.folder,
		imsize=tuple(args.imsize),
		sep=args.sep,
		sigma1=args.sigma1,
		pca=args.pca,
		subtract_background=args.subtract_background,
		thr=args.thr,
		ptonn=args.ptonn,
		interactive=args.interactive
	)
