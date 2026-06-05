#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os

from routines import *
from refinement_routines import *
from plot_routines import *
from detect_columns import *

from dicts_handling import unpack_to_dicts


lat_params = { 'abg':[0.3, 0.42, 89.8],
		'fit_abg':[True,True,True],
		'base':[0.15,0.1,90],
		'fit_base':[True,True,True]
}

motif = {'A_1':{'atom':'Pb',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[False,False]},
	'B_1':{'atom':'Zr',
			'coord':(0.,0.5),
			'I':1,
			'use':True,
			'fit':[False,False]}
}

extra_pars = {}


def run_fit_pipeline(folder, fname, calib, preview=False, dataset_fname=None):
	if dataset_fname is None:
		dataset_fname = fname

	sub_area = [1,4,1,4]
	_,lat_params_vec = refinement_run(folder,None,fname,calib,lat_params,motif,extra_pars=extra_pars,
				show_initial_spots=preview,vec_scale=0.1,sub_area=sub_area,
				max_dist=0.1,dataset_fname=dataset_fname)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)

	sub_area = [0.5,4.5,0.5,4.5]
	save_folder_name = dataset_fname+'_fix_motif'
	_,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit,motif_prefit,
				show_initial_spots=preview,vec_scale=0.1,
				sub_area=sub_area,max_dist=0.1,dataset_fname=dataset_fname)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)

	motif_prefit['A_1']['fit'] = [True,True]
	motif_prefit['B_1']['fit'] = [True,True]

	save_folder_name = dataset_fname+'_free_motif'
	metadata,lat_params_vec = refinement_run(folder,save_folder_name,fname,calib,lat_params_prefit,motif_prefit,
				show_initial_spots=preview,vec_scale=0.1,
				max_dist=0.1,sub_area=sub_area,
				export_sublattice_xy=True,dataset_fname=dataset_fname)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params_prefit, motif_prefit, extra_pars_prefit)

	return metadata


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--folder", default="./")
	parser.add_argument("--fname", required=True)
	parser.add_argument("--calib", type=float)
	parser.add_argument("--imsize", nargs=2, type=float, default=None)
	parser.add_argument("--preview", action="store_true")
	args = parser.parse_args()

	folder_s = os.path.join(args.folder, "")
	calib = args.calib if args.calib is not None else read_frame_calib(folder_s, args.fname)
	run_fit_pipeline(folder=folder_s, fname=args.fname, calib=calib, preview=args.preview)

	folder = Path(args.folder)
	fname = args.fname   # stem, without .tif

	sf_A = f"{fname}_full_free_motif_A"
	sf_B = f"{fname}_full_free_motif_B"
	csv_A = folder / sf_A / f"{sf_A}_full.csv"
	csv_B = folder / sf_B / f"{sf_B}_full.csv"

	ptonn_A = 0.6
	tmp_sf = "_A_rerun"
	detect_columns(
		fname=fname + '.tif',
		folder=str(folder),
		imsize=tuple(args.imsize),
		sep=2,
		sigma1=1,
		pca=False,
		subtract_background=True,
		thr=0.1,
		ptonn=ptonn_A,
		interactive=False,
		start_csv=str(csv_A),
		out_suffix=tmp_sf,
	)

	csv_A_r = folder / f"{fname}{tmp_sf}_xyI.csv"
	ptonn_B = 0.4
	tmp_sf = "_B_rerun"
	detect_columns(
		fname=fname + '.tif',
		folder=str(folder),
		imsize=tuple(args.imsize),
		sep=2,
		sigma1=1,
		pca=False,
		subtract_background=True,
		thr=0.1,
		ptonn=ptonn_B,
		interactive=False,
		source_fname=f"{fname}_A_rerun_2DG_ptnn_{ptonn_A}_diff2.tif",
		start_csv=str(csv_B),
		out_suffix=tmp_sf,
	)
	csv_B_r = folder / f"{fname}{tmp_sf}_xyI.csv"

	df_A = pd.read_csv(csv_A_r)
	df_B = pd.read_csv(csv_B_r)
	df_A["sub_id"] = "A"
	df_B["sub_id"] = "B"
	df_AB = pd.concat([df_A, df_B], ignore_index=True, sort=False)

	merged_name = f"{fname}_sub_AB"
	csv_AB = folder / f"{merged_name}_xyI.csv"
	df_AB.to_csv(csv_AB, index=False, float_format="%.8g")

	print('Second refinement')
	run_fit_pipeline(folder=folder_s, fname=args.fname, calib=calib, preview=args.preview, dataset_fname=merged_name)
