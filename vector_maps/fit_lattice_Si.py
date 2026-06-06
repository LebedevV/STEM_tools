#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os

from routines import *
from refinement_routines import *
from plot_routines import *

#Meant to be 0.3867, 0.5469
lat_params = { 'abg':[0.3805, 0.5369, 89.75],
		'fit_abg':[True,True,True],
		'base':[-0.0005,.20,1.88],
		'fit_base':[True,True,True]
}

#Atom at (0,0); first sublattice. Since all other atoms are functionally connected to this one,
#it is reasonable to fix it due to a full correlation with shx/shy (lat_params['base'][0] and lat_params['base'][1])
motif = {'A_1':{'atom':'Si_1',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[False,False]},
}

#Centered atom of the first sublattice. Since 'eq' are present and not None, 'coords' and 'fit' are disabled
motif['A_1c'] = {	'atom':'Si_2',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':  ["= motif['A_1'][0] + extra_pars['centering_a']", "= motif['A_1'][1] + extra_pars['centering_b']"]
			}

#Second sublattice; 'A_1' + dumbbell vector (in polar coordinates)
motif['B_1'] =  {'atom':'Si_3',
			'coord':(0.,0.2),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['A_1'][0] + extra_pars['db_dist']*np.sin(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][0]",
					"= motif['A_1'][1] + extra_pars['db_dist']*np.cos(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][1]"]}

#Second sublattice; centered
motif['B_1c'] = {'atom':'Si_4',
			'coord':(0.5,0.7),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['B_1'][0] + extra_pars['centering_a']", "= motif['B_1'][1] + extra_pars['centering_b']"]}

#Extra variables - dumbbell vector in absolute polar coordinates relative to b; expected to be (L,0) but can be refined
#Centering vector in fractional coordinates; True/False enables/disables refinement
extra_pars = {'db_dist':(0.1,True),
		'db_angle':(0,True),
		'centering_a':(0.5,True),
		'centering_b':(0.5,True)}

#Calibration: only the nm-per-pixel ratio matters (1024px frame from 90% of a 16nm scan)
CALIB = 16/1024*.9  # nm/pixel; override with --calib or a <fname>_frame.txt sidecar


def run_fit_pipeline(folder, fname, calib, preview=False, unit_cell=False, shift_ab=None, do_fft_align=False, do_fft_prefit=False):
	#prefit on a manual ROI
	sub_area = [2,4,2,4]  #in nm
	_,lat_params_vec = refinement_run(folder,None,fname,calib,lat_params,motif,extra_pars=extra_pars,
						show_initial_spots=preview,vec_scale=0.1,sub_area=sub_area,max_dist=0.1,
						do_fft_align=do_fft_align,do_fft_prefit=do_fft_prefit)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)

	#automated refinements with gradual expansion of the ROI
	st_p = 2
	r = 2
	k = st_p+r
	while k<16:
		sub_area = [st_p,k,st_p,k]
		lat_params_prefit['fit_abg'] = [False,False,False]
		_,lat_params_vec = refinement_run(folder,None,fname,calib,lat_params_prefit,motif_prefit,extra_pars=extra_pars_prefit,
							show_initial_spots=False,vec_scale=0.01,sub_area=sub_area,max_dist=0.1)
		lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
		lat_params_prefit['fit_abg'] = [True,True,True]
		_,lat_params_vec = refinement_run(folder,None,fname,calib,lat_params_prefit,motif_prefit,extra_pars=extra_pars_prefit,
							show_initial_spots=False,vec_scale=0.01,sub_area=sub_area,max_dist=0.1)
		lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
		k+=2

	#full image refinement with outputs
	_,lat_params_vec = refinement_run(folder,fname+'_fix_motif',fname,calib,lat_params,motif,extra_pars=extra_pars_prefit,
						show_initial_spots=preview,vec_scale=0.1,sub_area=None,max_dist=0.1)
	lat_params_prefit,motif_prefit,extra_pars_prefit = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)

	#central area refinement
	meta,lat_params_vec = refinement_run(folder,fname+'_fix_motif_center',fname,calib,lat_params,motif,extra_pars=extra_pars_prefit,
						show_initial_spots=preview,vec_scale=0.1,sub_area=[2,12,2,12],max_dist=0.1,shift_ab=shift_ab)
	lp, _, _ = unpack_to_dicts(lat_params_vec, lat_params, motif, extra_pars)
	if unit_cell:
		from unit_cell_average import unit_cell_average_to_tiffs
		unit_cell_average_to_tiffs(folder + fname + ".tif", lp, calib)
	return meta


if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--folder", default="./")
	p.add_argument("--fname", required=True)
	p.add_argument("--calib", type=float)
	p.add_argument("--preview", action="store_true")
	p.add_argument("--unit-cell", dest="unit_cell", action="store_true",
		       help="after the fit, write <fname>_uc_{mean,std,count}.tif")
	p.add_argument("--shift-ab", dest="shift_ab", action="store_true",
		       help="re-reference the origin A_1->B_1 before the final fit")
	p.add_argument("--fft-align", dest="fft_align", action="store_true",
		       help="FFT-seed the frame rotation before the first fit")
	p.add_argument("--fft-prefit", dest="fft_prefit", action="store_true",
		       help="FFT-seed rotation + a,b,gamma before the first fit")
	args = p.parse_args()
	folder = os.path.join(args.folder, "")
	calib = args.calib if args.calib is not None else read_frame_calib(folder, args.fname, fallback=CALIB)
	run_fit_pipeline(folder, args.fname, calib, preview=args.preview, unit_cell=args.unit_cell,
			 shift_ab=('A_1', 'B_1') if args.shift_ab else None,
			 do_fft_align=args.fft_align, do_fft_prefit=args.fft_prefit)
