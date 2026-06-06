#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os

from routines import *
from refinement_routines import *
from plot_routines import *

# --- TaTe2: one Ta + two Te sublattices ---
lat_params = {'abg': [0.32, 1.18, 145.5], 'fit_abg': [True, True, True],
	      'base': [0.0, 0.0, -45], 'fit_base': [True, True, True]}
motif = {
	'A_1': {'atom': 'Ta', 'coord': (0.0, 0.0),  'I': 1, 'use': True, 'fit': [False, False]},
	'B_1': {'atom': 'Te', 'coord': (0.0, 0.25), 'I': 1, 'use': True, 'fit': [False, False]},
	'B_2': {'atom': 'Te', 'coord': (0.0, 0.75), 'I': 1, 'use': True, 'fit': [False, False]},
}
extra_pars = {}
CALIB = 5/266  # nm/pixel; override with --calib or a <fname>_frame.txt sidecar


def run_fit_pipeline(folder, fname, calib, preview=False, unit_cell=False, shift_ab=None, do_fft_align=False, do_fft_prefit=False):
	# two prefits on a small ROI, motif fixed
	_, vec = refinement_run(folder, None, fname, calib, lat_params, motif,
				show_initial_spots=preview, vec_scale=0.01, sub_area=[1, 3, 1, 3], max_dist=0.15,
				do_fft_align=do_fft_align, do_fft_prefit=do_fft_prefit)
	lp, mo, ep = unpack_to_dicts(vec, lat_params, motif, extra_pars)
	_, vec = refinement_run(folder, None, fname, calib, lp, mo,
				show_initial_spots=preview, vec_scale=0.01, sub_area=[1, 3, 1, 3], max_dist=0.15)
	lp, mo, ep = unpack_to_dicts(vec, lat_params, motif, extra_pars)

	# fixed-motif fit with outputs
	_, vec = refinement_run(folder, 'Fixed', fname, calib, lp, mo,
				show_initial_spots=preview, vec_scale=0.25, sub_area=[0.5, 4.5, 0.5, 4.5], max_dist=0.15)
	lp, mo, ep = unpack_to_dicts(vec, lat_params, motif, extra_pars)

	# free the Te sublattices
	mo['B_1']['fit'] = [True, True]
	mo['B_2']['fit'] = [True, True]
	meta, vec = refinement_run(folder, 'free', fname, calib, lp, mo,
				show_initial_spots=preview, vec_scale=0.25, sub_area=[0.5, 4.5, 0.5, 4.5], max_dist=0.15, shift_ab=shift_ab)
	lp, mo, ep = unpack_to_dicts(vec, lat_params, motif, extra_pars)
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
