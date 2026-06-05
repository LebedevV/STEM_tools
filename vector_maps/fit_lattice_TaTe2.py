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


def run_fit_pipeline(folder, fname, calib, preview=False):
	# two prefits on a small ROI, motif fixed
	_, vec = refinement_run(folder, None, fname, calib, lat_params, motif,
				show_initial_spots=preview, vec_scale=0.01, sub_area=[1, 3, 1, 3], max_dist=0.15)
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
				show_initial_spots=preview, vec_scale=0.25, sub_area=[0.5, 4.5, 0.5, 4.5], max_dist=0.15)
	unpack_to_dicts(vec, lat_params, motif, extra_pars)
	return meta


if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument("--folder", default="./")
	p.add_argument("--fname", required=True)
	p.add_argument("--calib", type=float)
	p.add_argument("--preview", action="store_true")
	args = p.parse_args()
	folder = os.path.join(args.folder, "")
	calib = args.calib if args.calib is not None else read_frame_calib(folder, args.fname, fallback=CALIB)
	run_fit_pipeline(folder, args.fname, calib, preview=args.preview)
