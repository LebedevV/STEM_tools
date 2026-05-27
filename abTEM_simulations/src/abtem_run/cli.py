#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

# zarr<3 is needed!
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import abtem
import ase
import matplotlib.pyplot as plt
import numpy as np
import tomli_w

# Importing the package runs abtem monkey-patches in __init__.py before any
# abtem calls happen in this module.
from . import config as confread
from . import simulation as sim
from .config import AppConfig

@dataclass
class RunContext:
	cfg: AppConfig

	# resolved paths
	folder_sim: str
	folder: str

	# resolved microscope/sim
	do_full_run: bool
	HT_value: float
	do_diffraction: bool
	override_sampling: float | bool

	# resolved lamella geometry
	scan_start: tuple[float, float]
	scan_stop: tuple[float, float]
	lamella_sizes: tuple[float, float, float]
	global_tilt: tuple[float, float]
	tilt_degrees: bool

	convergence_angle: float
	cbed_max_angle: float | str

	# resolved detectors (abtem objects)
	haadf_detector: object
	abf_detector: object
	bf_detector: object

	element_to_remove: str
	probability_of_vac: float
	add_vacancies_toggle: bool
	
	# frozen phonons settings
	frozen_phonons: int | None
	fph_sigma: float | None
	phonons_seed: int

	# dask distributed client — held here so it is not GC'd before the run ends
	dask_client: object | None


#config is expected to be in the same folder as a code
#'config.toml' is in use if None is provided

#######
####### Loading params

#Short names for cif files
cif_files = {
		'Pbam':'Pbam.cif',
		'Pm3m':'Pm3m.cif',
		'I4cm':'I4cm.cif',
		'Ima2':'Ima2.cif',
		'CC':'CC.cif',
		'R-3c':'R-3c.cif',
		'2212':'2212_1541124.cif',
		'PZO':'PZO.cif'
}



def _as_list(v):
	return v if isinstance(v, list) else [v]


def expand_cfg(cfg: AppConfig):
	"""
	Yield AppConfig objects, each with scalar values, for the cartesian product of:
	frozen_phonons, fph_sigma, thickness, (global_tilt_a, global_tilt_b), probability_of_vac, HT_value.
	"""
	base = cfg.model_dump()

	# Sentinels ('None' for frozen_phonons, False for fph_sigma) pass through verbatim;
	# resolve_context unwraps them to Python None at use.
	frozen_list = _as_list(base["simulations"]["frozen_phonons"])
	sigma_list  = _as_list(base["simulations"]["fph_sigma"])
	thick_list  = [float(x) for x in _as_list(base["lamella_settings"]["thickness"])]
	pvac_list   = [float(x) for x in _as_list(base["lamella_settings"]["probability_of_vac"])]
	ht_list	 = [int(x)   for x in _as_list(base["microscope"]["HT_value"])]

	ta_list = [float(x) for x in _as_list(base["lamella_settings"]["global_tilt_a"])]
	tb_list = [float(x) for x in _as_list(base["lamella_settings"]["global_tilt_b"])]

	# tilt pairs: full cartesian of a and b
	tilt_pairs = list(product(ta_list, tb_list))

	for frozen, sigma, thick, (ta, tb), pvac, ht in product(
		frozen_list, sigma_list, thick_list, tilt_pairs, pvac_list, ht_list
	):
		d = deepcopy(base)
		d["simulations"]["frozen_phonons"] = frozen
		d["simulations"]["fph_sigma"] = sigma
		d["lamella_settings"]["thickness"] = float(thick)
		d["lamella_settings"]["global_tilt_a"] = float(ta)
		d["lamella_settings"]["global_tilt_b"] = float(tb)
		d["lamella_settings"]["probability_of_vac"] = float(pvac)
		d["microscope"]["HT_value"] = int(ht)

		yield AppConfig.model_validate(d)

def resolve_context(cfg, global_tilt: tuple[float, float] | None = None):
	folder_sim = cfg.paths.folder_sim + cfg.paths.extr
	folder = cfg.paths.folder

	do_full_run = cfg.simulations.do_full_run
	HT_value = cfg.microscope.HT_value
	do_diffraction = cfg.microscope.do_diffraction
	override_sampling = cfg.simulations.override_sampling

	borders = cfg.lamella_settings.borders
	scan_s = cfg.lamella_settings.scan_s
	thickness = cfg.lamella_settings.thickness

	scan_start = (borders * 2, borders * 2)
	scan_stop = (borders * 2 + scan_s, borders * 2 + scan_s)
	lamella_sizes = (borders * 2 + scan_s, borders * 2 + scan_s, thickness)

	haadf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.haadfinner, outer=cfg.microscope.haadfouter
	)
	abf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.abfinner, outer=cfg.microscope.abfouter
	)
	bf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.bfinner, outer=cfg.microscope.bfouter
	)

	if global_tilt is None:
		global_tilt = (cfg.lamella_settings.global_tilt_a, cfg.lamella_settings.global_tilt_b)

	element_to_remove = cfg.lamella_settings.element_to_remove
	probability_of_vac = cfg.lamella_settings.probability_of_vac
	add_vacancies_toggle = cfg.lamella_settings.add_vacancies_toggle
	
	###Configuring computational environment

	#No of threads; limited by video-memory, can fail if No is too high
	abtem.config.set(scheduler="processes", num_workers=1)

	#Here we are deciding if cpu or gpu computing happens
	use_gpu = cfg.gpu_related.use_gpu
	if use_gpu:
		abtem.config.set({"device": "gpu", "fft": "cufft",'dask.lazy': True})
		abtem.config.set({"cupy.fft-cache-size" : cfg.gpu_related.cupy_fft_cache_size})
		abtem.config.set({"dask.chunk-size-gpu" : cfg.gpu_related.dask_chunk_size_gpu})
		import cupy as cp
	else:
		abtem.config.set({"device": "cpu", "fft": "fftw",'dask.lazy': True})	

	abtem.config.set({"dask.chunk-size" : cfg.gpu_related.dask_chunk_size})

	dask_client = None
	if use_gpu and cfg.gpu_related.dask_cuda:
		from dask.distributed import Client
		dask_client = Client("tcp://127.0.0.1:8786")
		from rmm.allocators.cupy import rmm_cupy_allocator
		cp.cuda.set_allocator(rmm_cupy_allocator)
	elif cfg.gpu_related.dask_cuda:
		print('dask_cuda can run only if CUDA is allowed; skipping')


	#Number of frozen phonons
	frozen_phonons = cfg.simulations.frozen_phonons
	if frozen_phonons == 'None':
		frozen_phonons = None

	fph_sigma = cfg.simulations.fph_sigma
	if isinstance(fph_sigma, bool):
		fph_sigma = None

	return RunContext(
		cfg=cfg,
		folder_sim=folder_sim,
		folder=folder,
		do_full_run=do_full_run,
		HT_value=HT_value,
		do_diffraction=do_diffraction,
		override_sampling=override_sampling,
		scan_start=scan_start,
		scan_stop=scan_stop,
		lamella_sizes=lamella_sizes,
		global_tilt=global_tilt,
		tilt_degrees=cfg.lamella_settings.tilt_degrees,
		convergence_angle=cfg.microscope.convergence_angle,
		cbed_max_angle=cfg.microscope.cbed_max_angle,
		haadf_detector=haadf_detector,
		abf_detector=abf_detector,
		bf_detector=bf_detector,
		element_to_remove=element_to_remove,
		probability_of_vac=probability_of_vac,
		add_vacancies_toggle=add_vacancies_toggle,
		frozen_phonons=frozen_phonons,
		fph_sigma=fph_sigma,
		phonons_seed=cfg.job.phonons_seed,
		dask_client=dask_client,
	)

BLUR_SIGMAS = [0.025, 0.1, 0.25]

def save_images(img, out_dir, prefix, sg, tilt, line_hkl, det_names):
	for w, iimg in enumerate(img):
		det_s = det_names[w]
		cpu = iimg.copy().to_cpu()
		cpu.to_tiff(str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}.tif"))
		cpu.to_zarr(str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}.zarr"), overwrite=True)
		for k in BLUR_SIGMAS:
			cpu.gaussian_filter(k, boundary='constant').to_tiff(
				str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}_{str(k).replace('.','-')}.tif"))

def make_potential(target):
	"""abtem.Potential with our standard params; returns lazy (caller chooses to build/compute)."""
	return abtem.Potential(
		target,
		sampling=0.05,   # real space sampling
		projection='infinite',
		parametrization='kirkland',
		periodic=False,
	)

def plot_diffraction(ctx, pot,fname,ftitle):
	fname = str(fname)
	
	initial_waves = abtem.PlaneWave(energy=ctx.HT_value,device='cpu')
	# Try CPU-side multislice first; fall back to pot's native device.
	try:
		exit_waves = initial_waves.multislice(pot.to_cpu()).compute()
	except Exception:
		exit_waves = initial_waves.multislice(pot).compute()
	print('Exit waves')
	diffraction_patterns = exit_waves.diffraction_patterns(max_angle="valid", block_direct=True).compute()
	if diffraction_patterns.ensemble_dims > 0:
		diffraction_patterns = diffraction_patterns.reduce_ensemble()
	diffraction_patterns.show(
		explode=False,power=0.2,units="mrad",
		figsize=(10, 6),cbar=True,common_color_scale=True,)
	fig = plt.gcf()

	#fig.tight_layout(rect=[0, 0, 1, 0.9])#
	fig.suptitle(ftitle, y=1.005)
	plt.savefig(fname,dpi=600)
	plt.close()
	
	diffraction_patterns.to_cpu().to_tiff(fname[:-4]+'.tif')

def plot_cbed(ctx, pot, fname, ftitle, position=None):
	fname = str(fname)
	probe = sim.add_probe(ctx, pot)

	if position is None:
		position = np.array([[
			0.5 * (ctx.scan_start[0] + ctx.scan_stop[0]),
			0.5 * (ctx.scan_start[1] + ctx.scan_stop[1]),
		]], dtype=float)
	else:
		position = np.array([position], dtype=float)

	# Try CPU-side multislice first; fall back to pot's native device.
	try:
		exit_waves = probe.multislice(pot.to_cpu(), scan=position).compute()
	except Exception:
		exit_waves = probe.multislice(pot, scan=position).compute()

	print("CBED exit wave")
	
	cbed = exit_waves.diffraction_patterns(
		max_angle=ctx.cbed_max_angle,
		block_direct=False
	)
	
	if cbed.ensemble_dims > 0:
		cbed = cbed.reduce_ensemble()

	cbed = cbed.compute().squeeze()
	
	cbed.show(
		power=0.2,
		units="mrad",
		figsize=(8, 6),
		cbar=True,
		common_color_scale=True,
	)
	fig = plt.gcf()
	fig.suptitle(ftitle, y=1.005)
	plt.savefig(fname, dpi=600)
	plt.close()
	
	cbed.to_cpu().to_tiff(fname[:-4]+'.tif')

def prepare_job(ctx, hkl_set,is_uvw=True,inplane_angle=None):
	'''
	This function prepares a set of ase objects to use for the further computations
	Inputs:
		hkl_set - list (Nx3), list of hkl (or uvw) vectors
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
	Output:
		full_dataset - list of dicts, one entry per (phase, hkl) combination
	'''
	
	cfg = ctx.cfg
	full_dataset = []

	borders = cfg.lamella_settings.borders
	tol = cfg.lamella_settings.tol
	max_uvw = cfg.lamella_settings.max_uvw
	sblock_size = cfg.lamella_settings.sblock_size
	atom_to_zero = cfg.lamella_settings.atom_to_zero
	extra_shift_z = cfg.lamella_settings.extra_shift_z

	for i in hkl_set.keys():
		for j in hkl_set[i]:
			print('Generating',i,j)
			cif_path = ctx.folder + cif_files[i]
			surf = sim.make_lamella(cif_path,j,sblock_size,ctx.lamella_sizes,atom_to_zero,tol,max_uvw,
						is_uvw=is_uvw,inplane_angle=inplane_angle,
						extra_shift_z=extra_shift_z,vac_xy=borders,vac_z=borders,
						global_tilt=ctx.global_tilt,tilt_degrees=ctx.tilt_degrees)

			if ctx.add_vacancies_toggle:
				surf = sim.add_vacancies(surf,ctx.element_to_remove,ctx.probability_of_vac)
				print('Vacancies applied to '+ctx.element_to_remove+ ', probability '+str(ctx.probability_of_vac))

			# Eager build+compute is fine for a single static lattice.
			potential = make_potential(surf).build().compute()
			# Keep lazy — eager build of the N-config ensemble would blow memory.
			frozen = abtem.FrozenPhonons(surf, num_configs=ctx.frozen_phonons, sigmas=ctx.fph_sigma, seed=ctx.phonons_seed)
			fph_potential = make_potential(frozen)
			full_dataset.append({
				'symm':i,
				'hkl':j,
				'surface':surf,
				'potential':potential,
				'fph_potential':fph_potential,
			})
	return full_dataset

def simulation_run(s,cfg,
	is_uvw=True,
	inplane_angle=None
	):
	'''
	Main starter function
	Inputs:
		s - list (Nx3), list of hkl (or uvw) vectors
		cfg - AppConfig, scalar-valued config for this run (one expand_cfg iteration)
		is_uvw - boolean, defines are vectors provided as hkl to use a normal to the corresponding plane, or as uvw
		inplane_angle - float | None, inplane slab rotation in degrees (if defined) prior crop
	'''
	ctx = resolve_context(cfg, global_tilt=None)
	
	#cp.cuda.Stream.null.synchronize()
	dataset = prepare_job(ctx,s,is_uvw,inplane_angle)
	for entry in dataset:
		out_dir = Path(ctx.folder_sim)
		sg = entry['symm']
		line_hkl = ''.join([str(q) for q in entry['hkl']])
		
		cfg_out_path = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}.toml"

		run_cfg = deepcopy(ctx.cfg.model_dump())
		run_cfg["lamella_settings"]["global_tilt_a"] = float(ctx.global_tilt[0])
		run_cfg["lamella_settings"]["global_tilt_b"] = float(ctx.global_tilt[1])
		with cfg_out_path.open("wb") as f:  # binary mode for tomli_w
			tomli_w.dump(run_cfg, f)

		sim.plot_dataset(entry,ctx,is_uvw)
		
		surf_fname = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_surf.xyz"
		ase.io.write(surf_fname, entry['surface'], 'xyz')
		print('xyz output created')

		
		if ctx.do_diffraction:
			print('Diffraction - single')
			ttl = sg+', [' + line_hkl +'], '+ str(ctx.lamella_sizes[0])+'x'+str(ctx.lamella_sizes[1])+'x'+str(ctx.lamella_sizes[2])+r'$\AA$'
			plot_diffraction(ctx,entry['potential'], out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_single_diff.png", ttl)
			plot_diffraction(ctx,entry['fph_potential'], out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_fph_diff.png", ttl+', '+str(ctx.frozen_phonons)+' fph')
			plot_cbed(ctx, entry['potential'],
				out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_center_cbed.png",
				ttl + ', center CBED' )
			plot_cbed(ctx, entry['fph_potential'],
				out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_center_fph_cbed.png",
				ttl + ', center CBED, ' + str(ctx.frozen_phonons) + ' fph'
				)
				
		if ctx.do_full_run:
			potential = entry['potential']
			probe = sim.add_probe(ctx,potential)
			probe.grid.match(potential)
			scan = sim.add_scan(ctx,probe,potential)

			measurements = probe.scan(potential, scan=scan, detectors=[ctx.haadf_detector,ctx.abf_detector,ctx.bf_detector])
			img = measurements.compute()

			save_images(img, out_dir, '', sg, ctx.global_tilt, line_hkl, ['haadf','abf','bf'])

			#frozen phonon set
			fph_potential = entry['fph_potential']
			probe.grid.match(potential)
			fph_measurements = probe.scan(fph_potential, scan=scan, detectors=[ctx.haadf_detector,ctx.abf_detector])
			img = fph_measurements.compute()

			save_images(img, out_dir, 'fph_', sg, ctx.global_tilt, line_hkl, ['haadf','abf'])
	del dataset

def main():
	# NOTE: the config filename, the {phase: [hkl]} mapping, is_uvw, and
	# inplane_angle below are all set manually here. The [job] section in
	# config.toml (cfg.job.phase / .hkl_to_do / .is_uvw) is parsed and
	# validated by pydantic but is NOT consumed at runtime — edit the
	# literals on the next lines to change what runs. TODO: wire cfg.job
	# through and drop these hardcoded values.
	cfg0 = confread.load_config("config.toml")
	for cfg_run in expand_cfg(cfg0):
		simulation_run({'PZO': [[1,1,0]]}, cfg_run,inplane_angle=0)

	print('Finished')


if __name__ == "__main__":
	main()
