#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import logging
from copy import deepcopy
from dataclasses import dataclass
from itertools import product

import abtem

# Importing the package runs abtem monkey-patches in __init__.py before any
# abtem calls happen in this module.
from .config import AppConfig


log = logging.getLogger(__name__)


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
	do_cbed: bool
	detectors: list[str]
	test_enabled: bool
	blur_boundary: str
	blur_sigmas: list[float]
	emit_static_baseline: bool
	override_sampling: float | bool

	# resolved lamella geometry
	scan_start: tuple[float, float]
	scan_stop: tuple[float, float]
	lamella_sizes: tuple[float, float, float]
	global_tilt: tuple[float, float]
	tilt_degrees: bool

	convergence_angle: float
	cbed_max_angle: float | str
	defocus: float | str
	aberrations: dict

	# resolved detectors (abtem objects)
	haadf_detector: object
	abf_detector: object
	bf_detector: object

	element_to_remove: str
	probability_of_vac: float
	add_vacancies_toggle: bool
	vacancies_seed: int

	# frozen phonons settings
	frozen_phonons: int | None
	fph_sigma: float | None
	phonons_seed: int

	# dask distributed client — held here so it is not GC'd before the run ends
	dask_client: object | None


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
	do_cbed = cfg.microscope.do_cbed
	detectors = cfg.microscope.detectors
	test_enabled = cfg.simulations.test_enabled
	blur_boundary = cfg.simulations.blur_boundary
	blur_sigmas = list(cfg.simulations.blur_sigmas)
	emit_static_baseline = cfg.simulations.emit_static_baseline
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
	# !TODO - separate dask.distributed and dask_cuda
	if use_gpu and cfg.gpu_related.dask_cuda:
		from dask.distributed import Client
		dask_client = Client("tcp://127.0.0.1:8786")
		from rmm.allocators.cupy import rmm_cupy_allocator
		cp.cuda.set_allocator(rmm_cupy_allocator)
	elif cfg.gpu_related.dask_cuda:
		log.info('dask_cuda can run only if CUDA is allowed; skipping')


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
		do_cbed=do_cbed,
		detectors=detectors,
		test_enabled=test_enabled,
		blur_boundary=blur_boundary,
		blur_sigmas=blur_sigmas,
		emit_static_baseline=emit_static_baseline,
		override_sampling=override_sampling,
		scan_start=scan_start,
		scan_stop=scan_stop,
		lamella_sizes=lamella_sizes,
		global_tilt=global_tilt,
		tilt_degrees=cfg.lamella_settings.tilt_degrees,
		convergence_angle=cfg.microscope.convergence_angle,
		cbed_max_angle=cfg.microscope.cbed_max_angle,
		defocus=cfg.microscope.defocus,
		aberrations=dict(cfg.microscope.aberrations),
		haadf_detector=haadf_detector,
		abf_detector=abf_detector,
		bf_detector=bf_detector,
		element_to_remove=element_to_remove,
		probability_of_vac=probability_of_vac,
		add_vacancies_toggle=add_vacancies_toggle,
		vacancies_seed=cfg.lamella_settings.vacancies_seed,
		frozen_phonons=frozen_phonons,
		fph_sigma=fph_sigma,
		phonons_seed=cfg.job.phonons_seed,
		dask_client=dask_client,
	)


def make_potential(target):
	"""abtem.Potential with our standard params; returns lazy (caller chooses to build/compute)."""
	return abtem.Potential(
		target,
		sampling=0.05,   # real space sampling
		projection='infinite',
		parametrization='kirkland',
		periodic=False,
	)
