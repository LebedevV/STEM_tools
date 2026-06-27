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
	# Derived scan geometry.
	scan_start: tuple[float, float]
	scan_stop: tuple[float, float]

	# Resolved detector objects.
	haadf_detector: object
	abf_detector: object
	bf_detector: object

	# Frozen-phonon displacement sigma; False -> None.
	fph_sigma: float | None

	# Keep the dask client alive for the run.
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

def resolve_context(cfg):
	borders = cfg.lamella_settings.borders
	scan_s = cfg.lamella_settings.scan_s
	scan_start = (borders * 2, borders * 2)
	scan_stop = (borders * 2 + scan_s, borders * 2 + scan_s)

	haadf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.haadfinner, outer=cfg.microscope.haadfouter
	)
	abf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.abfinner, outer=cfg.microscope.abfouter
	)
	bf_detector = abtem.AnnularDetector(
		inner=cfg.microscope.bfinner, outer=cfg.microscope.bfouter
	)

	abtem.config.set(scheduler="processes", num_workers=1)

	use_gpu = cfg.gpu_related.use_gpu
	if use_gpu:
		abtem.config.set({"device": "gpu", "fft": "cufft", 'dask.lazy': True})
		abtem.config.set({"cupy.fft-cache-size" : cfg.gpu_related.cupy_fft_cache_size})
		abtem.config.set({"dask.chunk-size-gpu" : cfg.gpu_related.dask_chunk_size_gpu})
		import cupy as cp
	else:
		abtem.config.set({"device": "cpu", "fft": "fftw", 'dask.lazy': True})

	abtem.config.set({"dask.chunk-size" : cfg.gpu_related.dask_chunk_size})

	dask_client = None
	if use_gpu and cfg.gpu_related.dask_cuda:
		from dask.distributed import Client
		dask_client = Client("tcp://127.0.0.1:8786")
		try:
			from rmm.allocators.cupy import rmm_cupy_allocator
			cp.cuda.set_allocator(rmm_cupy_allocator)
		except ImportError:
			log.info("rmm not available; using cupy's default memory pool")
	elif cfg.gpu_related.dask_cuda:
		log.info('dask_cuda can run only if CUDA is allowed; skipping')

	fph_sigma = cfg.simulations.fph_sigma
	if isinstance(fph_sigma, bool):
		fph_sigma = None

	return RunContext(
		scan_start=scan_start,
		scan_stop=scan_stop,
		haadf_detector=haadf_detector,
		abf_detector=abf_detector,
		bf_detector=bf_detector,
		fph_sigma=fph_sigma,
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
