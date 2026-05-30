#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

"""
Legacy in-process pipeline for abtem-run.

Preserved as a historical record of the pre-worker orchestration: one process
holds the dataset, runs the multislice for static + frozen-phonon potentials,
and produces matplotlib PNG previews alongside the TIFF/ZARR outputs. The
worker pipeline (generate -> per-seed worker -> aggregator) supersedes this
end-to-end, with the aggregator now emitting the equivalent PNG previews
(``potential_projection.png``, ``diff.png``, ``cbed.png``).

This file is not part of the installable package — it lives in ``legacy/``
alongside ``abtem_run_v01.py`` as a reference / fallback. Importing it
requires ``abtem-run`` to be installed (it pulls shared helpers — RunContext,
make_potential, add_probe, etc. — from ``abtem_run.pipeline`` / ``.simulation``).

CLI:
    python legacy/in_process.py     # reads ./config.toml, runs simulation_run

Library:
    from legacy.in_process import simulation_run
"""

from copy import deepcopy
from pathlib import Path

import abtem
import ase
import matplotlib.pyplot as plt
import numpy as np
import tomli_w

from abtem_run.config import load_config
from abtem_run.pipeline import (
	BLUR_SIGMAS,
	expand_cfg,
	make_potential,
	resolve_context,
)
from abtem_run.simulation import (
	add_probe,
	add_scan,
	add_vacancies,
	make_lamella,
)


def save_images(img, out_dir, prefix, sg, tilt, line_hkl, det_names):
	for w, iimg in enumerate(img):
		det_s = det_names[w]
		cpu = iimg.copy().to_cpu()
		# Q: do we need .mean(axis=0) here?
		cpu.to_tiff(str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}.tif"))
		cpu.to_zarr(str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}.zarr"), overwrite=True)
		for k in BLUR_SIGMAS:
			cpu.gaussian_filter(k, boundary='constant').to_tiff(
				str(out_dir / f"{prefix}{sg}_{tilt}_{line_hkl}_{det_s}_{str(k).replace('.','-')}.tif"))


def save_config(cfg, path):
	path = Path(path)
	with path.open("wb") as f:  # binary mode for tomli_w
		tomli_w.dump(cfg, f)


def plot_diffraction(ctx, pot, fname, ftitle):
	fname = str(fname)

	initial_waves = abtem.PlaneWave(energy=ctx.HT_value, device='cpu')
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
		explode=False, power=0.2, units="mrad",
		figsize=(10, 6), cbar=True, common_color_scale=True,)
	fig = plt.gcf()

	fig.suptitle(ftitle, y=1.005)
	plt.savefig(fname, dpi=600)
	plt.close()

	diffraction_patterns.to_cpu().to_tiff(fname[:-4] + '.tif')


def plot_cbed(ctx, pot, fname, ftitle, position=None):
	fname = str(fname)
	probe = add_probe(ctx, pot)

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

	cbed.to_cpu().to_tiff(fname[:-4] + '.tif')


def plot_dataset(data, ctx, is_uvw):
	'''
	This function plots a few previews of probe and pseudopotential
	Inputs
		data - dict, in the same format as in the __main__
		ctx - RunContext, supplies all microscope, scan, and path parameters
		is_uvw - boolean, reflects if the requested orientation vector is UVW (True) or HKL (False)
	'''

	out_dir = Path(ctx.folder_sim)
	sample_name = ctx.cfg.paths.sample_name
	global_tilt = ctx.global_tilt
	scan_s = ctx.cfg.lamella_settings.scan_s
	borders = ctx.cfg.lamella_settings.borders

	surf = data['surface']
	sg = data['symm']
	potential = data['potential']
	fph_potential = data['fph_potential']

	probe = add_probe(ctx, potential)
	fph_probe = add_probe(ctx, fph_potential)
	scan = add_scan(ctx, probe, potential)

	line_hkl = ''.join([str(q) for q in data['hkl']])
	if is_uvw:
		str_hkl = 'uvw [' + line_hkl + ']'
	else:
		str_hkl = 'hkl [' + line_hkl + ']'

	proj_cpu = potential.project().to_cpu().compute()

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

	proj_cpu.show(
		cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
	)
	#probe.build()
	probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(sample_name + ', ' + sg + ', ' + str_hkl, fontsize=18)
	fig.tight_layout()
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_potential.png"), dpi=600)
	plt.close()

	proj_cpu.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_potential.tif"))
	proj_cropped = proj_cpu.crop([scan_s, scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_scanned_potential.tif"))

	fph_proj_cpu = fph_potential.project().to_cpu()
	fph_proj_mean = fph_proj_cpu.mean(axis=0).compute()

	#TODO this plot is optional
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

	fph_proj_mean.show(
		cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
	)

	fph_probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(sample_name + ', ' + sg + ', ' + str_hkl, fontsize=18)
	fig.tight_layout()
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_fph_potential.png"), dpi=600)
	plt.close()

	#This one is the most important - it draws 3 projections of a final block
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
	abtem.show_atoms(surf, ax=ax1, title="XY projection")  #, scans=scan)
	scan.add_to_plot(ax1)
	abtem.show_atoms(surf, ax=ax2, title="Cross-section", plane='xz')
	abtem.show_atoms(surf, ax=ax3, title="Cross-section", plane='yz')

	fig.suptitle(sample_name + ', ' + sg + ', ' + str_hkl, fontsize=18)
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_combined.png"), dpi=600)
	plt.close()
	print('Checkpoint')
	fph_proj_mean.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_fph_potential.tif"))

	proj_cropped = fph_proj_mean.crop([scan_s, scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_scanned_fph_potential.tif"))


def prepare_job(ctx, hkl_set, is_uvw=True, inplane_angle=None):
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

	# hkl_set is {<cif_filename>: [[h,k,l], ...]}. The key is the CIF filename
	# (appended to ctx.folder); its stem (minus .cif) is the 'symm' output label.
	for cif_filename, hkls in hkl_set.items():
		stem = cif_filename[:-4] if cif_filename.lower().endswith('.cif') else cif_filename
		for j in hkls:
			print('Generating', cif_filename, j)
			cif_path = ctx.folder + cif_filename
			surf = make_lamella(cif_path, j, sblock_size, ctx.lamella_sizes, atom_to_zero, tol, max_uvw,
					is_uvw=is_uvw, inplane_angle=inplane_angle,
					extra_shift_z=extra_shift_z, vac_xy=borders, vac_z=borders,
					global_tilt=ctx.global_tilt, tilt_degrees=ctx.tilt_degrees)

			if ctx.add_vacancies_toggle:
				surf = add_vacancies(surf, ctx.element_to_remove, ctx.probability_of_vac, seed=ctx.vacancies_seed)
				print('Vacancies applied to ' + ctx.element_to_remove + ', probability ' + str(ctx.probability_of_vac) + ', seed ' + str(ctx.vacancies_seed))

			# Eager build+compute is fine for a single static lattice.
			potential = make_potential(surf).build().compute()
			# Keep lazy — eager build of the N-config ensemble would blow memory.
			frozen = abtem.FrozenPhonons(surf, num_configs=ctx.frozen_phonons, sigmas=ctx.fph_sigma, seed=ctx.phonons_seed)
			fph_potential = make_potential(frozen)
			full_dataset.append({
				'symm': stem,
				'hkl': j,
				'surface': surf,
				'potential': potential,
				'fph_potential': fph_potential,
			})
	return full_dataset


def simulation_run(s, cfg,
		is_uvw=True,
		inplane_angle=None,
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
	dataset = prepare_job(ctx, s, is_uvw, inplane_angle)
	for entry in dataset:
		out_dir = Path(ctx.folder_sim)
		sg = entry['symm']
		line_hkl = ''.join([str(q) for q in entry['hkl']])

		cfg_out_path = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}.toml"

		run_cfg = deepcopy(ctx.cfg.model_dump())
		run_cfg["lamella_settings"]["global_tilt_a"] = float(ctx.global_tilt[0])
		run_cfg["lamella_settings"]["global_tilt_b"] = float(ctx.global_tilt[1])
		save_config(run_cfg, cfg_out_path)

		plot_dataset(entry, ctx, is_uvw)

		surf_fname = out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_surf.xyz"
		ase.io.write(surf_fname, entry['surface'], 'xyz')
		print('xyz output created')

		ttl = sg + ', [' + line_hkl + '], ' + str(ctx.lamella_sizes[0]) + 'x' + str(ctx.lamella_sizes[1]) + 'x' + str(ctx.lamella_sizes[2]) + r'$\AA$'
		if ctx.do_diffraction:
			print('Diffraction - single')
			plot_diffraction(ctx, entry['potential'], out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_single_diff.png", ttl)
			plot_diffraction(ctx, entry['fph_potential'], out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_fph_diff.png", ttl + ', ' + str(ctx.frozen_phonons) + ' fph')

		if ctx.do_cbed:
			print('CBED - center')
			plot_cbed(ctx, entry['potential'],
				out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_center_cbed.png",
				ttl + ', center CBED')
			plot_cbed(ctx, entry['fph_potential'],
				out_dir / f"{sg}_{line_hkl}_{ctx.global_tilt}_center_fph_cbed.png",
				ttl + ', center CBED, ' + str(ctx.frozen_phonons) + ' fph'
				)

		if ctx.do_full_run:
			potential = entry['potential']
			probe = add_probe(ctx, potential)
			probe.grid.match(potential)
			scan = add_scan(ctx, probe, potential)

			measurements = probe.scan(potential, scan=scan, detectors=[ctx.haadf_detector, ctx.abf_detector, ctx.bf_detector])
			img = measurements.compute()

			save_images(img, out_dir, '', sg, ctx.global_tilt, line_hkl, ['haadf', 'abf', 'bf'])

			#frozen phonon set
			fph_potential = entry['fph_potential']
			# !TODO - validate the approach on the probe.grid.match here
			probe.grid.match(potential)
			fph_measurements = probe.scan(fph_potential, scan=scan, detectors=[ctx.haadf_detector, ctx.abf_detector])
			img = fph_measurements.compute()

			save_images(img, out_dir, 'fph_', sg, ctx.global_tilt, line_hkl, ['haadf', 'abf'])
	del dataset


def main():
	# Everything that controls *what* runs comes from [job]: phase (CIF
	# filename), hkl_to_do, is_uvw, inplane_angle. Sweeps over physical
	# parameters (frozen_phonons, fph_sigma, tilt, ...) come from expand_cfg.
	cfg0 = load_config("config.toml")
	for cfg_run in expand_cfg(cfg0):
		hkl_set = {cfg_run.job.phase: cfg_run.job.hkl_list}
		simulation_run(
			hkl_set,
			cfg_run,
			is_uvw=cfg_run.job.is_uvw,
			inplane_angle=cfg_run.job.inplane_angle_resolved,
		)

	print('Finished')


if __name__ == "__main__":
	main()
