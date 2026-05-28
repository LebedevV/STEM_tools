#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

from pathlib import Path
import ase
import numpy as np
import diffpy.structure
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import dask.array as da
import dask
import abtem

def get_params(cif_path):
	'''
	Reads cif file from the path given
	cif_path - str, existing path to the valid cif file
	Output - tuple of lattice params, (a,b,s,alpha,beta,gamma) as in cif
	'''
	c = ase.io.read(cif_path)
	par = c.cell.lengths()
	ang = c.cell.angles()

	return par[0],par[1],par[2],ang[0],ang[1],ang[2]

def get_supercell(cif_path,sblock_size):
	'''
	Here we are expanding the unit cell to fit as close to the requested sblock size as we can in ints
	Then the superblock shifted by xyz to bring its center to 0
	
	Inputs:
	cif_path - str, existing path to the valid cif file
	sblock_size - int or float, size of the superblock cube
	
	Output - ase object
	'''

	c = ase.io.read(cif_path)	
	#par = ase.geometry.cell_to_cellpar(c.cell, radians=False)[:3]
	par = get_params(cif_path)[:3]
	multiplier = [ int(sblock_size/i) if int(sblock_size/i) >=1 else 1 for i in par  ]
	c = c*multiplier
	
	#Let's bring center to 1/2 of the volume
	
	#c.translate(-np.array(multiplier,dtype=int)/2*par)#rounded up
	c.translate(-ase.geometry.cell_to_cellpar(c.cell, radians=False)[:3]/3)#precise center

	return c

def hkl_to_uvw(param_list,hkl,max_uvw,around=True):
	'''
	Converts hkl to uvw with respect to the lattice parameters

	around - boolean, defines if we need uvw as ints
			Extremely important parameter!
				if True, structure is aligned by the nearest uvw
				if False, by the normal to hkl
	param_list - (a,b,c,alpha,beta,gamma)
	hkl - tuple of 3 ints is expected
	max_uvw - upper threshold for the equivalent uvw multiplier
	
	out - tuple of 3 ints if uvw, of 3 floats if normal to hkl
	'''
	print('Given HKL ',hkl)

	#First, create lattice and its reciprocal version
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	lat_r = lat.reciprocal()

	#Convert hkl vector to the real space
	vs = lat_r.cartesian(hkl)
	out = lat.fractional(vs)

	out = np.array(out)

	#Renorm, with respect to zeros
	u = out[abs(out) > 0.0001]
	out = out/min(abs(u))

	#Try to find a multiplier, within a given margins
	m = find_multiplier(out,max_uvw)  
	out = out*m

	#Round up if needed
	if around:
		out = out.round()
		out = out.astype(int)
	out = out.tolist()
	print('Proposed UVW ',out)

	return(out)

def find_multiplier(frac,max_uvw):
	'''
	Ugly way to find the best multiplier with respect to the upper threshold
	frac - uvw vector
	max_uvw - int, threshold
	'''
	multipliers = np.arange(1,max_uvw)
	m = 1
	found = False
	fl = True
	for i in multipliers:
		res = frac*i - np.round(frac*i)
		#if there is a way to get ints, there is no point to search further
		if np.all(abs(res) < 0.0001):
			print('Ideal multiplier ',i)
			m = i
			found = True
			break
		#if somehow reasonable multiplier found, we'd better keep it,
		#but continue with attempts to find an ideal one
		if np.all(abs(res) < 0.1) and fl:
			print('Non-ideal multiplier ',i)
			m = i
			found = True
			fl = False

	if not found:
		print(f'WARNING: no multiplier within max_uvw={max_uvw} brings {frac} close to integers; falling back to m=1')

	return m





#Here the rotation magic happens
def get_euler_uvw(param_list,uvw):
	'''
	This function finds a rotation matrix required to align a given [uvw] with Z
	There were a plenty of issues while I was trying to directly align these vectors,
		so here it is done step by step, one rotation after another
	
	param_list - (a,b,c,alpha,beta,gamma)
	uvw - vector
	
	returns rotation object as in scipy.spatial.transform
	'''
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
							  
	#Fractional coordinates of the real-space uvw and c vectors
	vv = lat.cartesian(uvw)
	vc = lat.cartesian([0,0,1])
	print('Check uvw',uvw)
	
	#Fractional coordinates of the real-space a,b,c vectors
	av,bv,cv = lat.cartesian([1,0,0]),lat.cartesian([0,1,0]),lat.cartesian([0,0,1])

	#Another way to get angles between a,b,c and x,y,z
	print('Sanity check')
	sal,sbt,sgm = np.linalg.norm(np.cross(av,[1,0,0])),np.linalg.norm(np.cross(bv,[0,1,0])),np.linalg.norm(np.cross(cv,[0,0,1]))
	print('Angles to axes',sal,sbt,sgm)
	
	#First rotation - bring a to OX by rotation around Z
	AtoX = R.from_matrix(np.eye(3))
	if sal != 0:
		AtoX = R.from_euler('z',-param_list[5]+90,degrees=True)
		print('Around z by',-param_list[5]+90)
	print(vv,vc)

	#Check
	an_c = lat.angle(uvw,[0,0,1])
	print('Angle uvw to c',an_c)

	#Warning for trigonal systems
	if param_list[3]-90 !=0:
		print('Careful! might be an issue there; this angle was not tested properly')
	
	#Second rotation, only for trigonal - around OX (and a), to bring c to XZ plane 
	CtoZ_bc = R.from_euler('x',param_list[3]-90,degrees=True) #angle between c and z in bc plane
	print('Around x by',param_list[3]-90)
	
	#Third rotation, around OY, to bring c to Z within XZ plane
	CtoZ_XZ = R.from_euler('y',param_list[4]-90,degrees=True) #angle between c and z in XZ plane
	print('Around y by',param_list[4]-90)
	
	#Combine 2nd and 3rd
	CtoZ = CtoZ_bc*CtoZ_XZ
	
	#Fourth rotation - align uvw and c
	rot_v = np.cross(vv,vc)
	if np.linalg.norm(rot_v) != 0:
		rot_v = rot_v/np.linalg.norm(rot_v)
		rot_v = rot_v*an_c
		print('Around axis',rot_v,'by',np.linalg.norm(rot_v))
		VtoC = R.from_rotvec(rot_v,degrees=True)
		rot = VtoC*CtoZ
	else:
		#special case of collinear v and c. we just need to find a sign then
		flip = R.from_matrix(np.eye(3)*(np.dot(vc,vv)/abs(np.dot(vc,vv))))
		rot = flip*CtoZ

	#Cumulative rotation
	rot = rot*AtoX
	
	print(np.round(rot.as_matrix(),2))
	return rot


def make_lamella(cif_path,hkl,sblock_size,lamella_sizes,atom_to_zero,tol,max_uvw,is_uvw=True,
			inplane_angle=None,extra_shift_z=0,vac_xy=0,vac_z=0,global_tilt=(0,0),tilt_degrees=True):
	'''
	High-level function; for a given crystal structure, generates the rectangular set of atoms - 'lamella'
		in such a way that the requested uvw is directed upwards
	Input:
		cif_path - str, existing path to the valid cif file
		hkl - tuple of three ints; desired orientation vector (uvw or normal to hkl)
		sblock_size - int or float, size of the superblock cube which later will be rotated and cropped
		lamella_sizes - tuple of 3 ints, XxYxZ sizes of the proposed lamella in Angstroms
		atom_to_zero - str, label of atom to be set to the point of origin after the rotation completed
			!NB not to the corner of the virtual scan; there is a gap
		tol - float, tolerance for atoms on surfaces and near zero, in A
		max_uvw - int, max value of the multiplier for hkl to uvw conversion
		
		is_uvw - boolean, defines if we provided hkl or uvw vector
		
		inplane_angle - float, extra rotation in XY plane, degrees
		extra_shift_z - float, shifts the superblock along Z before cropping
		vac_xy - float, gaps of empty space around the final slab, in A
		vac_z - float, empty space above and below the slab, in A
	output - ase object
	'''
	
	#Obtain rotation matrix for hkl/uvw and the structure given
	param_list = get_params(cif_path)
	if is_uvw:
		uvw = hkl
	else:
		uvw = hkl_to_uvw(param_list,hkl,max_uvw,around=False)
	# !TODO validation of directions: how [uvw] here relates to the abTEM beam settings
	rot = get_euler_uvw(param_list,uvw)
	rot_matrix = rot.as_matrix()
	
	#Create supercell
	sup = get_supercell(cif_path,sblock_size)
	da_atoms = da.from_array(sup.get_positions(), chunks=(100000, 3))
	da_elements = da.from_array(sup.get_chemical_symbols(),chunks=100000)
	del sup
	
	print('There are ',len(da_atoms),' atoms in the supercell')
	
	#Here we are rotating x,y,z set
	new_coords =  (da_atoms @ rot_matrix.T).rechunk({1:3})

	print('Rotated')

	ftol = 0.00001
	#lets select a relatively small test subset of atoms to:
	#	- find the atom of interest nearest to (0,0,0) - say, atom0
	#	- find the angle between OX and vector from the atom0 to the nearest atom of the same type
	box = max(param_list[:3])
	box = max(box,10)
	box = da.ones(3)*box
	mask = da.all(new_coords > -box - ftol, axis=1) & da.all(
						new_coords < box + ftol, axis=1 )
	print('Mask created')
	test_c = new_coords[mask]
	chem = da_elements[mask]

	print('Mask applied')
	
	#Here I wish to find an atom of interest nearby 0 and bring it to 0... on the subset of +-abc
	fin_selected = None
	if atom_to_zero is not None:
		mask_chem = da.isin(chem, atom_to_zero)
		el_check = mask_chem.any().compute()
		if el_check:
			ref_atoms = test_c[mask_chem].compute()

			dist = ase.geometry.get_distances((0,0,0), p2=ref_atoms )[1][0]

			new_zero = [ i for i,j in zip(ref_atoms,dist) if (j > min(dist) - tol) and ( j < min(dist) + tol ) ][0]

			print('Zero moved to',new_zero)
			new_coords -= new_zero
			ref_atoms -= new_zero

			#Lets find atoms of the same type, located nearby XY plane
			ref_atoms_xy = [ (x,y,z) for (x,y,z) in ref_atoms if (abs(z) < 5) and (abs(x) > 0.1) and (abs(y) > 0.1) ]
			proj_XY = np.array([ (x,y,0) for (x,y,z) in ref_atoms_xy ])

			#Here we are measuring the angle towards the nearest atom of the same type
			if len(proj_XY) > 1:
				dist = ase.geometry.get_distances((0,0,0), p2=proj_XY )[1][0]

				min_r_dist = min(dist[dist>=0.25])
				print('min_r_dist',min_r_dist)

				selected = np.atleast_2d(proj_XY[dist<min_r_dist*1.025])
				print('selected',selected)
				# Q: ensure we select the nearest one among them
				if len(selected) > 1:
					upper_half = [i for i in selected if i[1] > 0]
					if upper_half:
						angles = [np.arccos(np.clip(i[0]/np.linalg.norm(i), -1, 1)) for i in upper_half]
						fin_selected = upper_half[int(np.argmin(angles))]
					else:
						fin_selected = selected[0]
				else:
					print('No pref given')
					fin_selected = selected[0]
			elif len(proj_XY) == 1:
				fin_selected = proj_XY[0]
			else:
				print('No in-plane reference atoms found for',atom_to_zero,'; in-plane auto-rotation skipped')

			if fin_selected is not None:
				print('Proposed rotation towards',fin_selected)
		else:
			print('Proposed atom for (0,0,0) is not found; skip')
	#Extra shift by z applied here
	new_coords -= (0,0,-extra_shift_z)

	print('Slab translated')
	
	if inplane_angle is None and fin_selected is not None:
		rot_angle = np.arccos(np.dot(fin_selected,[1,0,0])/np.linalg.norm(fin_selected))/np.pi*180
		print('Proposed in-plane rotation',rot_angle)
	else:
		rot_angle = inplane_angle if inplane_angle is not None else 0.
		print('Requested in-plane rotation',rot_angle)
	print('in-plane rotation',rot_angle)
	###Here we are rotating the full set of coordinates (x,y,z)
	rot_matrix = R.from_euler('z',-rot_angle,degrees=True).as_matrix()
	new_coords = new_coords @ rot_matrix.T

	print(np.round(rot_matrix,5))

	###Here we are cropping the lamella, from 0 to lims
	margin = np.ones(3)*tol
	upper = np.asarray(lamella_sizes, dtype=float) + margin
	mask_fin = (da.all(new_coords > -margin, axis=1) & da.all(new_coords < upper, axis=1)).astype(bool)
	mask_fin = mask_fin.rechunk({0: "auto"})
	da_elements = da_elements.rechunk({0: mask_fin.chunks[0]})
	
	cropped = new_coords[mask_fin] + (vac_xy,vac_xy,vac_z)
	
	if not tilt_degrees:
		tilt = np.array(global_tilt)/1000
		print('tilt in mrad',tilt)
	else:
		tilt = global_tilt
	
	rot_matrix_x = R.from_euler('x',tilt[0],degrees=tilt_degrees).as_matrix()
	cropped = cropped @ rot_matrix_x.T
	rot_matrix_y = R.from_euler('y',tilt[1],degrees=tilt_degrees).as_matrix()
	cropped = cropped @ rot_matrix_y.T

	print(rot_matrix_x.T,rot_matrix_y.T)
	cropped_chem = da_elements[mask_fin]
	cropped,cropped_chem = dask.compute(cropped,cropped_chem)

	print('Atoms in the lamella',len(cropped))

	#TODO ensure tilt doesnt shift it out of range
	cell_size = (lamella_sizes[0]+2*vac_xy,lamella_sizes[1]+2*vac_xy,
			lamella_sizes[2]+2*vac_z,
			90,90,90)

	fin_cell = ase.Atoms(cropped_chem, cropped, cell=np.asarray(cell_size, float), pbc=False)

	return fin_cell
	

def add_probe(ctx, potential, defocus="scherzer"):
	probe = abtem.Probe(
		energy=ctx.HT_value,
		semiangle_cutoff=ctx.convergence_angle,
		defocus=defocus
	)
	probe.grid.match(potential)
	return probe

def add_scan(ctx, probe, pot):
	default_sampling = probe.ctf.nyquist_sampling * .9
	print('Proposed sampling', default_sampling)
	if not ctx.override_sampling:
		sampling = default_sampling
	else:
		sampling = ctx.override_sampling
		print('Overrided sampling', sampling)
	return abtem.scan.GridScan(
		start=ctx.scan_start,
		end=ctx.scan_stop,
		sampling=sampling,
		potential=pot
	)

def add_vacancies(surf,el,prob,seed=0):
	'''
	This function removes atoms of a certain type from a surf object with a given probability
	Inputs:
		surf - ase surface
		el - str, element name to remove
		prob - float, (0,1], probability of atom to disappear
		seed - int, RNG seed; same seed + surf + (el,prob) -> same vacancy pattern
	Output:
		cropped - ase surface
	'''

	at_types = np.array(surf.get_chemical_symbols())
	rng = np.random.default_rng(seed)
	#We select atom type and marking those to be removed
	mask = (at_types == el) & (rng.random(len(at_types)) < prob)
	return surf[~mask]

#Previews plot
def plot_dataset(data,ctx,is_uvw):
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
		str_hkl = 'uvw ['+line_hkl+']'
	else:
		str_hkl = 'hkl ['+line_hkl+']'

	proj_cpu = potential.project().to_cpu().compute()

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
	
	proj_cpu.show(
		cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
	)
	#probe.build()
	probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
	fig.tight_layout()
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_potential.png"),dpi=600)
	plt.close()

	proj_cpu.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_potential.tif"))
	proj_cropped = proj_cpu.crop( [scan_s,scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_scanned_potential.tif"))
		
	fph_proj_cpu = fph_potential.project().to_cpu()
	fph_proj_mean = fph_proj_cpu.mean(axis=0).compute()
	
	#TODO this plot is optional
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
		
	fph_proj_mean.show(
		cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
	)

	fph_probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
	fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
	fig.tight_layout()
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_fph_potential.png"),dpi=600)
	plt.close()
	
	#This one is the most important - it draws 3 projections of a final block
	fig,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))
	abtem.show_atoms(surf, ax=ax1, title="XY projection" )#, scans=scan)
	scan.add_to_plot(ax1)
	abtem.show_atoms(surf, ax=ax2, title="Cross-section", plane='xz')
	abtem.show_atoms(surf, ax=ax3, title="Cross-section", plane='yz')

	fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
	fig.savefig(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_combined.png"),dpi=600)
	plt.close()
	print('Checkpoint')
	fph_proj_mean.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_fph_potential.tif"))

	proj_cropped = fph_proj_mean.crop( [scan_s,scan_s], offset=(borders, borders))
	proj_cropped.to_tiff(str(out_dir / f"{sg}_{line_hkl}_{global_tilt}_scanned_fph_potential.tif"))
