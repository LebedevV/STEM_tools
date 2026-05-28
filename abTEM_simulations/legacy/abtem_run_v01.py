#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np
from scipy.spatial.transform import Rotation as R

from ase.visualize import view
from ase.cell import Cell
import abtem.atoms as ab_atoms
import diffpy.structure

import ase, abtem, py4DSTEM
import numpy as np
import matplotlib.pyplot as plt

import tifffile, scipy, numpy.linalg
import datetime

from simulation_lib import *

#General output path
folder_sim = '/home/vasily/test_abTEM/'
#folder_sim = 'C://Users//lebedevv//Desktop//sym//'

#Specific output path - please, make sure that path exists!
extr='output_test/'
sample_name = 'Sample'
#Path to cif files
folder = '/home/vasily/test_abTEM/'
#folder = 'C://Users//lebedevv//Desktop//cifs//'

folder_sim += extr

#Here we are deciding if cpu or gpu computing happens
abtem.config.set({"device": "gpu", "fft": "fftw",'dask.lazy': True})
#abtem.config.set({"device": "cpu", "fft": "fftw",'dask.lazy': True})

#No of threads; limited by video-memory, can fail if No is too high
num_workers = 2



####################
#Main variables
####################

#Number of frozen phonons
frozen_phonons = 3
#frozen_phonons = 10

#Max int to search for uvw
#needed for hkl to uvw conversion
max_uvw = 20

##############
#Lamella-related
#Size of a superblock
sblock_size = 500 #A
#!!!! there is no guaranty that the frame will be 100% filled after the rotation and crop !!!
#better to double-check, and to keep sblock size sufficiently large

#Scan size
scan_s = 10 #A
#scan_s = 50 #A

#Extra gaps, framing the sacn in XY directions
borders = 5 #A
#borders = 10 #A

#Thickness of the virtual lamella
thickness = 10 #A
#thickness = 30 #A

extra_shift_z = 0#-10 #A

#override XY sampling
#override_sampling = .5
override_sampling = False

#Spatial tolerance - in use for lamella margins, and for atoms search near (0,0,0) point
tol = .3 #tolerance

#Tilt
global_tilt = (0,0)#now - to tilt the lamella, not the beam; (around x, deg; around y, deg)

scan_start = (borders,borders)
scan_stop = (borders+scan_s,borders+scan_s)
lamella_sizes = (borders*2+scan_s,borders*2+scan_s,thickness)

#Please, check in case of issues; might fail if atom is not found
##############
#Which atom we would like to shift to (0,0,0) after rotation
#atom_to_zero = 'Ti'
#atom_to_zero = 'O'
#atom_to_zero = 'Zr'
atom_to_zero = 'Pb'
##############


#########
#detectors settings

#Nion, 200kV
haadfinner = 99  # mrad
haadfouter = 200  # mrad
haadf_detector = abtem.AnnularDetector(inner=haadfinner, outer=haadfouter)

#30-63?
abf_detector = abtem.AnnularDetector(inner=15, outer=33)

#0-9.3?
bf_detector = abtem.AnnularDetector(inner=0, outer=5)

#Short names for cif files
cif_files = {
		'Pbam':'2003071.cif'
}

def add_potential(surf):
	# Make the potential
	potential = abtem.Potential(
	surf,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	).build()

	return potential

def add_frozen_phonons_potential(surf):
	frozen = abtem.FrozenPhonons(surf, num_configs=frozen_phonons, sigmas=0.1, seed=100)
	# Make the potential
	potential = abtem.Potential(
	frozen,
	sampling=0.05,   # real space sampling
	projection='infinite',
	parametrization='kirkland',
	) 
	
	return potential

def add_probe(potential):
	# Make the probe
	tilt = (0,0)#(global_tilt[0]/180*np.pi*1000,global_tilt[1]/180*np.pi*1000)
	probe = abtem.Probe(
	energy=200e3, 
	semiangle_cutoff=30, 
	defocus="scherzer"
	)
	#,
	#tilt=tilt
	# Match probe grid to potential grid
	probe.grid.match(potential)
	print('tilt',global_tilt)
	return probe

def add_scan(probe,pot):
	# Define scan area
	print('Proposed sampling',probe.ctf.nyquist_sampling * .9)
	if not override_sampling:
		sampling = probe.ctf.nyquist_sampling * .9
	else:
		sampling = override_sampling
		print('Overrided sampling',sampling)

	scan = abtem.scan.GridScan(
		start = scan_start, 
		end = scan_stop, 
		sampling = sampling,
		potential=pot
	)
	
	return scan   

def prepare_job(hkl_set,is_uvw=True,inplane_angle=None):
	full_dataset = {}
	count = 0
	for i in hkl_set.keys():
		for j in hkl_set[i]:
			print('Generating',i,j)
			#scell = struct_set[i]
			cif_path = folder + cif_files[i]
			surf = make_lamella(cif_path,j,sblock_size,lamella_sizes,atom_to_zero,tol,max_uvw,
						is_uvw=is_uvw,inplane_angle=inplane_angle,extra_shift_z=extra_shift_z)
			potential = add_potential(surf)
			#potential.compute()
			fph_potential = add_frozen_phonons_potential(surf)
			#fph_potential.compute()
			probe = add_probe(potential)
			fph_probe = add_probe(fph_potential)
			data = {
				'symm':i,
				'hkl':j,
				'surface':surf,
				'potential':potential,
				'probe':probe,
				'fph_potential':fph_potential,
				'fph_probe':fph_probe,
			}
			full_dataset[count] = data
			count+=1
	return full_dataset   

def plot_dataset(data,is_uvw):
		#fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(10,5))
		surf = data['surface']
		probe = data['probe']
		fph_probe = data['fph_probe']
		#print(probe)
		scan = add_scan(probe,surf)
		sg = data['symm']
		potential = data['potential']
		fph_potential = data['fph_potential']

		line_hkl = ''.join([str(q) for q in data['hkl']])
		if is_uvw:
			str_hkl = 'uvw ['+line_hkl+']'
		else:
			str_hkl = 'hkl ['+line_hkl+']'

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
		abtem.show_atoms(surf, ax=ax1, title="XY projection" )#, scans=scan)
		scan.add_to_plot(ax1)
		
		leads = [ i for i,j in zip(surf.get_positions(), surf.get_chemical_symbols()) if j == 'Pb' ]
		x_pb = [x for (x,y,z) in leads]
		y_pb = [y for (x,y,z) in leads]
		zr = [ i for i,j in zip(surf.get_positions(), surf.get_chemical_symbols()) if j == 'Zr' ]
		x_zr = [x for (x,y,z) in zr]
		y_zr = [y for (x,y,z) in zr]
		
		ax2.scatter(x_pb,y_pb,s=1)
		ax2.scatter(x_zr,y_zr,s=1)
		ax2.axis('equal')
		ax2.set_xlim(0,scan_s+2*borders)
		ax2.set_ylim(0,scan_s+2*borders)
		#fig.tight_layout()
		fig.savefig(folder_sim+sg+'_'+line_hkl+'_test.png',dpi=600)
		plt.close()		

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
		
		potential.project().show(
			cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
		)
		#probe.build()
		probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
		fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
		fig.tight_layout()
		fig.savefig(folder_sim+sg+'_'+line_hkl+'_potential.png',dpi=600)
		plt.close()

		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
		
		fph_potential.project().show(
			cmap="magma", figsize=(4, 4), title="Projected Electrostatic Potential", ax=ax1
		)

		fph_probe.show(figsize=(4, 4), title="Real Space Probe", ax=ax2)
		fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
		fig.tight_layout()
		fig.savefig(folder_sim+sg+'_'+line_hkl+'_fph_potential.png',dpi=600)
		plt.close()

		fig,(ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15,5))
		abtem.show_atoms(surf, ax=ax1, title="XY projection" )#, scans=scan)
		scan.add_to_plot(ax1)
		abtem.show_atoms(surf, ax=ax2, title="Cross-section", plane='xz')
		abtem.show_atoms(surf, ax=ax3, title="Cross-section", plane='yz')

		fig.suptitle(sample_name+', '+sg+', '+str_hkl,fontsize=18)
		fig.savefig(folder_sim+sg+'_'+line_hkl+'_combined.png',dpi=600)
		plt.close()

def dry_run(s,is_uvw=True,inplane_angle=None):
	dataset = prepare_job(s,is_uvw,inplane_angle)
	for i in dataset.keys():
		potential = dataset[i]['potential']
		probe = add_probe(potential)
		scan = add_scan(probe,potential)
		sg = dataset[i]['symm']

		plot_dataset(dataset[i],is_uvw)
		


def full_run(s,is_uvw=True,inplane_angle=None):
	dataset = prepare_job(s,is_uvw,inplane_angle)

	for i in dataset.keys():
		f = open(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'presets.txt','w')
		f.write('sblock_size\t'+str(sblock_size)+'\n')
		f.write('tolerance\t'+str(tol)+'\n')
		f.write('max_uvw\t'+str(max_uvw)+'\n')
		f.write('borders\t'+str(borders)+'\n')
		f.write('scan_s\t'+str(scan_s)+'\n')
		f.write('thickness\t'+str(thickness)+'\n')
		f.write('override_sampling\t'+str(override_sampling)+'\n')
		f.write('atom_to_zero\t'+str(atom_to_zero)+'\n')
		
		f.write('haadfinner\t'+str(haadfinner)+'\n')
		f.write('haadfouter\t'+str(haadfouter)+'\n')
		f.write('is_uvw\t'+str(is_uvw)+'\n')
		f.write('inplane_angle\t'+str(inplane_angle)+'\n')
		f.write('global_tilt\t'+str(global_tilt)+'\n')
		f.write('f_phonons\t'+str(frozen_phonons)+'\n')
		
		f.close()
		
		probe = dataset[i]['probe']
		print(probe)

		sg = dataset[i]['symm']
		potential = dataset[i]['potential']
		scan = add_scan(probe,potential)
		print(probe,potential)
		probe.grid.match(potential)



		plot_dataset(dataset[i],is_uvw)
		
		measurements = probe.scan(potential, scan=scan, detectors=[haadf_detector,abf_detector,bf_detector])
		img = measurements.compute(scheduler="threads", num_workers=num_workers)
		
		w = 0
		while w<=2:
			iimg = img[w].copy()
			det_s = ['haadf','abf','bf'][w]
			iimg.to_tiff(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.tif')
			iimg.to_zarr(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.zarr',overwrite=True)
			
			for k in [0.025,0.1,0.25,0.5,1]:
				blurred = iimg.gaussian_filter(k,boundary='constant')
				blurred.to_tiff(folder_sim+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
			w+=1


		fph_potential = dataset[i]['fph_potential']

		probe.grid.match(potential)
		scan = add_scan(probe,potential)

		fph_measurements = probe.scan(fph_potential, scan=scan, detectors=[haadf_detector,abf_detector,bf_detector])
		img = fph_measurements.compute(scheduler="threads", num_workers=num_workers)

		
		w = 0
		while w<=2:
			iimg = img[w].copy()
			det_s = ['haadf','abf','bf'][w]
			iimg.to_tiff(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.tif')
			iimg.to_zarr(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'.zarr',overwrite=True)
			
			for k in [0.025,0.1,0.25,0.5,1]:
				blurred = iimg.gaussian_filter(k,boundary='constant')
				blurred.to_tiff(folder_sim+'fph_'+dataset[i]['symm']+'_'+str(global_tilt)+'_'+''.join([str(q) for q in dataset[i]['hkl']])+'_'+det_s+'_'+str(k).replace('.','-')+'.tif')
			w+=1



#################RUN######################

#dry_run({'Pbam': [[2,-1,0],[2,1,0],[0,0,2]]},is_uvw=True,inplane_angle=0.00)
#full_run({'Pbam': [[2,-1,0],[2,1,0],[0,0,2]]},is_uvw=True,inplane_angle=0.00)

dry_run({'Pbam': [[0,0,2]]},is_uvw=True,inplane_angle=0.00)
full_run({'Pbam': [[0,0,2]]},is_uvw=True,inplane_angle=0.00)
