#License: GNU GPL-v3

#%matplotlib inline #if in jupyter

#These libraries are essential to run the code
import numpy as np
import matplotlib.pyplot as plt
import os
import hyperspy.api as hs

def read_msa(path):
	f = os.listdir(path)
	#considering only one msa file per folder
	for i in f:
		if i.endswith('.msa'):
			fname = i
	s = hs.load(path+'/'+fname)
	return s
	
	
#Local variables
storage = '/home/vasily/test_EDX_sp/'
folders = os.listdir(storage)


def plot_sample(substrate_areas,close,savename):
	
	_substrate_EDS = [read_msa(i) for i in substrate_areas]
	_names = [i.split('/')[-3] for i in substrate_areas]

	names = sorted(_names, reverse=True)
	substrate_EDS = [ j for _,j in sorted(zip(_names,_substrate_EDS), reverse=True)]
	
	#here, if one wish to rename spectra from 'A,B,C,...', one may use names.replace('A','1')
	
	#Normalization
	for f in substrate_EDS:
		f.data /= f.data.max()
	
	#how to check the energy
	#hs.material.elements.Ga.Atomic_properties.Xray_lines
	E_FeKa = hs.material.elements.Fe.Atomic_properties.Xray_lines.Ka['energy (keV)']
	E_CrKa = hs.material.elements.Cr.Atomic_properties.Xray_lines.Ka['energy (keV)']
	E_CrKb = hs.material.elements.Cr.Atomic_properties.Xray_lines.Kb['energy (keV)']
	E_CrLa = hs.material.elements.Cr.Atomic_properties.Xray_lines.La['energy (keV)']
	E_CKa = hs.material.elements.C.Atomic_properties.Xray_lines.Ka['energy (keV)']
	E_GaKa = hs.material.elements.Ga.Atomic_properties.Xray_lines.Ka['energy (keV)']
	E_GaKb = hs.material.elements.Ga.Atomic_properties.Xray_lines.Kb['energy (keV)']
	E_GaLa = hs.material.elements.Ga.Atomic_properties.Xray_lines.La['energy (keV)']

	E_HfKa = hs.material.elements.Hf.Atomic_properties.Xray_lines.Ka['energy (keV)']
	E_HfKb = hs.material.elements.Hf.Atomic_properties.Xray_lines.Kb['energy (keV)']
	E_HfLa = hs.material.elements.Hf.Atomic_properties.Xray_lines.La['energy (keV)']
	E_HfLb1 = hs.material.elements.Hf.Atomic_properties.Xray_lines.Lb1['energy (keV)']
	E_HfMa = hs.material.elements.Hf.Atomic_properties.Xray_lines.Ma['energy (keV)']

	E_AuLa = hs.material.elements.Au.Atomic_properties.Xray_lines.La['energy (keV)']
	E_AuLb1 = hs.material.elements.Au.Atomic_properties.Xray_lines.Lb1['energy (keV)']
	E_AuMa = hs.material.elements.Au.Atomic_properties.Xray_lines.Ma['energy (keV)']
	
	E_OKa = hs.material.elements.O.Atomic_properties.Xray_lines.Ka['energy (keV)']
	
	
	plt.close('all')

	if close:
		ax = hs.plot.plot_spectra(substrate_EDS, style='cascade', padding=0.1,legend=names)
		
		plt.axvline(E_OKa, c='C2', ls=':', lw=0.5) 
		plt.axvline(E_CrKa, c='C1', ls=':', lw=0.5)
		plt.axvline(E_CrLa, c='C1', ls=':', lw=0.5) 
		
		plt.axvline(E_GaKa, c='C3', ls=':', lw=0.5) 	
		plt.axvline(E_GaKb, c='C3', ls=':', lw=0.5) 
		
		plt.axvline(E_GaLa, c='C3', ls=':', lw=0.5) 
		plt.axvline(E_HfMa, c='C4', ls=':', lw=0.5) 
		plt.axvline(E_HfLb1, c='C4', ls=':', lw=0.5) 
		plt.axvline(E_HfLa, c='C4', ls=':', lw=0.5) 

		plt.axvline(E_AuMa, c='C5', ls=':', lw=0.5) 
		plt.axvline(E_AuLb1, c='C5', ls=':', lw=0.5) 
		plt.axvline(E_AuLa, c='C5', ls=':', lw=0.5) 
		
		plt.axvline(8.040, c='k', ls=':', lw=0.5) 
		plt.title(savename)
		
		ax.figure.savefig(storage+savename+'0.png')
	else:
		
		ax = hs.plot.plot_spectra(substrate_EDS, style='cascade', padding=-1,legend=names)
		
		plt.axvline(E_OKa, c='C2', ls=':', lw=0.5) 
		plt.text(x=E_OKa+0.1, y=0.85, s='O-K', color='C2') 

		plt.axvline(E_CrKa, c='C1', ls=':', lw=0.5) 
		plt.text(x=E_CrKa+0.1, y=0.85, s='Cr-K$_\\alpha$', color='C1')
		plt.axvline(E_CrLa, c='C1', ls=':', lw=0.5) 
		plt.text(x=E_CrLa+0.1, y=0.7, s='Cr-L$_\\alpha$', color='C1')

		plt.axvline(E_CrKb, c='C1', ls=':', lw=0.5) 
		plt.text(x=E_CrKb+0.1, y=0.7, s='Cr-K$_\\beta$', color='C1')
		
		plt.axvline(E_GaKa, c='C3', ls=':', lw=0.5) 
		plt.text(x=E_GaKa+0.1, y=0.85, s='Ga-K$_\\alpha$', color='C3') 
		
		plt.axvline(E_GaKb, c='C3', ls=':', lw=0.5) 
		plt.text(x=E_GaKb+0.1, y=0.7, s='Ga-K$_\\beta$', color='C3')
		
		plt.axvline(E_GaLa, c='C3', ls=':', lw=0.5) 
		plt.text(x=E_GaLa+0.1, y=0.55, s='Ga-L$_\\alpha$', color='C3')

		plt.axvline(E_HfMa, c='C4', ls=':', lw=0.5) 
		plt.text(x=E_HfMa+0.1, y=0.4, s='Hf-M$_\\alpha$', color='C4') 
		plt.axvline(E_HfLb1, c='C4', ls=':', lw=0.5) 
		plt.text(x=E_HfLb1+0.1, y=0.55, s='Hf-L$_\\beta 1$', color='C4')
		plt.axvline(E_HfLa, c='C4', ls=':', lw=0.5) 
		plt.text(x=E_HfLa+0.1, y=0.7, s='Hf-L$_\\alpha$', color='C4') 

		plt.axvline(E_AuMa, c='C5', ls=':', lw=0.5) 
		plt.text(x=E_AuMa+0.1, y=0.85, s='Au-Ma', color='C5') 
		plt.axvline(E_AuLb1, c='C5', ls=':', lw=0.5) 
		plt.text(x=E_AuLb1+0.1, y=0.4, s='Au-L$_\\beta 1$', color='C5')
		plt.axvline(E_AuLa, c='C5', ls=':', lw=0.5) 
		plt.text(x=E_AuLa+0.1, y=0.4, s='Au-L$_\\alpha$', color='C5') 
		
		plt.axvline(8.040, c='k', ls=':', lw=0.5) 
		plt.text(x=8.14, y=0.35, s='Cu-K$_\\alpha$', color='k')
		
		plt.title(savename)
		
		ax.figure.savefig(storage+savename+'.png')
		
		

#Here we are extracting all the addresses for maps&areas within the path
maps,areas = [],[]
for q in folders:
	p = storage + q
	for root,d_names,f_names in os.walk(p, topdown=True, onerror=None, followlinks=False):
		for i in d_names:
			if 'map' in i or i.startswith('m'):
				print('map',root + '/' + i)
				maps.append(root + '/' + i+'/')
			elif 'area' in i or i.startswith('a'):
				print('area',root + '/' + i)
				areas.append(root + '/' + i+'/')

#Here we are distinguishing spectra by name
sample_areas = []
reference_areas = []

for a in areas:
	if 'ref' in a:
		reference_areas.append(a)
		#print('ref',a)
	else:
		sample_areas.append(a)
		#print('sample',a)

#Here - manually choosing ones, if needed
#sample_1_areas = [] #

plot_sample(sample_areas,True,'Sample')
plot_sample(sample_areas,False,'Sample')

plot_sample(reference_areas,True,'Sample')
plot_sample(reference_areas,False,'Sample')
