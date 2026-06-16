#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.patches as patches
import scipy.stats as st

import atomap.api as am

def plot_lattice(img,sublattice_list,fname,fl,sf,text):
	'''
	Purpose of this function is to create&save a figure which represents atomap sublattices
	inputs:
		img - image loaded as a hyperspy object ( see routines.load_frame )
		sublattice_list - array of atomap lattices to be plotted
		fl - str, general path to the workfolder
		sf - str, specific subpath to local workfolder
		fname - str, basename of file to be processed
		text - str, extra text field to be added to the safed file name
	'''
	
	
	atom_lattice = am.Atom_Lattice(
			image=img,
			sublattice_list=sublattice_list)
	s = atom_lattice.get_sublattice_atom_list_on_image()
	
	s.plot()
	p = s._plot
	fig = p.signal_plot.figure
	fig.delaxes(fig.axes[1]) #remove colorbar
	ax_list = fig.axes
	for i in ax_list:
		i.get_xaxis().set_visible(False)
		i.get_yaxis().set_visible(False)
		i.title.set_text('')
	fig.suptitle(text)
	fig.tight_layout()
	fig.savefig(fl+sf+'/'+text+'.png')

	plt.close('all')
	p.close()

def plot_unit_cell(fname_save, lat_params, motif, wrap=True, annotate=False, show_legend=True):
	a, b, gamma_deg = lat_params['abg']
	gamma = np.deg2rad(gamma_deg)

	# lattice vectors in Cartesian coordinates
	va = np.array([a, 0.0])
	vb = np.array([b * np.cos(gamma), b * np.sin(gamma)])

	# unit-cell outline
	cell = np.array([
		[0.0, 0.0],
		va,
		va + vb,
		vb,
		[0.0, 0.0]
	])

	fig, ax = plt.subplots(figsize=(4.5, 4.5))

	# draw cell border only
	ax.plot(cell[:, 0], cell[:, 1], lw=2, color='black')

	# collect unique elements and assign colors
	used_specs = [spec for spec in motif.values() if spec.get('use', True)]
	elements = []
	for spec in used_specs:
		atom_name = spec.get('atom', '')
		el = atom_name.split('_')[0] if atom_name else 'X'
		if el not in elements:
			elements.append(el)

	cmap = plt.get_cmap('tab10')
	color_map = {el: cmap(i % 10) for i, el in enumerate(elements)}

	pts_x = []
	pts_y = []

	# plot motif points
	for label, spec in motif.items():
		if not spec.get('use', True):
			continue

		fa, fb = spec['coord']
		if wrap:
			fa = fa % 1.0
			fb = fb % 1.0

		x = fa * a + fb * b * np.cos(gamma)
		y = fb * b * np.sin(gamma)

		atom_name = spec.get('atom', '')
		el = atom_name.split('_')[0] if atom_name else 'X'
		col = color_map[el]

		ax.scatter([x], [y], s=90, color=col, edgecolors='black', linewidths=0.7, zorder=3)

		pts_x.append(x)
		pts_y.append(y)

		if annotate:
			ax.text(x, y, ' ' + el, va='bottom', ha='left', fontsize=10)

	if show_legend:
		for el in elements:
			ax.scatter([], [], s=90, color=color_map[el], edgecolors='black',
					   linewidths=0.7, label=el)
		ax.legend(frameon=False, loc='best')

	all_x = list(cell[:, 0]) + pts_x
	all_y = list(cell[:, 1]) + pts_y

	xmin, xmax = min(all_x), max(all_x)
	ymin, ymax = min(all_y), max(all_y)

	pad = 0.15 * max(a, b)
	ax.set_xlim(xmin - pad, xmax + pad)
	ax.set_ylim(ymin - pad, ymax + pad)

	ax.set_aspect('equal')
	ax.set_xlabel('nm')
	ax.set_ylabel('nm')
	ax.set_title('Unit cell')
	plt.tight_layout()
	plt.savefig(fname_save + '.png', dpi=400)
	plt.close('all')

def plot_violin(fname_save,labels,df):
	'''
	Plot distributions of variables (intensities) specific for different atomic sites
	inputs:
		fname_save - str, full name of file to be saved, including path
		labels - list, descriptors to be plot
		df - pd dataframe to be used; expected to have a column 'motif'
			with a descriptors numbers and a column of 'I' with variables to plot
	'''
	plt.close()
	all_I = []
	#print(df)
	pos = []
	for i,j in enumerate(labels):
		all_I.append(df.loc[df['motif']==i,'I0'])
		#all_I.append(df.loc[df['motif']==i,'I_gauss'])
		pos.append(i)#TODO check these numbers vs atomic sites N as in config
	plt.violinplot(all_I,positions=pos)
	plt.xticks(ticks=pos, labels=labels)
	plt.ylabel('Intensity')
	plt.savefig(fname_save+'.png')
	plt.close('all')
	
	
def plot_quiver(fname_save,fin_lat,vdiff_xy,ang,vec_scale,hd_w=2,units_v='$1 \AA$',ell=False,calib=None,df=None):
	ref_angle = 0#np.pi/4
	vx = [i for i,j in fin_lat]
	vy = [j for i,j in fin_lat]
	#print(vdiff_xy)
	vu = [i for i,j in vdiff_xy]
	vv = [j for i,j in vdiff_xy]

	#vproj_a = [i for i,j in vproj]

	plt.close()
	fig1, ax1 = plt.subplots()
	ax1.set_box_aspect(1)
	ax1.set_title('')
	#ax1.scatter(x, y, color='blue', s=5)
	#color = matplotlib.colors.Normalize(vmin=0, vmax=1)
	#M = np.hypot(u, v)
	ax1.yaxis.set_inverted(True)
	
	ang = - np.array(ang)
	
	ang = (ang - ref_angle + np.pi) % (2*np.pi) - np.pi
	
	if ell:
		norm = mcolors.Normalize(vmin=0, vmax=np.pi)
		Q = ax1.quiver(vx, vy, vu, vv, ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
				width=.005,headwidth=1,norm=norm,pivot='middle')
		#ax1.quiver(vx, vy, -np.array(vu), -np.array(vv), ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
		#		width=.005,headwidth=hd_w,norm=norm,pivot='middle')
	else:
		norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
		Q = ax1.quiver(vx, vy, vu, vv, ang, angles='xy', scale_units='xy', scale=vec_scale,cmap='hsv',
				width=.005,headwidth=hd_w,norm=norm)#,norm=DivergingNorm(ref_angle))#cmap_bwr?
	ax1.quiverkey(Q, 0.6, 0.92, 0.1, r''+units_v, labelpos='E',
					   coordinates='figure', fontproperties={'size':18})#,fontsize=22
	ax1.set_xlabel('nm',fontsize=18)
	ax1.set_ylabel('nm',fontsize=18)
	ax1.tick_params(axis='both', which='major', labelsize=14)
	cb = fig1.colorbar(Q)
	#cb.set_label("arg", rotation=0, ha="center", va="bottom")
	#cb.ax.yaxis.set_label_coords(0.5, 1.01)
	if not ell:
		cb.set_ticks(np.array([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]))
		cb.set_ticklabels(
			[r"$-\pi$", r"$-\dfrac{\pi}{2}$", "$0$", r"$\dfrac{\pi}{2}$", r"$\pi$"]
		)
		
			
	else:
		cb.set_ticks(np.array([0,np.pi / 4, np.pi / 2, 3*np.pi / 4, np.pi]))
		cb.set_ticklabels(
			[r"$0$", r"$\dfrac{\pi}{4}$", r"$\dfrac{\pi}{2}$", r"$\dfrac{3\pi}{4}$",r"$\pi$"]
		)
	cb.ax.tick_params(labelsize=14)
	#plt.tight_layout()
	plt.savefig(fname_save+'.png',dpi=600)
	'''
	if not ell and df is not None:
		#try:
			req_cols = ['x_obs', 'y_obs', 'x0_std', 'y0_std']
			print(df['x_obs'])
			if all(c in df.columns for c in req_cols):
				for _, row in df.iterrows():
					xc = row['x_obs'][0]
					yc = row['y_obs'][0]
					sx = row['x0_std'][0] * calib
					sy = row['y0_std'][0] * calib

					if np.isfinite(xc) and np.isfinite(yc) and np.isfinite(sx) and np.isfinite(sy):
						ell_patch = patches.Ellipse(
							(xc, yc),
							width=2.0 * sx,
							height=2.0 * sy,
							angle=row['rot'][0],
							fill=False,
							edgecolor='black',
							linewidth=0.5,
							alpha=0.5,
							zorder=10
						)
						ax1.add_patch(ell_patch)
						plt.savefig(fname_save+'_2.png',dpi=600)
		#except Exception as e:
		#	print('Failed to draw uncertainty ellipses for', fname_save, ':', e)

	'''
	plt.close()
	fig1, ax1 = plt.subplots()
	ax1.set_box_aspect(1)
	ax1.set_title('')
	ax1.yaxis.set_inverted(True)
	ax1.scatter(np.array(vu)*1000,np.array(vv)*1000, s=80, facecolors='none', edgecolors='r')#,color=ang,cmap='hsv'
	ax1.spines['left'].set_position('zero')
	ax1.spines['right'].set_visible(False)
	ax1.spines['bottom'].set_position('zero')
	ax1.spines['top'].set_visible(False)
	ax1.spines['bottom'].set_visible(False)
	ax1.spines['left'].set_visible(False)
	ax1.xaxis.set_ticks_position('bottom')
	ax1.yaxis.set_ticks_position('left')
	
	''' #to plot reference sircle
	if not ell and calib is not None:
		ref_center = (0, 0)
		ref_radius = calib*1000 #nm to pm
		circle = patches.Circle(ref_center, ref_radius, linestyle='--', edgecolor='black', facecolor='none')
		ax1.add_patch(circle)
	'''	

	ax1.set_xlabel('pm', fontsize=16, labelpad=10)
	ax1.set_ylabel('pm', fontsize=16, labelpad=10, rotation=0)#, labelpad=10
	ax1.tick_params(axis='both', which='major', labelsize=14)
	print(max(vu)*1000*.9,max(vv)*1000*.75)
	ax1.yaxis.set_label_coords(.6, .95)
	ax1.xaxis.set_label_coords(.95,.45)
	xl = ax1.get_xlim()
	yl = ax1.get_ylim()
	ll = (min(xl[0],-xl[0],yl[0],-yl[0],xl[1],-xl[1],yl[1],-yl[1]),max(xl[0],-xl[0],yl[0],-yl[0],xl[1],-xl[1],yl[1],-yl[1]))
	ax1.set_xlim(ll[0],ll[1])
	ax1.set_ylim(ll[0],ll[1])
	
	ax1.annotate('', xy=(ax1.get_xlim()[1],0), xytext=(ax1.get_xlim()[0], 0), arrowprops=dict(arrowstyle="->", color='black'))#, xycoords=('axes fraction', 'data')
	ax1.annotate('', xy=(0,ax1.get_ylim()[1]), xytext=(0, ax1.get_ylim()[0]), arrowprops=dict(arrowstyle="->", color='black'))#, xycoords=('data', 'axes fraction')

	#ax1.text(ll[1]*.9,ll[1]*.9,'N = '+str(len(a)), fontsize=16)
	#Q = ax1.quiver([0 for i in vu],[0 for i in vu], vu, vv, ang, angles='xy', scale_units='xy', scale=1,cmap='hsv' )#cmap_bwr?
	#qk = ax1.quiverkey(Q, 0.8, 0.92, 0.1, r'$1 \AA$', labelpos='E',
	#				   coordinates='figure')
	plt.savefig(fname_save+'_fr0.png',dpi=600)
	plt.close('all')
	
def plot_stats_rep(vdist,fname_save,ang=False,ang_weights=None):
	#stats here

	if ang:
		N_st = 46
		a = np.array(vdist)/np.pi*180
		print(len(a),'angles',a[0])
	else:
		N_st = 35
		a = np.array(vdist)*1000
	e_min,e_max = min(a),max(a)
	edges = np.linspace(e_min, e_max, N_st, endpoint=True)
	q=st.lognorm.fit(a,scale=9,loc=3)
	mu=st.lognorm.mean(q[0],loc=q[1],scale=q[2])
	sigma=st.lognorm.std(q[0],loc=q[1],scale=q[2])
		
	pvalx = st.shapiro(np.log(a))[-1]
	pvalx2 = st.normaltest(a)[-1]
	print("p-value for `accepting` lognormality of x-data = ", pvalx)
	print("p-value for `accepting` normality of x-data = ", pvalx2)
	print("\n!!!!!!!Ok: the array come from lognormal distribution!!!!!!!!!\n" if pvalx>0.01  else "Hm...  the array isn't lognormal")
	print("Ok: the array come from normal distribution" if pvalx2>0.01  else "Hm...  the array isn't normal")
	plt.close()
	
	fig, ax = plt.subplots()
	n,bins,p = ax.hist(a, bins=edges, density=True, stacked=True,
				alpha = 0.1, lw=3, hatch='X', color='b', edgecolor='b',label='Averaged',weights=ang_weights)
	mid=[]
	i=1
	while i<len(edges):
		tmp=edges[i]+edges[i-1]
		tmp=tmp/2
		mid.append(tmp)
		i+=1

	gx=np.linspace(e_min,e_max,num=10000)
	dd=np.round(abs(sigma),1)

	#ddd=par[0]
	ddd=np.round(mu,1)
	if not ang:
		ax.plot(gx,st.lognorm.pdf(gx,q[0],q[1],q[2]))
	
		q=n.tolist()
		ax.text(mid[q.index(max(n))]+max(mid)/3,max(n)*.8,'$d_{mean}$='+str(ddd)+'$\pm$'+str(dd)+' pm', fontsize=16)
		ax.text(mid[q.index(max(n))]+max(mid)/3,max(n)*.95,'N = '+str(len(a)), fontsize=16)

	plt.subplots_adjust(right=0.95, left=0.15, top=0.92, bottom=0.18)
	ax.yaxis.grid(True)
	if not ang:
		ax.set_xlim(0,e_max)
	else:
		ax.set_xlim(e_min,e_max)
	
	ax.xaxis.label.set_size(20)
	ax.yaxis.label.set_size(20)
	ax.yaxis.set_visible(False)

	ax.tick_params(labelsize=16)
	[ax.spines[i].set_visible(False) for i in ["top","left","right"]]
	
	if ang:
		plt.xlabel("Direction, $^{\circ}$")	
	else:
		plt.xlabel("Residual distance, pm")
	plt.ylabel("Occurence")
	
	plt.savefig(fname_save+'_hist.png')
	
	ax.set_xlim(e_min,e_max)
	#ax.set_xscale('log')
	#plt.savefig(fname_save+'_hist_log.png')
	
	plt.close('all')
	
	if not ang:
		return '$d_{mean}$='+str(ddd)+'$\pm$'+str(dd)+' pm'
