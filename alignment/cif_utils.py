#License: GNU GPL-v3

import pandas as pd
import diffpy.structure
import os
import ase
import numpy as np

def filter_d_qq(all_hkl,param,d1,tol_sp):
	list_hkl = []

	lat = diffpy.structure.Lattice(param[0],param[1],
							  param[2],param[3],
							  param[4],param[5])
	lat_r = lat.reciprocal()
	
	dist = np.array([1/lat_r.norm(i) for i in all_hkl])
	print(dist)
	mask1 = dist>d1-tol_sp
	mask2 = dist<d1+tol_sp
	#print(dist[mask1&mask2])
	all_hkl = np.array(all_hkl)
	#*np.all(dist < d1+tol_sp)
	list_hkl = all_hkl[mask1&mask2].tolist()
	#print(list_hkl)

	return list_hkl

def get_params(s):
	if s.endswith('cif'):
		c = ase.io.read( s )
	else:
		raise ValueError("get_params expects a path ending with .cif")
	par = c.cell.lengths()
	ang = c.cell.angles()

	return par[0],par[1],par[2],ang[0],ang[1],ang[2]

def find_neighbour_hkls(all_hkl,param,v1,uvw,tol_d,tol_a):
	list_hkl = []

	lat = diffpy.structure.Lattice(param[0],param[1],
							  param[2],param[3],
							  param[4],param[5])
	lat_r = lat.reciprocal()

	ref_d = 1/lat_r.norm(v1)
	
	dist = np.array([1/lat_r.norm(i) for i in all_hkl])
	ang = np.array([lat_r.angle(i,v1) for i in all_hkl])
	
	print(dist)
	mask1 = (dist>ref_d-tol_d)&(dist<ref_d+tol_d)
	mask2 = abs(ang)<tol_a
	#print(dist[mask1&mask2])
	all_hkl = np.array(all_hkl)
	#*np.all(dist < d1+tol_sp)
	list_hkl = all_hkl[mask1&mask2].tolist()
	#print(list_hkl)

	fin_list = [i for i in list_hkl if np.dot(i,uvw)==0 and i!=v1]
	return list_hkl,fin_list

def gen_list(i_lim):
	d = []
	
	h = -i_lim
	while h <= i_lim:
		k = -i_lim
		while k <= i_lim:
			l = -i_lim
			while l <= i_lim:
				if h == 0 and k == 0 and l == 0:
					print("Zero")
				else:
					d.append((h,k,l))
				l+=1
			k+=1
		h+=1
	return d

	
def sym_check(sym,hkl):
	h,k,l = hkl
	res = True
	if sym=='I' and abs(h+k+l)%2 != 0:
		res = False
	if (sym=='A' or sym=='F') and abs(k+l)%2 != 0:
		res = False
	if (sym=='B' or sym=='F') and abs(h+l)%2 != 0:
		res = False
	if (sym=='C' or sym=='F') and abs(h+k)%2 != 0:
		print(h,k,abs(h+k)%2,abs(h+k)%2 != 0)
		res = False
	return res

def ipl_dist_q(param, v1):
		lat = diffpy.structure.Lattice(param[0],param[1],
							  param[2],param[3],
							  param[4],param[5])
		lat_r = lat.reciprocal()
		d = lat_r.norm(v1)

		return 1/d

def ipl_angle_q(param, v1, v2):
	lat = diffpy.structure.Lattice(param[0],param[1],
							  param[2],param[3],
							  param[4],param[5])
	lat_r = lat.reciprocal()
	phi = lat_r.angle(v1,v2)
	return phi
	
	
def fit_the_lattice(cif,param,sg,all_hkl,d1,d2,ang12,tol=0.1,tol_angle=5,full_output=[]):
	
	d1_list = filter_d_qq(all_hkl,param,d1,tol)
	d2_list = filter_d_qq(all_hkl,param,d2,tol)
	print(len(d1_list))
	print(len(d2_list))

	#cif file name, hkl1, hkl2, uvw, diff1,diff2
	
	
	list_of_norm = []
	for i in d1_list:
		for j in d2_list:
			
			sum_12 = np.array(i) + np.array(j)
			#print(sum_12,np.sum(sum_12%2>0))
			sum_12_ev = np.sum(sum_12%2>0)
			#print(sum_12_ev)
			sum_12 = sum_12.tolist()
			an = ipl_angle_q(param, i, j)
			
			if sum_12_ev==0 and an>ang12-tol_angle and an<ang12+tol_angle:
				print(i,j,np.cross(i,j))
				#print('Neighbours_1',find_neighbour_hkls(all_hkl,param,i,np.cross(i,j),tol,tol_angle))
				#print('Neighbours_2',find_neighbour_hkls(all_hkl,param,j,np.cross(i,j),tol,tol_angle))
				print(ipl_dist_q(param,i)/d1*100-100,ipl_dist_q(param,j)/d2*100-100)
				print(sum_12,ipl_dist_q(param,sum_12), np.dot(sum_12,np.cross(i,j)) )
				list_of_norm.append(np.cross(i,j))
				test = 'Unknown'
				if sg != '':
					symm = sg[0]
					if sym_check(symm,i) and sym_check(symm,j):
						test='Pass'
					else:
						test='Fail'
				raw_uvw = np.cross(i,j)
				u,v,w = raw_uvw
				div = np.gcd(u,v)
				div = np.gcd(div,w)
				uvw = np.array([u/div,v/div,w/div])
				full_output.append([sg,test,cif, i, j, uvw, ipl_dist_q(param,i)/d1,ipl_dist_q(param,j)/d2 ])
	return full_output
				
#weak to (anti)parallel v1,v2
def calc_uvw(params,v1,v2):
	raw_uvw = np.cross(v1,v2)
	u,v,w = raw_uvw
	div = np.gcd(u,v)
	div = np.gcd(div,w)
	uvw = np.array([int(u/div),int(v/div),int(w/div)])
	return uvw



d1,d2,ang=1.82,2.05,90 #in A

#Example
param = get_params('/path/to/cif')
print(ipl_angle_q(param,[0,0,2],[0,4,2]))


#another example - dists only, no angle

all_hkl = gen_list(20)
param = get_params('/path/to/cif')

tol = 0.1
d1 = 1.8
d2 = 2.5

d1_list = filter_d_qq(all_hkl,param,d1,tol)
d2_list = filter_d_qq(all_hkl,param,d2,tol)
print(len(d1_list))
print(len(d2_list))

list_of_norm = []
for i in d1_list:
	for j in d2_list:
		sum_12 = np.array(i)/2 + np.array(j)/2
		sum_12 = sum_12.tolist()
		if np.dot(i,j)==0:
			print(i,j,np.cross(i,j))
			print(sum_12,ipl_dist_q(param,sum_12), np.dot(sum_12,np.cross(i,j)) )
			list_of_norm.append(np.cross(i,j))
print('List of UVWs',list_of_norm)


#Full run
folder = '/path/to/allcifs'
f = os.listdir(folder)
full_output = []
all_hkl = gen_list(20)

for i in f:
	if i.endswith('.cif'):
		print('Fitting ',i)
		c = ase.io.read( folder + i )
		par = c.cell.lengths()
		ang_lat = c.cell.angles()
		ff = open(folder + i,'r')
		dd = ff.readlines()
		#print(dd[0])
		ff.close()
		sg = ''
		for l in dd:
			#print(l)
			if 'space_group_name' in l or 'space_group_name_H' in l:
				if len(l.split(' ')) > 1:
					sg = l.split(" ")[1]
					print(sg)
		param = [par[0],par[1],par[2],ang_lat[0],ang_lat[1],ang_lat[2]]
		print(sg)
		full_output = fit_the_lattice(i,param,sg,all_hkl,d1,d2,ang,full_output=full_output)
		
print(full_output)
