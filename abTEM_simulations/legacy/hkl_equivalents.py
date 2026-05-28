#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as np
import diffpy.structure


def drop_negatives(hkl):
	'''
	Filters out Friedels pairs from the set of directions.
	Preference is given to positive values (ones with h+k+l > -h-k-l )
	hkl - array of vectors [(h_i),(k_i),(l_i),...]
	'''
	new_set = []
	for i in hkl:
		h,k,l = i  # noqa: E741
		mh,mk,ml = -h,-k,-l
		#print(new_set)
		if [mh,mk,ml] in hkl and [mh,mk,ml] not in new_set and [h,k,l] not in new_set:
			if sum([mh,mk,ml])<=sum(i):
				new_set.append(i)
			else:
				new_set.append([mh,mk,ml])
	return new_set


'''
def gen_hkl(uvw,max_uvw=20):
	print('UVW ',uvw)
	all_hkl = gen_list(max_uvw)
	sel_hkl = np.array([i for i in all_hkl if np.dot(i,uvw)==0])
	simplest = []
	for i in sel_hkl:
		u,c = np.unique(i, return_counts=True)
		cc = dict(zip(u,c))
		if 0 in u and cc[0] == 2:
			simplest.append(i)
	#print('S',simplest)
	m = np.sum(sel_hkl**2,axis=1)
	smallest = sel_hkl[m == min(m)]
	smallest = drop_negatives(smallest.tolist())
	#print(smallest)
	i = 0
	while len(smallest)<2:
		#print('Error - not enough hkl found for uvw given')
		thr = int(min(m)+i)
		#print(type(min(m)),type(thr))
		#print(m == thr)
		#print(len(m),len(sel_hkl))
		sm2 = sel_hkl[m <= thr]
		smallest = drop_negatives(sm2.tolist())
		i+=1
	sel_hkl = drop_negatives(sel_hkl.tolist())
	return smallest
'''

'''
def uvw_to_hkl(param_list,uvw,max_uvw):
	print('UVW ',uvw)
	#param_list = get_params(s)
	lat = diffpy.structure.Lattice(param_list[0],param_list[1],
							  param_list[2],param_list[3],
							  param_list[4],param_list[5])
	lat_r = lat.reciprocal()
	G = lat.metrics
	Gr = lat_r.metrics

	v = lat.cartesian(uvw)
	out = lat_r.fractional(v)

	print(out)
	#out = [i if i == 0 else 1/i for i in out]
	out = np.array(out)

	u = out[abs(out) > 0.0001]
	out = out/min(abs(u))
	print(out)
	m = find_multiplier(out,max_uvw)
	out = out*m

	out = out.round()
	out = out.astype(int)

	out = out.tolist()
	print('HKL ',out)

	return(out)
'''
