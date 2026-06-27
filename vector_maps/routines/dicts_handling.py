#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import numpy as _np
import ast

'''
This set of functions meant to parse human-readable dicts into a vector of parameters and back.
There is a support for equations on variables.
Code has been created with AI assistance (OpenAI GPT-5) and manually reviewed
'''

'''
#Example configuration (Si dumbbells)

#Meant to be 0.3867, 0.5469
lat_params = { 'abg':[0.3805, 0.5369, 89.75],
		'fit_abg':[True,True,True],
		'base':[-0.0005,.20,1.88],
		'fit_base':[True,True,True]
}

#Atom at (0,0); first sublattice. Since all other atoms are functionally connected to this one,
#it is reasonable to fix it due to a full correlation with shx/shy (lat_params['base'][0] and lat_params['base'][1])

motif = {'A_1':{'atom':'Si_1',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[False,False]},
}

#Centered atom of the first sublattice. Since 'eq' are present and not None, 'coords' and 'fit' are disabled
motif['A_1c'] = {	'atom':'Si_2',
			'coord':(0.,0.),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':  ["= motif['A_1'][0] + extra_pars['centering_a']", "= motif['A_1'][1] + extra_pars['centering_b']"]
			}
			
#Second sublattice; 'A_1' + dumbbell vector (in polar coordinates)
motif['B_1'] =  {'atom':'Si_3',
			'coord':(0.,0.2),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['A_1'][0] + extra_pars['db_dist']*np.sin(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][0]",
					"= motif['A_1'][1] + extra_pars['db_dist']*np.cos(extra_pars['db_angle']/180*np.pi)/lat_params['abg'][1]"]}

#Second sublattice; centered
motif['B_1c'] = {'atom':'Si_4',
			'coord':(0.5,0.7),
			'I':1,
			'use':True,
			'fit':[True,True],
			'eq':["= motif['B_1'][0] + extra_pars['centering_a']", "= motif['B_1'][1] + extra_pars['centering_b']"]}

#Extra variables - dumbbell vector in absolute polar coordinated relative to b; expected to be (L,0) but can be refined
#Centering vector in fractional coordinates
#True/False enables/disables refinement

extra_pars = {'db_dist':(0.1,True),
		'db_angle':(0,True),
		'centering_a':(0.5,True),
		'centering_b':(0.5,True)}

#'''



ALLOWED_NODES = (
	ast.Expression, ast.BinOp, ast.UnaryOp,
	ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
	ast.USub,
	ast.Constant, ast.Name, ast.Load,
	ast.Subscript, ast.Index, ast.Slice,
	ast.Attribute, ast.Call
)

def build_layout(lat_params, motif, extra_pars):
	idx = 0
	lat_idx = {'abg': [], 'base': []}

	# lattice (abg then base)
	for name in ('abg', 'base'):
		for _ in lat_params[name]:
			lat_idx[name].append(idx)
			idx += 1

	# motif coords, in motif key order, only if use=True
	motif_idx = {}
	for label, m in motif.items():
		if not m.get('use', True):
			continue
		x_idx = idx
		y_idx = idx + 1
		motif_idx[label] = (x_idx, y_idx)
		idx += 2

	# extra_pars at the end (in dict order)
	extra_idx = {}
	for name in extra_pars:
		extra_idx[name] = idx
		idx += 1

	return {
		'lat': lat_idx,
		'motif': motif_idx,
		'extra': extra_idx,
		'size': idx,
	}


def init_param_and_fit(lat_params, motif, extra_pars, layout):
	n = layout['size']
	p = _np.zeros(n, float)
	fit = _np.zeros(n, bool)

	# lattice
	for name, fit_name in (('abg', 'fit_abg'), ('base', 'fit_base')):
		for j, val in enumerate(lat_params[name]):
			i = layout['lat'][name][j]
			p[i] = val
			fit[i] = lat_params[fit_name][j]

	# motif
	for label, m in motif.items():
		if label not in layout['motif']:
			continue
		(ix, iy) = layout['motif'][label]
		x0, y0 = m['coord']
		fx, fy = m.get('fit', (False, False))
		p[ix], p[iy] = x0, y0
		fit[ix], fit[iy] = fx, fy

	# extra
	for name, (val, spec) in extra_pars.items():
		i = layout['extra'][name]
		p[i] = val
		if spec is True:
			fit[i] = True		   # independent, fitted
		elif spec is False:
			fit[i] = False		  # independent, fixed
		elif isinstance(spec, str) and spec.lstrip().startswith('='):
			fit[i] = False		  # dependent (eq will handle it)
		else:
			raise ValueError(f"Bad extra_pars spec for {name}: {spec}")

	return p, fit

def _compile_eq(expr_src):
	# expr_src: string without leading '='
	tree = ast.parse(expr_src, mode='eval')
	for node in ast.walk(tree):
		if not isinstance(node, ALLOWED_NODES):
			raise ValueError(f"Disallowed syntax in equation: {expr_src}")
	code = compile(tree, "<eq>", "eval")
	return code


def _make_env(p, layout):
	# Proxies exposing the same syntax as in your eq strings
	lat_idx = layout['lat']
	motif_idx = layout['motif']
	extra_idx = layout['extra']

	class LatProxy(dict):
		def __getitem__(self, key):
			idxs = lat_idx[key]
			return tuple(p[i] for i in idxs)

	class MotifProxy(dict):
		def __getitem__(self, label):
			ix, iy = motif_idx[label]
			return (p[ix], p[iy])

	class ExtraProxy(dict):
		def __getitem__(self, key):
			return p[extra_idx[key]]
			
	# minimal safe np-like namespace for expressions
	class NPProxy:
		sin = staticmethod(_np.sin)
		cos = staticmethod(_np.cos)
		tan = staticmethod(_np.tan)
		exp = staticmethod(_np.exp)
		sqrt = staticmethod(_np.sqrt)
		pi  = _np.pi

	return {
		'lat_params': LatProxy(),
		'motif': MotifProxy(),
		'extra_pars': ExtraProxy(),
		'np': NPProxy,
	}


def compile_equations(lat_params, motif, extra_pars, layout):
	n = layout['size']
	eq_mask = _np.zeros(n, bool)
	eq_funcs = [None] * n

	# --- helper factory
	def make_f(code):
		return lambda p, c=code: eval(c, {"__builtins__": {}}, _make_env(p, layout))

	for label, m in motif.items():
		if label not in layout['motif']:
			continue
		ix, iy = layout['motif'][label]
		eq = m.get('eq', (None, None))
		if eq is None:
			eq = (None, None)
			
		# x component
		if eq[0] is not None:
			src = eq[0].lstrip()
			if src.startswith('='):
				src = src[1:].strip()
			code = _compile_eq(src)
			eq_mask[ix] = True
			eq_funcs[ix] = make_f(code)

		# y component
		if len(eq) > 1 and eq[1] is not None:
			src = eq[1].lstrip()
			if src.startswith('='):
				src = src[1:].strip()
			code = _compile_eq(src)
			eq_mask[iy] = True
			eq_funcs[iy] = make_f(code)



	# --- 1) lattice equations: eq_abg, eq_base
	for name, eq_key in (('abg', 'eq_abg'), ('base', 'eq_base')):
		eq_list = lat_params.get(eq_key)
		if not eq_list:
			continue
		idxs = layout['lat'][name]
		for j, expr in enumerate(eq_list):
			if expr is None:
				continue
			src = expr.lstrip()
			if src.startswith('='):
				src = src[1:].strip()
			code = _compile_eq(src)
			i = idxs[j]
			eq_mask[i] = True
			eq_funcs[i] = make_f(code)


		   
	# 3) extra_pars eq via "= ..." spec
	for name, (val, spec) in extra_pars.items():
		if not (isinstance(spec, str) and spec.lstrip().startswith('=')):
			continue
		i = layout['extra'][name]
		s = spec.lstrip()[1:].strip()   # drop leading '='
		code = _compile_eq(s)
		eq_mask[i] = True
		eq_funcs[i] = make_f(code)

	return eq_mask, eq_funcs
	
def build_independent_index(fit, eq_mask):
	# independent = fit == True AND not equation-defined
	indep = _np.where(fit & ~eq_mask)[0]
	return indep

def inflate_params(x_indep, base_p, indep_idx, eq_mask, eq_funcs):
	"""
	Take reduced vector x_indep and:
	  - put values into their positions,
	  - recompute all eq-defined entries.
	Returns a fresh full param vector p.
	"""
	p = base_p.copy()
	p[indep_idx] = x_indep

	# Apply equations
	for i, is_eq in enumerate(eq_mask):
		if is_eq:
			p[i] = eq_funcs[i](p)

	return p
	
def unpack_to_dicts(p, lat_params, motif, extra_pars):
	'''
	Update original parameter dicts from a full parameter vector p
	'''
	layout = build_layout(lat_params, motif, extra_pars)
	# 1. lattice
	for name in ('abg', 'base'):
		idxs = layout['lat'][name]
		lat_params[name] = [p[i] for i in idxs]

	# 2. motif
	for label, (ix, iy) in layout['motif'].items():
		if label not in motif:
			continue
		motif[label]['coord'] = (p[ix], p[iy])

	# 3. extra parameters
	for name, i in layout['extra'].items():
		val, spec = extra_pars[name]
		extra_pars[name] = (p[i], spec)

	return lat_params, motif, extra_pars
	

def dicts_to_vector(lat_params, motif, extra_pars):
	layout = build_layout(lat_params, motif, extra_pars)
	p0, fit = init_param_and_fit(lat_params, motif, extra_pars, layout)
	eq_mask, eq_funcs = compile_equations(lat_params, motif, extra_pars, layout)
	indep_idx = build_independent_index(fit, eq_mask)
	x0 = p0[indep_idx]
	p = inflate_params(x0, p0, indep_idx, eq_mask, eq_funcs)

	return p,fit,eq_mask,eq_funcs

'''
#Usage example:


res = unpack_to_dicts(p, layout, lat_params, motif, extra_pars)
#'''
