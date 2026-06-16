#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for refinement_routines helpers.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refinement_routines as rr


def test_empty_subarea_raises_clear_error():
	# a sub_area that excludes every observed point -> actionable message, not IndexError
	df = pd.DataFrame({'x_obs0': [10., 20., 30., 40.], 'y_obs0': [10., 20., 30., 40.]})
	with pytest.raises(ValueError, match='no observed points'):
		rr.preprocess_dataset({'base': [0., 0., 0.]}, {}, {}, df, 0.008,
				      recall_zero=True, sub_area=[100, 200, 100, 200])
