#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
# Tests for the output-panel image slot: prefer <fname>.png, fall back to the TIFF.
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from routines import _panel_image


def test_panel_image_uses_png_when_present(tmp_path):
	import cv2
	(tmp_path / "sub").mkdir()
	cv2.imwrite(str(tmp_path / "f.png"), np.full((4, 4), 128, np.uint8))
	img = _panel_image(str(tmp_path / "sub") + os.sep, "f")
	assert img.ndim == 3 and img.shape[2] == 3


def test_panel_image_falls_back_to_tiff(tmp_path):
	import tifffile
	(tmp_path / "sub").mkdir()
	tifffile.imwrite(str(tmp_path / "f.tiff"),
			 np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4))
	# no f.png next to the frame -> the processing tiff is shown instead
	img = _panel_image(str(tmp_path / "sub") + os.sep, "f")
	assert img.ndim == 3 and img.shape[2] == 3
