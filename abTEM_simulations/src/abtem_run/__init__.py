#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
__author__ = "Vasily A. Lebedev"
__license__ = "GPL-v3"

# Monkey-patches for system abtem compatibility. Must run before any code in
# this package calls abtem, so they live in __init__.py and execute on import.
# 1. cupy rejects `object` dtype in da.blockwise meta — fix ArrayObject._partition_args
# 2. fft="cufft" + numpy array crashes _fft_dispatch — add numpy fallback for that branch
# TODO (packaging phase 3): make these conditional on abtem version detection.
import inspect
import textwrap

import abtem.array as _ab_array
import abtem.core.fft as _ab_fft
import numpy as np

_src = textwrap.dedent(inspect.getsource(_ab_array.ArrayObject._partition_args))
_src = _src.replace("meta=xp.array((), object)", "meta=np.array((), dtype=object)")
_patch_ns = {**vars(_ab_array), 'np': np}
exec(_src, _patch_ns)
_ab_array.ArrayObject._partition_args = _patch_ns['_partition_args']
del _src, _patch_ns, _ab_array

_src = textwrap.dedent(inspect.getsource(_ab_fft._fft_dispatch))
_src = _src.replace(
    "        else:\n            raise RuntimeError()",
    "        elif config.get(\"fft\") == \"cufft\":\n            return getattr(np.fft, func_name)(x, **kwargs)\n        else:\n            raise RuntimeError()"
)
_patch_ns = {**vars(_ab_fft), 'np': np}
exec(_src, _patch_ns)
_ab_fft._fft_dispatch = _patch_ns['_fft_dispatch']
del inspect, textwrap, _src, _patch_ns, _ab_fft
