#License: MIT

import os
import numpy as np
import hyperspy.api as hs
import atomap.api as am
import atomap.initial_position_finding as ipf
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import scipy
#import scipy.ndimage
import tifffile

import sidpy
import pyTEMlib

import pyTEMlib.file_tools
import pyTEMlib.image_tools 
import pyTEMlib.probe_tools
import pyTEMlib.atom_tools

import numpy as np
import sidpy
from sidpy.sid.dimension import DimensionType

def hyperspy_to_sidpy(s, title=None):
    """Convert a HyperSpy Signal (2D/ND) to sidpy.Dataset with axis metadata."""
    if title is None:
        title = getattr(s, "title", "hyperspy_dataset")

    # Materialize if lazy:
    data = s.data #s.data.compute() if hasattr(s.data, "compute") else s.data
    ds = sidpy.Dataset.from_array(data, title=title)
    # (optional) carry HyperSpy metadata
    try:
        ds.metadata['experiment'] = dict(s.metadata)
    except Exception:
        pass
        
    def _safe_units(name, units):
        """Return non-empty units string acceptable to sidpy."""
        if units is None or str(units).strip() == "":
            # sensible defaults
            if name.lower() in ("x", "y", "row", "col", "kx", "ky"):
                return "px"          # pixel units for images
            if name.lower() in ("energy", "e", "eV"):
                return "eV"
            if name.lower() in ("time", "t"):
                return "s"
            if name.lower() in ('z','frames'):
                return 'frame'
            return "a.u."            # generic fallback
        return str(units)
        
    # Copy signal (and navigation) axes in order
    for i, ax in enumerate(s.axes_manager._axes):
        name   = getattr(ax, "name", None) or f"dim_{i}"
        units  = _safe_units(name, getattr(ax, "units", None))
        size   = int(getattr(ax, "size", data.shape[i]))
        # Prefer explicit coordinate array if present; else use scale*index
        if getattr(ax, "axis", None) is not None:
            vals = np.asarray(ax.axis)
        else:
            vals = ax.scale * np.arange(ax.size) + ax.offset
        if vals.size != size:
            vals = np.arange(size) * ax.scale + ax.offset
        print(ax.units)
        dim = sidpy.Dimension(vals, name=name, units=units)
        ds.set_dimension(i, dim)

    return ds
    
    
'''
Here we use the Diffeomorphic Demon Non-Rigid Registration as provided by simpleITK.
Please Cite:
    simpleITK
    and
    T. Vercauteren, X. Pennec, A. Perchant and N. Ayache Diffeomorphic Demons Using ITK's Finite Difference Solver Hierarchy The Insight Journal, 2007
'''

def alignment(s,folder,fname,NRA=False,bin_factor=1):
    s.data = s.data.astype(np.float64)
    Y, X = s.data.shape[1], s.data.shape[2]
    frames_per_chunk = 4
    s.data = s.data.rechunk((frames_per_chunk, Y, X))

    sd = s.copy().rebin(scale=(1, bin_factor, bin_factor))

    aligned = sd.copy()
    aligned.align2D(
        crop=True,
        reference='current',
        sub_pixel_factor=bin_factor*4
    )

    ds = hyperspy_to_sidpy(aligned, title="raw")
    ds.data_type = 'IMAGE_STACK'
    ds.x.dimension_type = DimensionType.SPATIAL
    ds.y.dimension_type = DimensionType.SPATIAL
    ds.z.dimension_type = DimensionType.TEMPORAL

    mean_ra = ds.mean(axis=0).compute()
    
    tifffile.imwrite(folder+fname+"_RA.tiff", mean_ra.astype(np.float32))

    if NRA:
        nonrigid_registered = pyTEMlib.image_tools.demon_registration(ds)
        tifffile.imwrite(folder+fname+"_NRA.tiff", nonrigid_registered.mean(axis=0).compute().astype(np.float32))
        
        
folder = '/path/to/folder'
ff = os.listdir(folder)
ending = '.dm3'
ff = [i[:-4] for i in ff if i.endswith(ending) ]#or dm4, or emd

for fname in ff:
    s = hs.load(folder+fname+ending,lazy=True)
    if len(s.data.shape)>2:
        print(fname)
        try:
            alignment(s,folder,fname,NRA=False,bin_factor=1)
        except:
            print('failed')
