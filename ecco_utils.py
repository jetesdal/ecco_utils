# -----------------------------------------------------------
# ecco_utils
# Functions to read and process ECCOv4 netcdf files
# -----------------------------------------------------------
# Copyright (c) 2017 Jan-Erik Tesdal
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import numpy as np 
import xarray as xr

def open_ecco_single_tendency(varname, dirname, base_dir):
    ds = xr.open_mfdataset(os.path.join(base_dir, dirname, '*.nc'),concat_dim='face')

    # make time an actual dimension
    if 'tim' in ds:
        tdim = ds['tim'].dims[0]
        ds = ds.swap_dims({tdim: 'tim'})
        ds = ds.rename({'tim': 'time'})
    # make depth an actual dimension
    if 'dep' in ds:
        ddim = ds['dep'].dims[0]
        ds = ds.swap_dims({ddim: 'dep'})
        ds = ds.rename({'dep': 'depth'})
    
    # drop fake coordinates
    for dim in list(ds.coords):
        if dim not in ('time','depth') and len(ds[dim].dims) == 1 and ds[dim].dims[0] in ('time','depth'):
                ds.__delitem__(dim)
    
    # get coordinates before processing
    dims = list(ds.dims)
    
    ds.coords['area'] = ds.area
    ds.coords['land'] = ds.land
    ds.coords['thic'] = ds.thic
    ds.coords['timstep'] = ds.timstep
    
    ds.coords['face'] = ds.face
    # face should be starting from 1 like it is defined in the file names
    ds.face.values = ds.face.values + 1
    
    # possibly transpose
    dims0 = ['time','face','depth']
    newdims = []
    for dim in dims0:
        if dim in dims:
            newdims.append(dim)
    for dim in dims:
        if not dim in dims0:
            newdims.append(dim)
    if dims != newdims:
        ds = ds.transpose(*newdims)
    
    return ds

def open_ecco_tendencies(*varnames, **keyword_parameters):
    base_dir = '/mnt/efs/data/ecco_v4/nctiles_tendencies'
    if ('base_dir' in keyword_parameters):
        base_dir = keyword_parameters['base_dir']
    darrays = [open_ecco_single_tendency(v, v, base_dir) for v in varnames]
    return xr.merge(darrays)

def open_ecco_snapshots(*varnames, **keyword_parameters):
    base_dir = '/mnt/efs/data/ecco_v4/nctiles_tendencies/'
    if ('base_dir' in keyword_parameters):
        base_dir = keyword_parameters['base_dir']
    darrays = [open_ecco_single_tendency(v,
                    'Ssnapshot' if v == 'SALT' else 'Tsnapshot',
                    base_dir) for v in varnames]
    return xr.merge(darrays)

def open_ecco_grid(grid_dir=None):
    if grid_dir is None:
        grid_dir = '/nctiles_grid/'
    grid = xr.open_mfdataset(grid_dir + 'GRID.*.nc', concat_dim='face')
    # Renaming dimensions to match data variables
    grid = grid.rename({'i3': 'i4'}).rename({'i2': 'i3'}).rename({'i1': 'i2'})
    return grid

def open_ecco_single_variable(varname, base_dir=None):
    if base_dir is None:
        base_dir = '/nctiles_monthly'
    ds = xr.open_mfdataset(os.path.join(base_dir, varname, '*.nc'),concat_dim='face')
    # make time an actual dimension
    if 'tim' in ds:
        tdim = ds['tim'].dims[0]
        ds = ds.swap_dims({tdim: 'tim'})
        ds = ds.rename({'tim': 'time'})
    # make sure that the dimension names are consistent
    if ds['area'].dims == ('face', 'i2', 'i3'):
        # we probably have a 2D field
        ds = ds.rename({'i3': 'i4'}).rename({'i2': 'i3'})
    ds = ds.reset_coords()
    da = ds[varname]
    dims = list(da.dims)
    # possibly transpose
    if 'time' in dims:
        if dims[0] != 'time':
            newdims = [d for d in dims]
            newdims[0] = 'time'
            newdims[1] = dims[0]
            da = da.transpose(*newdims)
    return da

def open_ecco_variables(ncdir, grid_dir=None, variables=None):
    if variables is None:
        variables = []
        for fname in os.listdir(ncdir):
            if os.path.isdir(os.path.join(ncdir,fname)):
                variables.append(fname)
    darrays = [open_ecco_single_variable(v, ncdir) for v in variables]
    if not grid_dir is None:
        grid = open_ecco_grid(grid_dir)
        grid = grid.set_coords(grid.data_vars)
        darrays.append(grid)
    return xr.merge(darrays)
