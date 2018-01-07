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


def open_ecco_grid(grid_dir):
    """
    Load ECCOv4 grid into xarray dataset.

    Parameters
    ----------
    grid_dir : str
        Directory path of ECCOv4 grid files
        
    Returns
    -------
    xarray.Dataset with ECCOv4 grid info
    """
    grid = xr.open_mfdataset(os.path.join(grid_dir, 'GRID.*.nc'), concat_dim='face')

    # Renaming dimensions to match data variables
    grid = grid.rename({'i3': 'i4'}).rename({'i2': 'i3'}).rename({'i1': 'i2'})

    return grid

def open_ecco_single_variable(ncdir, varname):
    """
    Load ECCOv4 variable into xarray dataset.

    Parameters
    ----------
    ncdir : str
        Directory path of ECCOv4 variable files

    varname: str
        Name of variable
        
    Returns
    -------
    xarray.Dataset with ECCOv4 variable
    """
    ds = xr.open_mfdataset(os.path.join(ncdir, varname, '*.nc'),concat_dim='face')

    # Make time an actual dimension
    if 'tim' in ds:
        tdim = ds['tim'].dims[0]
        ds = ds.swap_dims({tdim: 'tim'})
        ds = ds.rename({'tim': 'time'})

    # Make sure that the dimension names are consistent
    if ds['area'].dims == ('face', 'i2', 'i3'):
        # we probably have a 2D field
        ds = ds.rename({'i3': 'i4'}).rename({'i2': 'i3'})
    ds = ds.reset_coords()
    da = ds[varname]
    dims = list(da.dims)

    # Possibly transpose
    if 'time' in dims:
        if dims[0] != 'time':
            newdims = [d for d in dims]
            newdims[0] = 'time'
            newdims[1] = dims[0]
            da = da.transpose(*newdims)
    return da

def open_ecco_variables(ncdir, grid_dir=None, variables=None):
    """
    Load multiple ECCOv4 variables into single xarray dataset.

    Parameters
    ----------
    ncdir : str
        Directory path of ECCOv4 variable files

    grid_dir : str
        Directory path of ECCOv4 grid files


    variables:
        List of variables to be imported
        
    Returns
    -------
    xarray.Dataset with ECCOv4 variables
    """
    if variables is None:
        variables = []
        for fname in os.listdir(ncdir):
            if os.path.isdir(os.path.join(ncdir,fname)):
                variables.append(fname)
    darrays = [open_ecco_single_variable(ncdir, v) for v in variables]
    if not grid_dir is None:
        grid = open_ecco_grid(grid_dir)
        grid = grid.set_coords(grid.data_vars)
        darrays.append(grid)
    return xr.merge(darrays)

def open_ecco_single_tendency(ncdir, dirname):
    """
    Load ECCOv4 tendency field into xarray dataset. This function is useful for ECCOv4r2 files.

    Parameters
    ----------
    ncdir : str
        Directory path of tendency files

    dirname: str
        Name of tendency
        
    Returns
    -------
    xarray.Dataset with ECCOv4 tendency
    """
    ds = xr.open_mfdataset(os.path.join(ncdir, dirname, '*.nc'),concat_dim='face')

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

def open_ecco_tendencies(ncdir, grid_dir=None, variables=None):
    """
    Load multiple ECCOv4 tendencies into single xarray dataset. This function is useful for ECCOv4r2 files.

    Parameters
    ----------
    ncdir : str
        Directory path of ECCOv4 variable files

    grid_dir : str
        Directory path of ECCOv4 grid files


    variables:
        List of variables to be imported
        
    Returns
    -------
    xarray.Dataset with ECCOv4 tendencies
    """
    if variables is None:
        variables = []
        for fname in os.listdir(ncdir):
            if os.path.isdir(os.path.join(ncdir,fname)):
                variables.append(fname)
    darrays = [open_ecco_single_tendency(ncdir, v) for v in variables]
    if not grid_dir is None:
        grid = open_ecco_grid(grid_dir)
        grid = grid.set_coords(grid.data_vars)
        darrays.append(grid)
    return xr.merge(darrays)

def open_ecco_snapshots(ncdir, grid_dir=None, variables=None):
    """
    Load ECCOv4 snapshots into xarray dataset. This function is useful for ECCOv4r2 files.

    Parameters
    ----------
    ncdir : str
        Directory path of snapshot files

    grid_dir : str
        Directory path of ECCOv4 grid files

    variables:
        List of variables to be imported
        
    Returns
    -------
    xarray.Dataset with ECCOv4 snapshots
    """
    if variables is None:
        variables = []
        for fname in os.listdir(ncdir):
            if os.path.isdir(os.path.join(ncdir,fname)):
                variables.append(fname)
    darrays = [open_ecco_single_tendency(ncdir, 'Ssnapshot' if v == 'SALT' else 'Tsnapshot') for v in variables]
    if not grid_dir is None:
        grid = open_ecco_grid(grid_dir)
        grid = grid.set_coords(grid.data_vars)
        darrays.append(grid)
    return xr.merge(darrays)
