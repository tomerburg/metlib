import os, sys
import numpy as np
import xarray as xr
from metpy.units import units
from metpy.xarray import preprocess_xarray
import metpy.calc as calc

@preprocess_xarray
def calcavg(x,xavg,lon2d,lat2d,nlon,nlat,rad,box,eqrm):
    
    nbox = (2*box+1)*(2*box+1)
    
    #Iterate over latitude and longitude
    for j in range((box),(nlon-box)):
        for i in range((box),(nlat-box)):

            lon1d = lon2d[i-box:i+box+1,j-box:j+box+1].reshape((nbox))
            lat1d = lat2d[i-box:i+box+1,j-box:j+box+1].reshape((nbox))
            x1d = x[i-box:i+box+1,j-box:j+box+1].reshape((nbox))

            d1d = eqrm * np.sqrt(( (lon2d[i,j]-lon1d)*np.cos( (lat2d[i,j]+lat1d)/2.0 ) )**2.0 + (lat2d[i,j]-lat1d)**2.0)
            z = x1d[d1d < rad] / len(x1d[d1d < rad])
            xavg[i,j] = np.sum(z)
   
    return xavg

@preprocess_xarray
def area_average(var,rad,lon,lat):
    
    """Performs horizontal area-averaging of a field in latitude/longitude format.
    Parameters
    ----------
    var : (M, N) ndarray
        Variable to perform area averaging on. Can be 2, 3 or 4 dimensions. If 2D, coordinates must
        be lat/lon. If using additional dimensions, area-averaging will only be performed on the last
        2 dimensions, assuming those are latitude and longitude.
    rad : `pint.Quantity`
        The radius over which to perform the spatial area-averaging.
    lon : array-like
        Array of longitudes defining the grid
    lat : array-like
        Array of latitudes defining the grid
        
    Returns
    -------
    (M, N) ndarray
        Area-averaged quantity, returned in the same dimensions as passed.
    
    Notes
    -----
    This function was originally provided by Matthew Janiga and Philippe Papin using a Fortran wrapper for NCL,
    and converted to python with further efficiency modifications by Tomer Burg, with permission from the original
    authors.
    
    This function assumes that the last 2 dimensions of var are ordered as (....,lat,lon).
    """
    
    #convert radius to kilometers
    rad = rad.to('kilometers')
    
    #res = distance in km of dataset resolution, at the equator
    londiff = lon[1]-lon[0]
    latdiff = lat[1]-lat[0]
    lat_0 = 0.0 - (latdiff/2.0)
    lat_1 = 0.0 + (latdiff/2.0)
    dx,dy = calc.lat_lon_grid_deltas(np.array([lon[0],lon[1]]), np.array([lat_0,lat_1]))
    dx = dx.to('km')
    res = int((dx[0].magnitude + dx[1].magnitude)/2.0) * units('km')
    
    #---------------------------------------------------------------------
    #Error checks
    
    #Check to make sure latitudes increase
    reversed_lat = 0
    if lat[1] < lat[0]:
        reversed_lat = 1
        
        #Reverse latitude array
        lat = lat[::-1]
        
        #Determine which axis of variable array to reverse
        lat_dim = len(var.shape)-2
        var = np.flip(var,lat_dim)
        
    #Check to ensure input array has 2, 3 or 4 dimensions
    var_dims = np.shape(var)
    if len(var_dims) not in [2,3,4]:
        print("only 2D, 3D and 4D arrays allowed")
        return
    
    #---------------------------------------------------------------------
    #Prepare for computation
    
    #Number of points in circle (with buffer)
    box = int((rad/res)+2)

    #Define empty average array
    var_avg = np.zeros((var.shape))
        
    #Convert lat and lon arrays to 2D
    nlat = len(lat)
    nlon = len(lon)
    lon2d,lat2d = np.meshgrid(lon,lat)
    RPD = 0.0174532925
    lat2d = lat2d*RPD
    lon2d = lon2d*RPD

    #Define radius of earth in km
    eqrm = 6378.137
    
    #Create mask for elements of array that are outside of the box
    mask = np.zeros((lon2d.shape))
    nbox = (2*box+1)*(2*box+1)
    mask[box:nlat-box,box:nlon-box] = 1
    mask[mask==0] = np.nan
    
    #Calculate area-averaging depending on the dimension sizes
    if len(var_dims) == 2:
        var_avg = calcavg(var.magnitude, var_avg, lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
    elif len(var_dims) == 3:
        for t in range(var_dims[0]):
            var_avg[t,:,:] = calcavg(var[t,:,:].magnitude, var_avg[t,:,:], lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
    elif len(var_dims) == 4:
        for t in range(var_dims[0]):
            for l in range(var_dims[1]):
                var_avg[t,l,:,:] = calcavg(var[t,l,:,:].magnitude, var_avg[t,l,:,:], lon2d, lat2d, nlon, nlat, rad.magnitude, box, eqrm) * mask
                
    #If latitude is reversed, then flip it back to its original order
    if reversed_lat == 1:
        lat_dim = len(var.shape)-2
        var_avg = np.flip(var_avg,lat_dim)
    
    #Return area-averaged array with the same units as the input variable
    return var_avg * var.units

