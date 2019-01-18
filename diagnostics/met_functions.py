import numpy as np
import xarray as xr
import scipy.ndimage as ndimage
from scipy.ndimage.filters import generic_filter as gf

#==============================================================================

#Define constants
r_earth = 6.371 * 10**6 #radius of Earth in meters
Rd = 287.0 #J/kg/K
Rv = 461.0 #J/kg/K
Cp = 1005.0 #J/kg/K
g = 9.80665 #m/s2
Lv = 2.501 * 10**6 #Latent heat of vaporization
kappa = (2.0/7.0) #Rd/Cp
rotation_rate = 7.292 * 10**-5 #rotation period of Earth (1/s)
pi = np.pi #pi

#==============================================================================
#//////////////////////////////////////////////////////////////////////////////
# THERMODYNAMICS
#//////////////////////////////////////////////////////////////////////////////
#==============================================================================


#Returns vapor pressure in units of hPa
#e = Vapor pressure = dewpoint
#es = Saturated vapor pressure = temperature
    
#temp = 2D array of temp or dewpoints (K)
    
def vapor_pressure(temp):
    
    #Calculate vapor pressure in hPa\
    part1 = np.multiply(17.67,np.subtract(temp,273.15))
    part2 = np.add(np.subtract(temp,273.15),243.5)
    part3 = np.divide(part1,part2)
    part4 = np.exp(part3)
    e = np.multiply(6.112,part4)
    
    return e

#==============================================================================
#Calculate mixing ratio, returns in units of kg/kg
#ws = saturated mixing ratio = temperature (@erified)
#w = mixing ratio = dewpoint

#temp = 2D array of temperature or dewpoint (K)
#pres = integer specifying pressure level (hPa)

def mixratio(temp,pres):
    
    #Calculate vapor pressure in hPa
    e = vapor_pressure(temp)
    
    #w = 0.622 * (e / (pres))
    w = np.multiply(0.622,np.divide(e,pres))
    
    return w

#==============================================================================
#Calculate wetbulb temperature, returns in Kelvin

#temp = 2D array of temperature (K)
#dwpt = 2D array of Dewpoints (K)
#pres = integer specifying pressure level (hPa)

def wetbulb(temp,dwpt,pres):
    
    #Calculate mixing ratios
    ws = mixratio(temp,pres)
    w = mixratio(dwpt,pres)

    #Formula used: Tw = T - (Lv/cp)(ws-w)\
    part1 = np.divide(Lv,Cp)
    part2 = np.subtract(ws,w)
    part3 = np.multiply(part1,part2)
    wb = np.subtract(temp,part3)
    
    return wb
    
#==============================================================================
#Calculate wetbulb temperature, returns in Kelvin

#temp = 2D array of temperature (K)
#q = 2D array of specific humidity (K)
#pres = integer specifying pressure level (hPa)

def wetbulb_q(temp,q,pres):
    
    #Calculate mixing ratios
    ws = mixratio(temp,pres)
    w = np.divide(q,np.add(q,1))
    w = np.divide(w,100.0)

    #Formula used: Tw = T - (Lv/cp)(ws-w) 
    part1 = np.divide(Lv,Cp)
    part2 = np.subtract(ws,w)
    part3 = np.multiply(part1,part2)
    wb = np.subtract(temp,part3)
    
    return wb
    
#==============================================================================
#Compute relative humidity given temperature and dewpoint
#Returns in units of percent (100 = 100%)

#temp,dwpt = 2D arrays or integer of temperature and dewpoint (K)

def relh_temp(temp,dwpt):
    
    #Compute actual and saturated vapor pressures
    e = vapor_pressure(dwpt)
    es = vapor_pressure(temp)
    
    #Compute RH
    rh = relh(e,es)
    
    return rh
    
#==============================================================================
#Compute relative humidity given mixing ratio or vapor pressure
#Returns in units of percent (100 = 100%)

#w,ws = actual & saturated mixing ratios (or vapor pressures)

def relh(w,ws):
    
    #Compute RH
    rh = np.multiply(np.divide(w,ws),100.0)
    
    return rh
    
#==============================================================================

#Compute the potential temperature
#Returns in units of Kelvin

#temp (2D array) = scalar temperature field (Kelvin)
#pres (float) = pressure level (hPa)

def theta(temp,pres):

    #Compute theta using Poisson's equation
    refpres = np.divide(1000.0,pres)
    refpres = np.power(refpres,kappa)
    theta = np.multiply(temp,refpres)
    
    return theta
    
#==============================================================================

#Compute the saturated potential temperature, following AMS methodology
#See: http://glossary.ametsoc.org/wiki/Equivalent_potential_temperature
#Returns in units of Kelvin

#temp (2D array/float) = scalar temperature field (Kelvin)
#pres (float) = pressure level (hPa)

def thetae(temp,dwpt,pres):
    
    #Calculate potential temp & saturated mixing ratio
    thta = theta(temp,pres)
    ws = mixratio(dwpt,pres)
    rh = np.divide(relh_temp(temp,dwpt),100.0)
    
    #Calculate potential temperature
    term1 = thta
    term2 = np.power(rh,np.divide(-1*np.multiply(Rv,ws),Cp))
    term3 = np.exp(np.divide(np.multiply(Lv,ws),np.multiply(Cp,temp)))
    
    thte = np.multiply(term1,np.multiply(term2,term3))
    
    return thte

#==============================================================================

#Calculates the dewpoint given RH and temperature
#Returns dewpoint in Kelvin

#temp = temperature in K
#rh = relative humidity in percent (0-100)

def dewpoint(temp,rh):

    #Source: http://andrew.rsmas.miami.edu/bmcnoldy/Humidity.html
    part1 = 243.04 * (np.log(rh/100.0) + ((17.625 * (temp-273.15)) / (243.04 + (temp-273.15))))
    part2 = 17.65 - np.log(rh/100.0) - ((17.625 * (temp-273.15)) / (243.04 + (temp-273.15)))
    
    Td = (part1 / part2) + 273.15
    
    return Td
    
#==============================================================================
#Calculate specific humidity
#Returns q in kg/kg
    
def specific_humidity(temp,pres,rh):
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    w = 0.622 * (e / pres) #used to be 0.622
    q = w
    
    return q
    
#==============================================================================
#Approximate the MSLP from the Hypsometric equation

#pres = surface pressure (Pa)
#hght = surface height (m)
#temp = 2m temperature (K)
#lapse = lapse rate (either constant or array in the same dimension as the other
#        variables. If none, assumed to be moist adiabatic lapse rate 6.5 K/km.)

def mslp(pres,hght,temp,lapse=6.5):
    
    #Approximate the virtual temperature as the average temp in the layer
    #using a 6.5 K/km lapse rate
    tslp = (lapse/1000)*hght + temp
    Tavg = 0.5*(tslp + temp)
    
    #Use the hypsometric equation to solve for lower pressure
    mslp = pres * np.exp((hght * g) / (Rd * Tavg))
    
    return mslp
    
#==============================================================================
#//////////////////////////////////////////////////////////////////////////////
# DIAGNOSTICS
#//////////////////////////////////////////////////////////////////////////////
#==============================================================================


#Computes the normalized anomaly of a variable with respect to its climatology.

#var = Variable of any dimension
#var_mean = Mean of the climatology of this variable
#var_std = Standard deviation of the climatology of this variable

def normalized_anomaly(var,var_mean,var_std):

    #Compute standardized anomaly
    var_anom = (var - var_mean) / var_std
    
    return var_anom

#==============================================================================

#Compute the magnitude of horizontal advection of a scalar quantity by the wind
#Returns in units of (scalar unit) per second

#var = 2D scalar field (e.g. temperature)
#u,v = 2D arrays of u & v wind components, in meters per second

def advection(var,u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(var)

    #Compute the gradient of the variable
    ddx,ddy = compute_gradient(var,lats,lons)
    
    #Compute advection (-v dot grad var)
    #adv = -1 * ((ddx*u) + (ddy*v))
    adv = np.add(np.multiply(ddx,u),np.multiply(ddy,v))
    adv = np.multiply(-1.0,adv)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        adv = xr.DataArray(adv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return adv
    
#==============================================================================
    
#Compute the horizontal divergence of a vector
#Returns in units of per second

#var = 2D scalar field (e.g. temperature)
#u,v = 2D arrays of u & v wind components, in meters per second

def divergence(u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(u)

    #Compute the gradient of the wind
    dudx = compute_gradient(u,lats,lons)[0]
    dvdy = compute_gradient(v,lats,lons)[1]

    #div = dudx + dvdy #dv/dx - du/dy
    div = np.add(dudx,dvdy)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        div = xr.DataArray(div, coords=[lats, lons], dims=['lat', 'lon'])
    
    return div
    
#==============================================================================

#Compute the relative vertical vorticity of the wind
#Returns in units of per second

#var = 2D scalar field (e.g. temperature)
#u,v = 2D arrays of u & v wind components, in meters per second

def relvort(u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(u)
    
    #Compute the gradient of the wind
    dudy = compute_gradient(u,lats,lons)[1]
    dvdx = compute_gradient(v,lats,lons)[0]

    #Compute relative vorticity (dv/dx - du/dy)
    vort = np.subtract(dvdx,dudy)
    
    #Account for southern hemisphere
    tlons,tlats = np.meshgrid(lons,lats)
    vort[tlats<0] = vort[tlats<0] * -1
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        vort = xr.DataArray(vort, coords=[lats, lons], dims=['lat', 'lon'])
    
    return vort
    
#==============================================================================

#Compute the absolute vertical vorticity of the wind
#Returns in units of per second

#var = 2D scalar field (e.g. temperature)
#u,v = 2D arrays of u & v wind components, in meters per second

def absvort(u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(u)
    
    #Compute relative vorticity
    vort = relvort(u,v,lats,lons)
    
    #Compute the Coriolis parameter (after converting lat to radians)
    cor2d = coriolis(lats,lons)

    #Compute absolute vorticity (relative + coriolis parameter)
    vort = np.add(vort,cor2d)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        vort = xr.DataArray(vort, coords=[lats, lons], dims=['lat', 'lon'])
    
    return vort
    
#==============================================================================

#Computes and returns a 2D array of the Coriolis parameter
#Returns coriolis in units of 1/s

#lats,lons (2D array) = array of latitude and longitudes (degrees)

def coriolis(lats,lons):
    
    #Compute the Coriolis parameter (after converting lat to radians)
    lons2,lats2 = np.meshgrid(lons,lats)
    sinlat = np.sin(lats2 * (pi/180))
    cor = np.multiply(2.0,np.multiply(rotation_rate,sinlat))
    
    return cor
    
#==============================================================================

#Compute the u and v components of the geostrophic wind
#Returns in units of meters per second

#hght = 2D scalar geopotential height field (m)
#lats,lons = 2D arrays of lat and lon

def geo(hght,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(hght)

    #Compute geopotential height gradient on pressure surface
    dzdx,dzdy = compute_gradient(hght,lats,lons)
    
    #2D array of Coriolis parameter for each lat/lon
    cor = coriolis(lats,lons)
    
    #Compute the geostrophic wind
    ug = (-1.0 * dzdy * g) / cor
    vg = (dzdx * g) / cor
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ug = xr.DataArray(ug, coords=[lats, lons], dims=['lat', 'lon'])
        vg = xr.DataArray(vg, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ug,vg
    
#==============================================================================

#Compute the u and v components of the ageostrophic wind
#Returns in units of meters per second

#hght = 2D scalar geopotential height field (m)
#u,v = 2D arrays of u and v components of wind (m/s)
#lats,lons = 2D arrays of lat and lon

def ageo(hght,u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(hght)

    #Compute the geostrophic wind
    ug,vg = geo(hght,lats,lons)
    
    #Compute the ageostrophic wind
    ua = u - ug
    va = v - vg
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ua = xr.DataArray(ua, coords=[lats, lons], dims=['lat', 'lon'])
        va = xr.DataArray(va, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ua,va
    
#==============================================================================

#Compute the u and v components of the Q-vector
#Returns in units of meters per second

#temp = 2D scalar temperature field (K)
#hght = 2D scalar geopotential height field (m)
#lev = Pressure level (hPa)
#lats,lons = 2D arrays of lat and lon
#smooth = integer representing sigma level of smoothing
#static stability = assumed to be 1, unless provided

def qvect(temp,hght,lev,lats,lons,smooth,static_stability=1):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(temp)
    
    #Smooth data
    hght = ndimage.gaussian_filter(hght,sigma=smooth,order=0)
    temp = ndimage.gaussian_filter(temp,sigma=smooth,order=0)
    
    #Convert pressure to Pa
    levPa = lev * 100.0

    #Compute the geostrophic wind
    ug,vg = geo(hght,lats,lons)
    
    #Compute the constant out front
    const = (-1.0 * Rd) / (levPa * static_stability)
    
    #Compute gradient quantities
    dtdx,dtdy = compute_gradient(temp,lats,lons)
    dudx,dudy = compute_gradient(ug,lats,lons)
    dvdx,dvdy = compute_gradient(vg,lats,lons)
    
    #Compute u,v components of Q-vector
    Qu = const * ((dudx*dtdx) + (dvdx*dtdy))
    Qv = const * ((dudy*dtdx) + (dvdy*dtdy))
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        Qu = xr.DataArray(Qu, coords=[lats, lons], dims=['lat', 'lon'])
        Qv = xr.DataArray(Qv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return Qu,Qv
    
#==============================================================================
#Compute integrated vapor transport, assuming the pressure
#interval is constant

#temp = 3D array of temperature (K)
#rh = 3D array (lat,lon,lev) of relative humidity (in %)
#levs = 1D array of pressure levels (hPa)
#u = u-wind (m/s)
#v = v-wind (m/s)

def ivt(temp,rh,levs,u,v,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(temp)
    
    #If using xarray, convert to numpy arrays
    if use_xarray == 1:
        try:
            temp = temp.values
        except:
            pass
        try:
            rh = rh.values
        except:
            pass
        try:
            u = u.values
        except:
            pass
        try:
            v = v.values
        except:
            pass
    
    #Get list of pressure levels in hPa, convert to Pa
    levs = levs * 100.0 #convert pressure to Pa
    
    #determine vertical dz in Pa, assuming levs array is uniform
    vint = (levs[1]-levs[0])
    
    nvert, nlat, nlon = np.shape(rh)
    pres = np.copy(rh) * 0.0
    
    #Arrange a 3D pressure array
    for k in range(0,nvert):
        pres[k] += levs[k]
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    q = 0.622 * (e / pres)
    
    #Compute u and v components of IVT vector
    ut = np.trapz(u*q, axis=0, dx=vint) / -9.8
    vt = np.trapz(v*q, axis=0, dx=vint) / -9.8

    #Compute magnitude of IVT vector
    ivt = magnitude(ut,vt)
    
    #Convert back to xarray dataset, if initially passed as one
    if use_xarray == True:
        ut = xr.DataArray(ut, coords=[lats, lons], dims=['lat', 'lon'])
        vt = xr.DataArray(vt, coords=[lats, lons], dims=['lat', 'lon'])
    
    return ut, vt, ivt

#==============================================================================
#Compute integrated vapor over a certain pressure layer, assuming the pressure
#interval is constant

#temp = 3D array of temperature (K)
#rh = 3D array (lat,lon,lev) of relative humidity (in %)
#pressfc = Surface pressure (hPa)
#levs = 1D array of pressure levels (hPa)
    
def integrated_vapor(temp,rh,pressfc,levs,lats,lons):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(temp)
    
    #If using xarray, convert to numpy
    if use_xarray == 1:
        try:
            temp = temp.values
        except:
            pass
        try:
            rh = rh.values
        except:
            pass
    
    #Convert pressure to hPa
    levs = levs * 100.0
    
    #determine vertical dz in Pa, assuming levs array is uniform
    vint = (levs[1]-levs[0])
    
    #pres,lons0,lats0 = np.meshgrid(rh.lev,lons,lats)
    nvert, nlat, nlon = np.shape(rh)
    pres = np.copy(rh) * 0.0
    
    #Arrange a 3D pressure array
    for k in range(0,nvert):
        pres[k] += levs[k]
        
        #Mask by surface pressure
        tmp = rh[k]
        tmp[pressfc < levs[k]] = 0.0
        rh[k] = tmp
    
    #saturated vapor pressure in Pa
    es = vapor_pressure(temp) * 100.0
    
    #get e from RH (in decimals) in Pa
    e = (rh / 100.0) * es
    
    #Approximate specific humidity q ~ w
    w = 0.622 * (e / pres) #used to be 0.622
    q = w
    
    #Compute integrated vapor
    iv = np.trapz(q, axis=0, dx=vint) / -9.8
    
    #Return ut, vt as xarray DataArrays
    if use_xarray == 1:
        iv = xr.DataArray(iv, coords=[lats, lons], dims=['lat', 'lon'])
    
    return iv
    
#==============================================================================
#Calculates moisture convergence

#u = u-wind (m/s)
#v = v-wind (m/s)
#dwpt = dewpoint (Kelvin)
#pres = pressure (hPa)
#smth = sigma to smooth with gaussian filter. Default is no smoothing.

def moisture_conv(u,v,temp,dwpt,pres,lats,lons,smth=0):
    
    #Smooth all fields
    u = smooth(u,smth)
    v = smooth(v,smth)
    temp = smooth(temp,smth)
    dwpt = smooth(dwpt,smth)
    pres = smooth(pres,smth)
    
    #Compute relative humidity
    rh = relh_temp(temp,dwpt)
    
    #Compute q
    q = specific_humidity(temp,pres,rh)# * 1000.0
    
    #Compute moisture convergence
    term1 = advection(q,u,v,lats,lons)
    term2 = np.multiply(q,divergence(u,v,lats,lons))
    
    return (term1 - term2)
    
#==============================================================================
#Computes the horizontal gradient of a 2D scalar variable
#Returns ddx, ddy (x and y components of gradient) in units of (unit)/m

def compute_gradient(var,lats,lons):

    #Pull in lat & lon resolution
    latres = abs(lats[1]-lats[0])
    lonres = abs(lons[1]-lons[0])    
    
    #compute the length scale for each gridpoint as a 2D array
    lons2,lats2 = np.meshgrid(lons,lats)
    dx = calculate_distance_2d(lats2,lats2,lons2-(lonres),lons2+(lonres))
    dy = calculate_distance_2d(lats2-(latres),lats2+(latres),lons2,lons2)
    
    #Compute the gradient of the variable
    dvardy,dvardx = np.gradient(var)
    ddy = np.multiply(2,np.divide(dvardy,dy))
    ddx = np.multiply(2,np.divide(dvardx,dx))

    return ddx,ddy

#==============================================================================
#Calculates dx and dy for 2D arrays

def calculate_distance_2d(lat1,lat2,lon1,lon2):
    #=ACOS(COS(RADIANS(90-Lat1)) *COS(RADIANS(90-Lat2)) +SIN(RADIANS(90-Lat1)) *SIN(RADIANS(90-Lat2)) *COS(RADIANS(Long1-Long2))) *6371
    step1 = np.cos(np.radians(90.0-lat1))
    step2 = np.cos(np.radians(90.0-lat2))
    step3 = np.sin(np.radians(90.0-lat1))
    step4 = np.sin(np.radians(90.0-lat2))
    step5 = np.cos(np.radians(lon1-lon2))
    dist = np.arccos(step1 * step2 + step3 * step4 * step5) * r_earth
    
    return dist
    
#==============================================================================

#Computes Ertel Potential Vorticity
#Returns EPV in units of PVU
    
#u = u-wind (m/s)
#v = v-wind (m/s)
#dthetadp = change of theta with respect to pressure

def pv_ertel(u,v,dthetadp,lats,lons):

    #Compute absolute vorticity
    vort = absvort(u,v,lats,lons)
    
    #pv = -g(absvort)(dthetadp)
    pv = np.multiply(g,np.multiply(vort,dthetadp))
    pv = np.multiply(pv,-1.0)
    
    #convert to PVU
    pv = pv * 10**6

    return pv
    
#==============================================================================

#Transform a variable to isentropic coordinates, specifying a single isentropic level

#temp = 3D temperature array (K)
#u = 3D wind array (u)
#v = 3D wind array (v)
#lev = Desired isentropic level (K, scalar)
#lats = 1D lat array
#lons = 1D lon array
#levs = 1D pressure array
#tomask = mask array values where the desired isentropic surface is below the
#         ground. Yes=1, No=0. Default is yes (1).
    
#Returns a python list with the following quantities:
#[0] = 2D pressure array
#[1] = u-wind
#[2] = v-wind
#[3] = d(theta)/dp
#[4] = 2D array corresponding to the first k-index of where the
#      theta threshold is exceeded.

def isentropic_transform(temp,u,v,lev,lats,lons,levs,tomask=1):
    
    #Check if input is an xarray dataarray
    use_xarray = check_xarray(temp)
    
    #If using xarray, convert to numpy
    if use_xarray == 1:
        temp = temp.values
        u = u.values
        v = v.values
    
    #Subset data values to below 100 hPa
    tlev = float(lev)
    
    #Arrange a 3D pressure array of theta
    vtheta = np.copy(temp) * 0.0
    
    nvert, nlat, nlon = np.shape(temp)
    for k in range(0,nvert):
        vtheta[k] = theta(temp[k],temp[k].lev.values)
        
    #Arrange 2D arrays of other values
    tpres = 0
    tu = 0
    tv = 0
    
    #Eliminate any NaNs to avoid issues
    temp = np.nan_to_num(temp)
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)
    vtheta = np.nan_to_num(vtheta)
    
    #==================================================================
    
    #Step 0: Get 3d array of pressure
    levs3d = np.copy(temp) * 0.0
    
    nvert, nlat, nlon = np.shape(temp)
    for k in range(0,nvert):
        levs3d[k] = (vtheta[0] * 0.0) + levs[k]
    
    #------------------------------------------------------------------
    #Step 1: find first instances bottom-up of theta exceeding threshold
    
    #Check where the theta threshold is exceeded in the 3D array
    check_thres = np.where(vtheta >= tlev)
    check_ax1 = check_thres[0]
    check_ax2 = check_thres[1]
    check_ax3 = check_thres[2]
    
    #This is a 2D array corresponding to the first k-index of where the
    #theta threshold is exceeded.
    thres_pos = np.copy(vtheta[0]) * 0.0
    
    #Loop through all such positions and only record first instances
    for i in range(0,len(check_ax1)):
        pres_level = check_ax1[i]
        jval = check_ax2[i]
        ival = check_ax3[i]
        
        if thres_pos[jval][ival] == 0: thres_pos[jval][ival] = pres_level
    
    #------------------------------------------------------------------
    #Step 2: get the theta values corresponding to this axis
    
    #Convert the position of the theta threshold values to something readable
    thres_pos = thres_pos.astype('int64')
    thres_last = thres_pos - 1
    thres_last[thres_last < 0] = 0
    
    #replace NaNs, if any
    thres_pos = np.nan_to_num(thres_pos)
    thres_last = np.nan_to_num(thres_last)
    vtheta = np.nan_to_num(vtheta)
    
    #Get theta values where it's first exceeded and 1 vertical level below it
    ktheta = np.ndarray.choose(thres_pos,vtheta)
    ltheta = np.ndarray.choose(thres_last,vtheta)
    
    #Get the difference in theta between levels
    diffu = np.abs(np.subtract(tlev,ktheta))
    diffl = np.abs(np.subtract(tlev,ltheta))
    
    #Percentage from the lower level to the upper one
    perc = np.divide(diffl,np.add(diffu,diffl))
    
    #------------------------------------------------------------------
    #Step 3: find pressure at this level
    
    valu = np.ndarray.choose(thres_pos,levs3d)
    vall = np.ndarray.choose(thres_last,levs3d)
    
    #Adjustment factor
    fac = np.multiply(np.subtract(vall,valu),perc)
                    
    #New pressure
    tpres = np.subtract(vall,fac)
    
    #------------------------------------------------------------------
    #Step 3a: get d(theta)/dp array
    
    #d(theta)/dp = ktheta-ltheta / valu-vall
    dthetadp = np.divide(np.subtract(ktheta,ltheta),np.subtract(valu,vall))
    
    #Convert to units of K/Pa
    dthetadp = np.divide(dthetadp,100)
    
    #------------------------------------------------------------------
    #Step 4: find wind at this level
    
    uu = np.ndarray.choose(thres_pos,u)
    ul = np.ndarray.choose(thres_last,u)
    vu = np.ndarray.choose(thres_pos,v)
    vl = np.ndarray.choose(thres_last,v)
    
    fac = np.multiply(np.subtract(ul,uu),perc)
    tu = np.subtract(ul,fac)

    fac = np.multiply(np.subtract(vl,vu),perc)
    tv = np.subtract(vl,fac)
    
    #==================================================================
    # ALL RESUMES HERE
    #==================================================================
    
    if tomask == 1:
        pres = np.ma.masked_where(thres_pos <= 1.0,tpres)
        u = np.ma.masked_where(thres_pos <= 1.0,tu)
        v = np.ma.masked_where(thres_pos <= 1.0,tv)
    
    #Convert back to xarray, if specified initially
    if use_xarray == 1:
        pres = xr.DataArray(pres,coords=[lats,lons],dims=['lat','lon'])
        u = xr.DataArray(u,coords=[lats,lons],dims=['lat','lon'])
        v = xr.DataArray(v,coords=[lats,lons],dims=['lat','lon'])
    
    return pres,u,v,dthetadp,thres_pos
    
    
#==============================================================================
#//////////////////////////////////////////////////////////////////////////////
# OTHER UTILITIES
#//////////////////////////////////////////////////////////////////////////////
#==============================================================================


#Compute the magnitude of any vector

def magnitude(x,y):
    
    mag = np.sqrt(np.add(np.square(x),np.square(y)))
    
    return mag
    
#==============================================================================
#Calculate the difference between two arrays, returns in their original units
    
#upper = upper level
#lower = lower level

def thickness(upper,lower):
    
    thick = upper - lower
    
    return thick

#==============================================================================
#Gaussean smooth (sig = sigma to smooth by)

def smooth(prod,sig):
    
    #Check if variable is an xarray dataarray
    try:
        lats = prod.lat.values
        lons = prod.lon.values
        prod = ndimage.gaussian_filter(prod,sigma=sig,order=0)
        prod = xr.DataArray(prod, coords=[lats, lons], dims=['lat', 'lon'])
    except:
        prod = ndimage.gaussian_filter(prod,sigma=sig,order=0)
    
    return prod

#==============================================================================
#Area averaging a lat/lon grid by a specified radius in degrees (not kilometers)
    
#Prod = 2D variable to be area-averaged
#deg = Degree radius to smooth over (e.g., 2 for 2 degrees)

def area_average_degree(prod,deg,lats,lons):
    
    #Check if input product is an xarray dataarray
    use_xarray = check_xarray(prod)
    
    #Determine radius in gridpoint numbers
    res = abs(lats[1] - lats[0])
    
    #Perform area-averaging
    radius = int(float(deg)/res)
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y1,x1 = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x1**2 + y1**2 <= radius**2
    kernel[mask] = 1
    prod = gf(prod, np.average, footprint=kernel)
    
    #Convert back to xarray dataarray, if specified
    if use_xarray == 1:
        prod = xr.DataArray(prod, coords=[lats, lons], dims=['lat', 'lon'])
    
    #Return product
    return prod
    
#==============================================================================
#Return the index of the value closest to the one passed in the array

def find_nearest_index(array,val):
    return np.abs(array - val).argmin()

#==============================================================================
#Return the value closest to the one passed in the array
    
def find_nearest_value(array,val):
    return array[np.abs(array - val).argmin()]

#==============================================================================
#Plug a small array into a large array, assuming they have the same lat/lon
#resolution.

#small = 2D array to be inserted into "large"
#small_lat = 1D array of lats
#small_lon = 1D array of lons
#large = 2D array for "small" to be inserted into
#large_lat = 1D array of lats
#large_lon = 1D array of lons

def plug_array(small,small_lat,small_lon,large,large_lat,large_lon):
    
    small_minlat = min(small_lat)
    small_maxlat = max(small_lat)
    small_minlon = min(small_lon)
    small_maxlon = max(small_lon)
    
    if small_minlat in large_lat:
        minlat = np.where(large_lat==small_minlat)[0][0]
    else:
        minlat = min(large_lat)
    if small_maxlat in large_lat:
        maxlat = np.where(large_lat==small_maxlat)[0][0]
    else:
        maxlat = max(large_lat)
    if small_minlon in large_lon:
        minlon = np.where(large_lon==small_minlon)[0][0]
    else:
        minlon = min(large_lon)
    if small_maxlon in large_lon:
        maxlon = np.where(large_lon==small_maxlon)[0][0]
    else:
        maxlon = max(large_lon)
    
    large[minlat:maxlat+1,minlon:maxlon+1] = small
    
    return large

#==============================================================================
#Check if the passed array is an xarray dataaray by a simple try & except block.
#Returns 0 if false, 1 if true.
    
def check_xarray(arr):
    
    try:
        temp_val = arr.values
        return 1
    except:
        return 0