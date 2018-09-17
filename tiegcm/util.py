import numpy as np
from collections import namedtuple, defaultdict
import pandas as pd
import numpy.ma as ma

R_e = 6.371008e8 #cm

k = 1.38064852e-23 # [J/K]
m_p = 1.6726e-27 # [kg]
g_0 = 9.807 # [m/s^2]

def gravity(h):
    """For h in same units as R_e"""
    return g_0*pow(R_e/(R_e + h), 2)

def scale_height(T, height, molecular_mass):
	'''Scale height in meters, assuming mks'''
	return k*T/(gravity(height)*molecular_mass*m_p)

def to_date(year, doy):
	return pd.to_datetime(str(year) + str(doy),format ='%Y%j')


def time_in_interval(time, interval): 
	'''time in [t0, t1)'''
	return interval[0] <= time < interval[1]

def datetime_to_epoch(time):
    return (time - pd.datetime(1970,1,1)).total_seconds()

Point4D = namedtuple("Point4D", ['time','height', 'latitude','longitude'])
Point3D = namedtuple("Point3D", ['height', 'latitude','longitude'])
Point2D = namedtuple("Point2D", ['latitude', 'longitude'])
Point2DT = namedtuple("Point2D", ['time', 'latitude', 'longitude'])
ColumnSlice4D = namedtuple("ColumnSlice4D", ['time', 'height', 'latitude', 'longitude'])
ColumnSlice3D = namedtuple("ColumnSlice3D", ['height', 'latitude', 'longitude'])
Slice3D = namedtuple("Slice3D", ['time', 'latitude', 'longitude'])
Slice_key4D = namedtuple("Slice_key4D", ["time", "height", "latitude", "longitude", "variable"])
Slice_key3D = namedtuple("Slice_key3D", ["height", "latitude", "longitude", "variable"])
Slice_key2DT = namedtuple("Slice_key2DT", ['time', 'latitude', 'longitude', 'variable'])
Slice_keyTV = namedtuple("Slice_keyTV", ['time', 'variable'])

Point3DCartesian = namedtuple("Point3DCartesian", ['x','y','z'])
Point3DSpherical = namedtuple("Point3DSpherical", ['r','theta','phi'])


boundary_conditions = dict( TEC = 'average',
                            QJOULE_INTEG = 'average',
                            EFLUX = 'average',
                            HMF2 = 'average',
                            NMF2 = 'average',
                            TLBC = 'average',
                            ULBC = 'extend', #?
                            VLBC = 'extend', #?
                            TLBC_NM = 'average', #?
                            ULBC_NM = 'extend', #?
                            VLBC_NM = 'extend', #?
                            TN = 'average',
                            UN = 'extend',
                            VN = 'extend',
                            O1 = 'average',
                            NO = 'average',
                            N4S = 'average',
                            HE = 'average',
                            NE = 'average',
                            TE = 'average',
                            TI = 'average',
                            O2 = 'average',
                            O2P_ELD = 'average', #?
                            OMEGA = 'average',
                            POTEN = 'average',
                            VI_ExB = 'extend',
                            UI_ExB = 'extend',
                            WI_ExB = 'average', #?
                            OP = 'average',
                            N2P_ELD = 'average',
                            NPLUS = 'average',
                            NOP_ELD = 'average',
                            SIGMA_PED = 'average',
                            SIGMA_HAL = 'average',
                            DEN = 'average',
                            QJOULE = 'average',
                            Z = 'average',
                            ZG = 'average',
                            O_N2 = 'average',
                            N2D_ELD = 'average',
                            O2N = 'average',
                            N2N = 'average',
                            ZMAG = 'average',                           
                          )

def group_dimensional(rootgrp, verbose = False):
	dimensional = defaultdict(list)
	for k,v in rootgrp.variables.items():
		for i in range(5):
			if len(v.shape) == i:
				dimensional[str(i)+'-d'].append(k)
	
	if verbose:
		for k,v in dimensional.items():
			print k
			print describe(rootgrp, dimensional[k])
	return dimensional

def describe(rootgrp, 
			 variables = ['time', 'ilev','lat', 'lon'], 
			 attributes = ['units', 'long_name', 'shape']):
	results = pd.DataFrame(columns = attributes + ['min', 'max'])
	for v in variables:
		array = rootgrp.variables[v]
		attrs = []
		for attr in attributes:
			try:
				attribute = array.__getattribute__(attr)
			except:
				attribute = None
			attrs.append(attribute)
		try:
			attrs.append(array.__array__().min())
			attrs.append(array.__array__().max())
		except:
			attrs.append(None)
			attrs.append(None)
		results = results.append(pd.Series(attrs, index = attributes + ['min', 'max'], name = v))
	return results

def geo_to_spherical(point3D):
	r = point3D.height + R_e
	theta = np.pi*(90 - point3D.latitude)/180
	phi = np.pi*(180 + point3D.longitude)/180
	return Point3DSpherical(r, theta, phi)

def spherical_to_cartesian(point3D):
	x = point3D.r*np.sin(point3D.theta)*np.cos(point3D.phi)
	y = point3D.r*np.sin(point3D.theta)*np.sin(point3D.phi)
	z = point3D.r*np.cos(point3D.theta)
	return Point3DCartesian(x, y, z)

def geo_to_cartesian(point3D, R_e = 6.371008e8):
	return spherical_to_cartesian(geo_to_spherical(point3D, R_e))

def copy_longitude(a): 
	"""Copy longitude values, returning size nlat+2,
	assumes longitude is last index"""
	if len(a.shape) == 4: #4d variable
		ntime, nilev, nlat, nlon = a.shape
		south = a[:,:,0,:]
		north = a[:,:,-1,:]
		result = np.insert(a, 0, south, axis = 2)
		result = np.insert(result, result.shape[2], north, axis = 2)

	else: #3d variable
		ntime, nlat, nlon = a.shape
		south = a[:,0,:]
		north = a[:,-1,:]

		result = np.insert(a, 0, south, axis = 1)
		result = np.insert(result, result.shape[1], north, axis = 1)
	return result

def fill_masked(a):
	if a[:,-1,:,:].mask.all():
		a[:,-1,:,:] = a[:,-2,:,:].data
	return a.filled()

def average_longitude(a, verbose = False): 
	"""Averages longitude values, returning size nlat+2,
	assumes longitude is last index"""
	if len(a.shape) == 4: #4d variable
		ntime, nilev, nlat, nlon = a.shape
		south_avg = np.mean(a[:,:,0,:].T, axis = 0).T
		south_avg = south_avg.repeat(nlon, axis = 1).reshape((ntime, nilev, nlon))
		if verbose:
			print 'south min, max', south_avg.min(), south_avg.max()

		north_avg = np.mean(a[:,:,-1,:].T, axis = 0).T
		north_avg = north_avg.repeat(nlon, axis = 1).reshape((ntime, nilev, nlon))
		if verbose:
			print 'north min, max', north_avg.min(), north_avg.max()

		result = np.insert(a, 0, south_avg, axis = 2)
		if verbose:
			print 'after south insert, min, max', result.min(), result.max()
		result = np.insert(result, result.shape[2], north_avg, axis = 2)
	else: #3d variable
		ntime, nlat, nlon = a.shape
		south_avg = np.mean(a[:,0,:], axis = 1)
		south_avg = south_avg.repeat(nlon).reshape((ntime, nlon))

		north_avg = np.mean(a[:,-1,:], axis = 1)
		north_avg = north_avg.repeat(nlon).reshape((ntime, nlon))

		result = np.insert(a, 0, south_avg, axis = 1)
		result = np.insert(result, result.shape[1], north_avg, axis = 1)
	return result