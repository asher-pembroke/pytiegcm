from netCDF4 import Dataset
import numpy as np

from collections import defaultdict

from collections import namedtuple
from scipy.spatial import Delaunay
from numpy import isclose
import scipy 
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy import interpolate
from collections import OrderedDict
import time
from util import *
from util import boundary_conditions

def geo_to_spherical(point3D, R_e = 6.371008e8):
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

def index_range(a, v, lr = 'right'):
    """return indices bounding value in range (a_i, a_i+1] or [a_i, a_i+1)
    ex: index_range(a,a[i], 'left') == [a[i-1], a[i]]
    """
    side = np.searchsorted(a, v, side = lr)
    return [side - 1, side]

def to_range(x, start, limit):
	"""wraps x into range [start, limit]"""
	return start + (x - start) % (limit - start)
   
class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class TimeInterpolator(object):
	def __init__(self, interpolators):
		"""takes an ordered dictionary of interpolators keyed by time"""
		# print 'creating TimeInterpolator'
		self.interpolators = interpolators
		t0, t1 = interpolators.keys()
		self.t0 = t0
		self.t1 = t1
	def __call__(self, point, time):
		w = (time - self.t0)/(self.t1 - self.t0)
		v0 = self.interpolators[self.t0](point)
		v1 = self.interpolators[self.t1](point)
		return v0*(1.0-w) + v1*w

class TimeInterpolator2D(object):
	def __init__(self, interpolators):
		"""takes an ordered dictionary of interpolators keyed by time"""
		# print 'creating TimeInterpolator'
		self.interpolators = interpolators
		t0, t1 = interpolators.keys()
		self.t0 = t0
		self.t1 = t1
	def __call__(self, point, time):
		w = (time - self.t0)/(self.t1 - self.t0)
		v0 = self.interpolators[self.t0](*point)
		v1 = self.interpolators[self.t1](*point)
		return float(v0*(1.0-w) + v1*w)


class TIEGCM(object):
	def __init__(self, filename):
		self.rootgrp = Dataset(filename, 'r')
		self.lat = np.concatenate(([-90], np.array(self.rootgrp.variables['lat']), [90]))
		self.lon = np.array(self.rootgrp.variables['lon'])

		self.boundary_set = []
		self.wrapped = []

		self.wrap_longitude()

		self.ilev = np.array(self.rootgrp.variables['ilev'])

		self.ilev_, self.lat_, self.lon_ = scipy.meshgrid(self.ilev, self.lat, self.lon, indexing = 'ij')

		self.ut = np.array(self.rootgrp.variables['ut'])

		self.z_scale = 100000

		self.set_interpolators()

		self.set_3d_boundary_conditions()
		self.wrap_3d_variables()
		
		self.set_points2D() # for testing

		self.set_variable_boundary_condition('Z') #prior to wrapping to avoid double counting
		self.wrap_variable('Z')
		self.z = np.array(self.rootgrp.variables['Z']) # Z is "geopotential height" -- use geometric height ZG?

		self.fill_value = 1.0

	def get_variable_unit(self, variable_name):
		return self.rootgrp.variables[variable_name].units

	def list_2d_variables(self):
	    return [k for k in self.rootgrp.variables if len(self.rootgrp.variables[k].shape) == 3]

	def list_3d_variables(self):
	    return [k for k in self.rootgrp.variables if len(self.rootgrp.variables[k].shape) == 4]

	def set_variable_boundary_condition(self, variable_name):
		"""If the boundary condition has not been set, set it"""
		if variable_name not in self.boundary_set:
			bc = boundary_conditions[variable_name]
			if bc == 'average':
				self.rootgrp.variables[variable_name] = average_longitude(self.rootgrp.variables[variable_name])
				self.boundary_set.append(variable_name)
			elif bc == 'extend':
				self.rootgrp.variables[variable_name] = copy_longitude(self.rootgrp.variables[variable_name])
				self.boundary_set.append(variable_name)
			else:
				pass

	def set_3d_boundary_conditions(self):
		"""Wrap all 3D variables by longitude"""
		for k, variable in self.rootgrp.variables.items():
			if len(variable.shape) == 3: # 3D variable
				self.set_variable_boundary_condition(k)
				

	def wrap_variable(self, variable_name):
		if variable_name not in self.wrapped:
			variable = self.rootgrp.variables[variable_name]
			if len(variable.shape) == 3: # 3D variable
				self.rootgrp.variables[variable_name] = np.concatenate((variable, variable[:,:,0:1]), axis = 2)
				self.wrapped.append(variable_name)
			if len(variable.shape) == 4: # 4D variable
				self.rootgrp.variables[variable_name] = np.concatenate((variable, variable[:,:,:,0:1]), axis = 3)
				self.wrapped.append(variable_name)


	def wrap_3d_variables(self):
		"""Wrap all 3D variables by longitude"""
		for variable_name, variable in self.rootgrp.variables.items():
			if len(variable.shape) == 3: # 3D variable
				self.wrap_variable(variable_name)


	def wrap_longitude(self):
		"""puts longitude from [min, max] to [min, wrapped(min)]"""
		self.longitude_min = self.lon.min()
		self.longitude_max = to_range(self.longitude_min, 0, 360)
		self.lon = np.hstack((self.lon, self.longitude_max))

	def set_interpolators(self):
		self.interpolators = keydefaultdict(self.create_interpolator)
		self.time_interpolators = keydefaultdict(self.create_3Dtime_interpolator)
		self.time_interpolators2D = keydefaultdict(self.create_2Dtime_interpolator)


	def set_points2D(self):
		"""allows for validating interpolation of 2D (lat,lon) variables"""
		points = Point2DT(*scipy.meshgrid(self.ut, self.lat, self.lon, indexing = 'ij'))
		self.rootgrp.variables['latitude'] = points.latitude
		self.rootgrp.variables['longitude'] = points.longitude

	def get_column_slicer_4D(self, point):
	    t_indices = index_range(self.ut, point.time)
	    t_slice = slice(t_indices[0], t_indices[1]+1)
	    
	    lat_indices = index_range(self.lat, point.latitude)
	    lat_slice = slice(lat_indices[0], lat_indices[1]+1)

	    wrapped_lon = to_range(point.longitude, self.longitude_min, self.longitude_max)
	    lon_indices = index_range(self.lon, wrapped_lon)
	    lon_slice = slice(lon_indices[0], lon_indices[1]+1)
	    return ColumnSlice4D(t_slice, slice(None), lat_slice, lon_slice)

	def get_slicer_1D(self, point):
	    t_indices = index_range(self.ut, point.time)
	    t_slice = slice(t_indices[0], t_indices[1]+1)
	    return t_slice

	def get_column_slice_3D(self, point):
	    lat_indices = index_range(self.lat, point.latitude)
	    lat_slice = slice(lat_indices[0], lat_indices[1]+1)
	    
	    lon_indices = index_range(self.lon, point.longitude)
	    lon_slice = slice(lon_indices[0], lon_indices[1]+1)
	    return ColumnSlice3D(slice(None), lat_slice, lon_slice)

	def get_3D_column(self, column_slice):
		lat_column = self.lat_[column_slice[1:]] #3d
		lon_column = self.lon_[column_slice[1:]] #3d
		z_column = self.z[column_slice] #4d
		return z_column, lat_column, lon_column


	def get_delaunay_3D(self, z_column, lat_column, lon_column, time_index = 0):
		points = zip(z_column[time_index].ravel()/self.z_scale, lat_column.ravel(), lon_column.ravel())
		return Delaunay(points)

	def get_delaunay_3D_cartesian(self, z_column, lat_column, lon_column, time_index = 0):
		z = z_columns[time_index].ravel()
		points = zip(z_column[time_index].ravel()/self.z_scale, lat_column.ravel(), lon_column.ravel())
		return Delaunay(points)


	def get_slice_key_4D(self, column_slice, variable_name):
		reduced = (s.__reduce__() for s in column_slice)
		return Slice_key4D(*reduced, variable = variable_name)

	def get_slice_key_3D(self, column_slice, variable_name, time_index):
		time_slice = slice(time_index, time_index+1, None).__reduce__()
		reduced = (s.__reduce__() for s in column_slice)
		return Slice_key4D(time_slice, *reduced, variable = variable_name)

	def get_3D_interpolator(self, slice_key, time_index):
		column_slice = ColumnSlice4D(*tuple(slice(*s[1]) for s in slice_key[:-1]))
		z_column, lat_column, lon_column = self.get_3D_column(column_slice)
		try:
			delaunay = self.get_delaunay_3D(z_column, lat_column, lon_column, time_index = time_index)
		except:
			print 'time index:', time_index
			print slice_key.__repr__()
			print column_slice.__repr__()
			raise
		self.set_variable_boundary_condition(slice_key.variable)
		self.wrap_variable(slice_key.variable)
		variable = np.array(self.rootgrp[slice_key.variable])[column_slice][time_index].ravel()

		try:
			return LinearNDInterpolator(delaunay, variable, fill_value = self.fill_value)
		except:
			print '\n', slice_key
			print column_slice
			print delaunay.points.shape, variable.shape
			print delaunay.points[:4,:]
			print z_column.shape
			raise

	def get_3D_interpolator_cartesian(self, slice_key, time_index):
		column_slice = ColumnSlice4D(*tuple(slice(*s[1]) for s in slice_key[:-1]))
		z_column, lat_column, lon_column = self.get_3D_column(column_slice)
		try:
			delaunay = self.get_delaunay_3D(z_column, lat_column, lon_column, time_index = time_index)
		except:
			print 'time index:', time_index
			print slice_key.__repr__()
			print column_slice.__repr__()
			raise
		self.set_variable_boundary_condition(slice_key.variable)
		self.wrap_variable(slice_key.variable)
		variable = np.array(self.rootgrp[slice_key.variable])[column_slice][time_index].ravel()

		try:
			return LinearNDInterpolator(delaunay, variable, self.fill_value)
		except:
			print '\n', slice_key
			print column_slice
			print delaunay.points.shape, variable.shape
			print delaunay.points[:4,:]
			print z_column.shape
			raise

	def get_2D_interpolator(self, slice_key, time_index):
		"""Create a 2D interpolator for this variable"""
		time_slicer = slice(*slice_key.time[1])
		variable = self.rootgrp[slice_key.variable]
		return RectBivariateSpline(self.lat, self.lon, variable[time_slicer][time_index])

	def create_interpolator(self, slice_key, time_index = 0):
		"""takes a slice key and returns an interpolator object"""
		return self.get_3D_interpolator(slice_key, time_index)

	def create_3Dtime_interpolator(self, slice_key):
		"""takes a 4d slice key and returns a TimeInterpolator object"""
		times = self.ut[slice(*slice_key.time[1])]
		try:
			interpolators = OrderedDict([(t, self.get_3D_interpolator(slice_key, i)) for i,t in enumerate(times)])
		except:
			print 'create_3Dtime_interpolator - times', times
			raise
		return TimeInterpolator(interpolators)

	def create_2Dtime_interpolator(self, slice_key):
		"""takes a 3d slice key and returns a TimeInterpolator object
		slice key only needs to depend on time and variable, since lat and lon are fixed
		"""
		times = self.ut[slice(*slice_key.time[1])]
		interpolators = OrderedDict([(t, self.get_2D_interpolator(slice_key, i)) for i,t in enumerate(times)])
		return TimeInterpolator2D(interpolators)

	def interpolate_4D_point(self, point, variable_name, time):
		column_slicer = self.get_column_slicer_4D(Point4D(time, *point))
		slice_key = self.get_slice_key_4D(column_slicer, variable_name)

		p = point.height/self.z_scale, point.latitude, point.longitude
		return self.interpolators[slice_key](p)		

	def time_interpolate(self, point, variable_name, time):
		column_slicer = self.get_column_slicer_4D(Point4D(time, *point))
		slice_key = self.get_slice_key_4D(column_slicer, variable_name)
		p = point[0]/self.z_scale, point[1], to_range(point[2], self.longitude_min, self.longitude_max)
		return self.time_interpolators[slice_key](p, time)

	def time_interpolate_2D(self, point, variable_name, time):
		time_slicer = self.get_slicer_1D(Point2DT(time, *point))
		slice_key = Slice_keyTV(time_slicer.__reduce__(), variable_name)
		point = point[0], to_range(point[1],self.longitude_min, self.longitude_max)
		return self.time_interpolators2D[slice_key](point, time)

	def interpolate_3D_point(self, point, variable_name, time_index):
		slice_key = self.get_slice_key_3D(self.get_column_slice_3D(point), variable_name, time_index)
		t = self.ut[slice(*slice_key.time[1])]
		p = point.height/self.z_scale, point.latitude, point.longitude
		return self.interpolators[slice_key](p)		

z_test = 39005780. # a mid range test height in cm 

test_file = "sample_data/jasoon_shim_052317_IT_10/out/s001.nc"

def test_3D_column():
	tiegcm = TIEGCM(test_file)
	point = Point4D((tiegcm.ut[0]+tiegcm.ut[1])/2, 128.14398737, 87.,  170.  )	
	columns = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))
	assert columns[0].shape == (2, len(tiegcm.ilev), 2, 2)
	for i in range(1,3):
		assert columns[i].shape == (len(tiegcm.ilev), 2, 2)

def test_column_slice():
	tiegcm = TIEGCM(test_file)
	point = Point4D((tiegcm.ut[0]+tiegcm.ut[1])/2, 128.14398737, 87.,  171.  )
	column = tiegcm.get_column_slicer_4D(point)

	assert point.latitude > tiegcm.lat[column.latitude][0]
	assert point.latitude < tiegcm.lat[column.latitude][1]
	assert point.longitude > tiegcm.lon[column.longitude][0]
	assert point.longitude < tiegcm.lon[column.longitude][1]

	

def test_Delaunay_height():
	tiegcm = TIEGCM(test_file)
	point = Point4D(3.5, z_test,  87.,  170. )
	z_column, lat_column, lon_column = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))
	delaunay = tiegcm.get_delaunay_3D(z_column, lat_column, lon_column, time_index = 0)

	z = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))[0][0]
	
	scaled_point = z_test/tiegcm.z_scale, point[2], point[3]
	linear_interpolator = LinearNDInterpolator(delaunay, z.ravel()/tiegcm.z_scale, fill_value = tiegcm.fill_value)
	
	assert isclose(scaled_point[0], linear_interpolator(scaled_point))


def test_custom_time_interpolator():
	class TestInterpolator(object):
		def __init__(self, value):
			self.value = value
		def __call__(self, point):
			return self.value

	interpolators = OrderedDict([(0.0, TestInterpolator(5.0)),
								(1.0, TestInterpolator(4.0))])

	time_interpolator = TimeInterpolator(interpolators)
	assert time_interpolator(None, .5) == 4.5


def test_time_interpolate():
	## This should replicate the delaunay test above
	tiegcm = TIEGCM(test_file)
	point = Point3D(z_test,  87.,  170. )
	variable_name ='Z'

	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)
	variable_name = 'NE'
	print variable_name, point, tiegcm.time_interpolate(point, variable_name, 3.5)

def test_time_interpolate_edge():
	tiegcm = TIEGCM(test_file)
	variable_name ='Z'

	point = Point3D(z_test,  87.,  tiegcm.lon.min() )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  87.,  tiegcm.lon.min() - 1 )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  87.,  tiegcm.lon.max() )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  87.,  tiegcm.lon.max() + 1)
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)


def test_time_interpolate_2D():
	tiegcm = TIEGCM(test_file)
	point = Point2D(87.,  170. )
	variable_name = 'EFLUX'
	print variable_name, point, tiegcm.time_interpolate_2D(point, variable_name, 3.5)
	variable_name = 'latitude'
	assert isclose(point.latitude, tiegcm.time_interpolate_2D(point, variable_name, 3.5))

def test_time_interpolate_2D_edge():
	tiegcm = TIEGCM(test_file)
	variable_name = 'longitude' # longitude has not been converted to [longitude_min, longitude_max]

	point = Point2D(87.,  tiegcm.lon.min() )
	assert isclose(point.longitude, tiegcm.time_interpolate_2D(point, variable_name, 3.5))
	point = Point2D(87.,  tiegcm.lon.max() + 1 )
	assert isclose(to_range(point[1],tiegcm.longitude_min, tiegcm.longitude_max), 
							tiegcm.time_interpolate_2D(point, variable_name, 3.5))
	
def test_interpolate_3D():
	tiegcm = TIEGCM(test_file)
	point = Point3D(z_test,  87.,  170. )
	result = tiegcm.interpolate_3D_point(point, 'Z', 5)
	expected = np.array(z_test)
	assert isclose(result, expected)

def test_units():
	tiegcm = TIEGCM(test_file)
	assert tiegcm.get_variable_unit('NE') == 'cm-3'

def test_variable_list():
	tiegcm = TIEGCM(test_file)

	varlist_2d = tiegcm.list_2d_variables()
	for variable_name in varlist_2d:
		assert len(tiegcm.rootgrp.variables[variable_name].shape) == 3

	varlist_3d = tiegcm.list_3d_variables()
	for variable_name in varlist_3d:
		assert len(tiegcm.rootgrp.variables[variable_name].shape) == 4

def test_time_interpolate_speed():
	## This should replicate the delaunay test above
	tiegcm = TIEGCM(test_file)
	npoints = 100
	print 'speed test started with', npoints, 'points'
	t = time.time()
	expected = np.array(z_test)
	rand_seed = 0 # gives qhull facet errors Looks like edge issue
	# rand_seed = 1 # gives qhull index errors
	np.random.seed(rand_seed)
	try:
		for lat, lon in zip( np.random.uniform(-80, 80, npoints), np.random.uniform(-180, 180, npoints)):
			point = Point3D(z_test,  lat,  lon )
			variable_name ='Z'
			result = tiegcm.time_interpolate(point, variable_name, 3.5)
			assert np.isclose(result, expected)	
	except:
		print 'test failed at', point
		print 'ranges:', Point2D(	[str(len(tiegcm.lat))+":", tiegcm.lat.min(), tiegcm.lat.max()],
									[str(len(tiegcm.lon))+":", tiegcm.lon.min(), tiegcm.lon.max()])
		raise
	dt = time.time() - t
	print 'speed test finished', dt, 'seconds', dt/npoints, '[sec/point]'
	# variable_name = 'NE'
	# print variable_name, point, tiegcm.time_interpolate(point, variable_name, 3.5)

def test_time_interpolate_pole():
	tiegcm = TIEGCM(test_file)
	variable_name ='Z'

	point = Point3D(z_test,  88.,  tiegcm.lon.min() )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  88.,  tiegcm.lon.min() - 1 )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  88.,  tiegcm.lon.max() )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)

	point = Point3D(z_test,  88.,  tiegcm.lon.max() + 1)
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_test)
	assert isclose(result, expected)


def test_high_altitude():
	tiegcm = TIEGCM(test_file)
	variable_name ='Z'
	z_max = 1.1*tiegcm.z.max()
	print 'test_high_altitude:', z_max

	point = Point3D(z_max,  88.,  tiegcm.lon.min() )
	result = tiegcm.time_interpolate(point, variable_name, 3.5)
	expected = np.array(z_max)
	assert isclose(result, tiegcm.fill_value)



