from netCDF4 import Dataset
import numpy as np
import os
from collections import defaultdict

from collections import namedtuple
from scipy.spatial import Delaunay
from numpy import isclose
import scipy 
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.spatial import kdtree
from scipy import interpolate
from collections import OrderedDict
import time
from util import *
from util import boundary_conditions, fill_masked
import numpy.ma as ma

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
		try:
			t0, t1 = interpolators.keys()
		except:
			if len(interpolators.keys()) == 1:
				t0 = interpolators.keys()[0]
				t1 = t0
			else:
				print 'could not construct TimeInterpolator with keys',
				print interpolators.keys()
				raise
		self.t0 = t0
		self.t1 = t1
	def __call__(self, point, time):
		v0 = self.interpolators[self.t0](point)
		if self.t0 != self.t1:
			w = (time - self.t0)/(self.t1 - self.t0)
			v1 = self.interpolators[self.t1](point)
			return v0*(1.0-w) + v1*w
		else:
			return v0

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
	def __init__(self, filename, outermost_layer = -1):
		# print 'initializing tiegcm'
		self.rootgrp = Dataset(filename, 'r')
		self.lat = np.concatenate(([-90], np.array(self.rootgrp.variables['lat']), [90]))
		self.lon = np.array(self.rootgrp.variables['lon'])

		self.boundary_set = []
		self.wrapped = []

		self.set_outermost_layer(outermost_layer)

		self.wrap_longitude()

		self.ilev = np.array(self.rootgrp.variables['ilev'])[:self.outermost_layer]

		self.ilev_, self.lat_, self.lon_ = scipy.meshgrid(self.ilev, self.lat, self.lon, indexing = 'ij')

		self.ut = np.array(self.rootgrp.variables['ut'])
		if self.ut[-1] < self.ut[0]:
			self.ut[-1] += 24

		self.z_scale = 100000

		self.set_interpolators()

		self.set_3d_boundary_conditions()
		self.wrap_3d_variables()
		
		self.set_points2D() # for testing

		self.set_variable_boundary_condition('Z') #prior to wrapping to avoid double counting
		self.wrap_variable('Z')
		self.z = np.array(self.rootgrp.variables['Z'])[:,:self.outermost_layer,:,:] # Z is "geopotential height" -- use geometric height ZG?
		self.high_altitude_trees = dict()
		self.fill_value = np.nan

	def set_outermost_layer(self, layer = None):
		if layer is None:
			self.outermost_layer = len(self.rootgrp.variables['ilev'])
		else:
			self.outermost_layer = layer

	def get_variable_unit(self, variable_name):
		return self.rootgrp.variables[variable_name].units

	def list_2d_variables(self):
		return [k for k in self.rootgrp.variables if len(self.rootgrp.variables[k].shape) == 3]

	def list_3d_variables(self):
		return [k for k in self.rootgrp.variables if len(self.rootgrp.variables[k].shape) == 4]

	def get_outer_boundary_kdtree(self, time_index):
		"""Initialize a spatial index for the upper boundary of the model"""
		try:
			return self.high_altitude_trees[time_index]
		except:
			z = self.z[time_index, -1, :, :]
			lat = self.lat_[-1]
			lon = self.lon_[-1]

			outer_points = np.array(zip(z.ravel()/z.max(), lat.ravel(), lon.ravel()))
			tree = scipy.spatial.KDTree(outer_points)
			self.high_altitude_trees[time_index] = tree
			return self.high_altitude_trees[time_index]

	def set_variable_boundary_condition(self, variable_name, verbose = False):
		"""If the boundary condition has not been set, set it"""
		if variable_name not in self.boundary_set:
			bc = boundary_conditions[variable_name]
			variable = self.rootgrp.variables[variable_name].__array__()
			if verbose:
				print 'setting boundary condition:', variable_name, bc, type(variable)
			if bc == 'average':
				if type(variable) == ma.MaskedArray:
					variable = fill_masked(variable)
					average = average_longitude(variable, verbose)
				else:
					average = average_longitude(variable, verbose)
				if verbose:
					print 'average min,max', average.min(), average.max()
				self.rootgrp.variables[variable_name] = average
				self.boundary_set.append(variable_name)
			elif bc == 'extend':
				self.rootgrp.variables[variable_name] = copy_longitude(variable)
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
			variable = self.rootgrp.variables[variable_name].__array__()
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
		return Delaunay(points, incremental = True)

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
			# import ipdb; ipdb.set_trace()
			raise
		self.set_variable_boundary_condition(slice_key.variable)
		self.wrap_variable(slice_key.variable)
		variable = np.array(self.rootgrp[slice_key.variable])[:,:self.outermost_layer,:,:][column_slice][time_index].ravel()

		try:
			return LinearNDInterpolator(delaunay, variable, fill_value = self.fill_value)
		except:
			print '\n', slice_key
			print column_slice
			print delaunay.points.shape, variable.shape
			print delaunay.points[:4,:]
			print z_column.shape
			# import ipdb; ipdb.set_trace()
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
			# import ipdb; ipdb.set_trace()
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
			# import ipdb; ipdb.set_trace()
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
		# import ipdb; ipdb.set_trace()
		result = self.time_interpolators[slice_key](p, time)
		if np.isnan(result):
			return self.time_interpolate_high_altitude(point, variable_name, time)
		else:
			return result

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


	def get_coordinate_indices(self, vertices, target_shape):
		return np.array(zip(*np.unravel_index(vertices, target_shape)))

	def interpolate_high_altitude(self, p, variable_name, time_index, return_variable = False):
		self.set_variable_boundary_condition(variable_name) #prior to wrapping to avoid double counting
		self.wrap_variable(variable_name)

		tree = self.get_outer_boundary_kdtree(time_index) #store this in a dict

		distances, vertices = tree.query(p, p = 1, k = 3) # p = 1 for Manhattan distance!

		coord_indices = self.get_coordinate_indices(vertices, self.z[time_index, -1, :, :].shape)
		lat_indices, lon_indices = coord_indices[:,0], coord_indices[:,1]

		try:
			variable = self.rootgrp.variables[variable_name][:,:self.outermost_layer,:,:][time_index, -1, lat_indices, lon_indices].ravel()
			lnd = LinearNDInterpolator(tree.data[vertices][:,1:], variable)
		except:
			print 'problem in interpolate high altitude:'
			print self.rootgrp.variables[variable_name].shape
			print self.rootgrp.variables['Z'].shape
			# import ipdb; ipdb.set_trace()
			raise
		if not return_variable:
			return float(lnd(p[1:]))
		else:
			return float(lnd(p[1:])), variable, lat_indices, lon_indices

	def time_interpolate_high_altitude(self, p, variable_name, time, return_variables = False):
		column_slicer = self.get_column_slicer_4D(Point4D(time, *p))
		t0 = column_slicer.time.start
		t1 = t0 + 1
		# import ipdb; ipdb.set_trace()

		if not return_variables:
			r0 = self.interpolate_high_altitude(p, variable_name, t0)
			try:
				r1 = self.interpolate_high_altitude(p, variable_name, t1)
				return np.interp(time, [t0, t1], [r0, r1])
			except:
				return r0
		else:
			r0, variables0, lat_indices0, lon_indices0 = self.interpolate_high_altitude(p, variable_name, t0, return_variables)
			try:
				r1, variables1, lat_indices1, lon_indices1 = self.interpolate_high_altitude(p, variable_name, t1, return_variables)
			except:
				return r0, variables0, lat_indices0, lon_indices0

		return np.interp(time, [t0, t1], [r0, r1]), (variables0, variables1), (lat_indices0, lat_indices1), (lon_indices0, lon_indices1)

	def molecular_mass_density(self, point, variable_name, time, density_tot):
		mmr = self.time_interpolate(point, variable_name, time)
		return mmr*density_tot/(1+mmr)

	def mass_density(self, point, time):
		'''Total extrapolated neutral density in g/cm^3'''
		# Todo: have it consistent with only 3 species above and below max alt
		# Fix N2
		density_tot = self.time_interpolate(point, 'DEN', time) # total mass density [g/cm^3]

		den_O1 = self.molecular_mass_density(point, 'O1', time, density_tot)
		den_O2 = self.molecular_mass_density(point, 'O2', time, density_tot)

		N2_n = self.time_interpolate(point, 'N2N', time)
		den_N2 = N2_n*m_p*28


		h_bndy = self.time_interpolate(point, 'Z', time)

		if point.height > h_bndy: 

			T = self.time_interpolate(point, 'TN', time) # neutral temperature [K]

			h_O1 = scale_height(T, point.height, 16)*100
			den_O1 *= np.exp((h_bndy - point.height)/h_O1)

			h_O2 = scale_height(T, point.height, 32)*100
			den_O2 *= np.exp((h_bndy - point.height)/h_O2)

			h_N2 = scale_height(T, point.height, 28)*100
			den_N2 *= np.exp((h_bndy - point.height)/h_N2)

		rho = den_O1 + den_O2 + den_N2
		return rho

	def density(self, xlat, xlon, xalt, time):
		return self.mass_density(Point3D(xalt, xlat, xlon), time)

	def get_time_range(self):
		start_year = self.rootgrp.variables['year'][0]
		start_day = self.rootgrp.variables['day'][0]
		date = to_date(start_year, start_day)
		start_ut = self.ut[0]
		end_ut = self.ut[-1]
		start = date + pd.Timedelta(hours = start_ut)
		end = date + pd.Timedelta(hours = end_ut)
		if end < start:
			raise IOError
		return start, end


class Model_Manager(TIEGCM):
	"""Class to manage time interpolation for multiple files
	Since files begin at +20 minutes and end on the hour, we need to handle the case
	where a query lies between the time ranges of successive files.
	"""
	def __init__(self, directory = None, outermost_layer = -1):
		self.outermost_layer = outermost_layer
		if directory is not None:
			self.files = self.get_files(directory)
			self.set_file_times()
			
		self.last_interval = self.file_times[self.files[0]]
		
		TIEGCM.__init__(self, self.files[0], outermost_layer = self.outermost_layer)
		print 'model manager initialized'
		print 'current time', self.file_times[self.files[0]]


	def file_interpolate(self, xlat, xlon, xalt, ut):
		print 'interpolating between files'
		print xlat, xlon, xalt, ut
		
	def get_files(self, directory = None, file_list = None, file_type = ".nc", match_str = "s", 
					start = 0, stop = None, **kwargs):
		"""Walks through subdirectories and retrieves 's*.nc' files sorted by filename"""
		nc_files = []                 
		if directory is not None:
			for path, subdirs, files in os.walk(os.path.expanduser(directory)):
				for name in files:
					if name.endswith(file_type) and match_str in name: 
						nc_files.append(os.path.join(path,name))
		elif file_list is not None:
			for name in file_list:
				if name.endswith(file_type) and match_str in name: 
					nc_files.append(name)
		else:
			raise IOError("Need to specify a directory or file listing")
		return nc_files

	def set_file_times(self):
		self.file_times = OrderedDict()
		for f in self.files:
			TIEGCM.__init__(self, f)
			self.file_times[f] = self.get_time_range()
			self.rootgrp.close()
			
	def get_file_for_time(self, time):
		for filename, interval in self.file_times.items(): 
			if time_in_interval(time, interval):
				return filename

	
	def time_to_ut(self, timestamp):
		time = timestamp.time()
		return time.hour + time.minute/60. + time.second/3600.

	def time_in_range(self, time):
		if self.file_times[self.files[0]][0] < time < self.file_times[self.files[-1]][1]:
			for i in range(len(self.files)-1):
				if time < self.file_times[self.files[i+1]][0]:
					return self.files[i], self.files[i+1]
		else:
			return None


	def density(self, xlat, xlon, xalt, gregorian_string, raise_errors = False, debug = False):
		if debug:
			print xlat, xlon, xalt, gregorian_string, raise_errors
		try:
			time = pd.to_datetime(gregorian_string)
			if not time_in_interval(time, self.last_interval):
				try:
					self.rootgrp.close()
				except:
					pass
				filename = self.get_file_for_time(time) # look for time in file date ranges
				if filename is not None: # found time in file
					self.last_interval = self.file_times[filename] # so we can jump straight to interpolation next time
					TIEGCM.__init__(self, filename, self.outermost_layer)
				else: # file not in time ranges
					closest_files = self.time_in_range(time) # see if time is in data gap between files
					if debug:
						print closest_files
					if closest_files is not None: # linearly interpolate between these two
						t0 = self.file_times[closest_files[0]][1]
						t1 = self.file_times[closest_files[1]][0]

						TIEGCM.__init__(self, closest_files[0], self.outermost_layer)
						tiegcm1 = TIEGCM(closest_files[1], self.outermost_layer)

						d0 = TIEGCM.density(self, xlat, xlon, xalt, self.time_to_ut(t0))
						d1 = tiegcm1.density(xlat, xlon, xalt, self.time_to_ut(t1))
						w = (time - t0)/(t1 - t0)
						return d0*(1.0-w) + d1*w
					else: # outside all available data
						raise ValueError('Could not find time in files')
			return TIEGCM.density(self, xlat, xlon, xalt, self.time_to_ut(time))
		except:
			if raise_errors:
				print 'TIEGCM error at xlat: {}, xlon: {}, xalt: {}, gregorian str: {}'.format(xlat, xlon, xalt, gregorian_string)
				raise
			else:
				return 0

	# repeat for other methods, although it would be nicer to do check type(time) instead


z_test = 39005780. # a mid range test height in cm 

# test_file = "sample_data/jasoon_shim_052317_IT_10/out/s001.nc"

test_file = "sample_data/jasoon_shim_040118_IT_1/s001.nc"

# test_file2 = "sample_data/jasoon_shim_040118_IT_1/s001.nc"


def test_time_range():
	tiegcm = TIEGCM(test_file)
	start, end = tiegcm.get_time_range()
	test_start, test_end = pd.Timestamp('2015-03-10 00:20:00'), pd.Timestamp('2015-03-10 08:00:00')
	assert start == test_start
	assert end == test_end


# def test_model_manager_density():
# 	# datetime 2012-10-01T01:00:02.000 utcepoch 26201.5416898 xlat: -3.69857 xlon: -156.48846  xalt [km]: 360.01468
# 	test_dir = os.path.dirname(os.path.realpath(test_file))
# 	mm = Model_Manager(test_dir)

# 	print 'model manager file times:\n'
# 	for f in sorted(mm.file_times.keys()):
# 		print f.split('/')[-1], mm.file_times[f]
# 	time_str = '2015-03-10 00:20:00'

# 	time = pd.Timestamp(time_str)
# 	epoch_time = datetime_to_epoch(time)
# 	time_ut = mm.time_to_ut(time)

# 	tiegcm = TIEGCM(test_file)

# 	xlat = -3.69857
# 	xlon = -156.48846
# 	xalt = 360.10342*1e5 #cm

# 	# import ipdb; ipdb.set_trace()
# 	result = tiegcm.density(xlat, xlon, xalt, time_ut)*1e3
# 	result2 = mm.density(xlat, xlon, xalt, time_str, raise_errors = True)*1e3
# 	print "{}: {} = {} [kg/m^3] ?".format(epoch_time, result, result2)
# 	assert np.isclose(result, result2)

# 	print 'trying result 3'
# 	result3 = mm.density(xlat, xlon, xalt, 0)
# 	assert result3 == 0

# 	print 'trying result 4'
# 	result4 = mm.density(xlat, xlon, xalt, '2015-03-10 08:10:00', raise_errors = True, debug = True)
# 	print result4


def test_model_manager_speed():
	# datetime 2012-10-01T01:00:02.000 utcepoch 26201.5416898 xlat: -3.69857 xlon: -156.48846  xalt [km]: 360.01468
	test_dir = '~/Downloads/2013.03.01.tie-gcm.data'
	# test_dir = os.path.dirname(os.path.realpath(test_file))
	mm = Model_Manager(test_dir)

	print 'model manager file times:\n'
	for f in sorted(mm.file_times.keys()):
		print f.split('/')[-1], mm.file_times[f]
	

	times = pd.date_range(start ='2013-03-01 00:20:00', 
	                  end = '2013-03-01 08:20:00', 
	                  freq = '15S')
	xlat = -3.69857
	xlon = -156.48846
	xalt = 360.10342*1e5 #cm
	for t in times:
		result = mm.density(xlat, xlon, xalt, t, raise_errors = True, debug = False) 
		if isclose(result, 0, atol = 1e-30):
			raise ValueError("result: {} xlat:{} xlon:{} xalt:{} t:{}".format(result, xlat, xlon, xalt, t))


# def test_3D_column():
# 	tiegcm = TIEGCM(test_file)
# 	point = Point4D((tiegcm.ut[0]+tiegcm.ut[1])/2, 128.14398737, 87.,  170.  )	
# 	columns = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))
# 	assert columns[0].shape == (2, len(tiegcm.ilev), 2, 2)
# 	for i in range(1,3):
# 		assert columns[i].shape == (len(tiegcm.ilev), 2, 2)

# def test_column_slice():
# 	tiegcm = TIEGCM(test_file)
# 	point = Point4D((tiegcm.ut[0]+tiegcm.ut[1])/2, 128.14398737, 87.,  171.  )
# 	column = tiegcm.get_column_slicer_4D(point)

# 	assert point.latitude > tiegcm.lat[column.latitude][0]
# 	assert point.latitude < tiegcm.lat[column.latitude][1]
# 	assert point.longitude > tiegcm.lon[column.longitude][0]
# 	assert point.longitude < tiegcm.lon[column.longitude][1]

	
# def test_Delaunay_height():
# 	tiegcm = TIEGCM(test_file)
# 	point = Point4D(3.5, z_test,  87.,  170. )
# 	z_column, lat_column, lon_column = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))
# 	delaunay = tiegcm.get_delaunay_3D(z_column, lat_column, lon_column, time_index = 0)

# 	z = tiegcm.get_3D_column(tiegcm.get_column_slicer_4D(point))[0][0]
	
# 	scaled_point = z_test/tiegcm.z_scale, point[2], point[3]
# 	linear_interpolator = LinearNDInterpolator(delaunay, z.ravel()/tiegcm.z_scale, fill_value = tiegcm.fill_value)
	
# 	assert isclose(scaled_point[0], linear_interpolator(scaled_point))


# def test_custom_time_interpolator():
# 	class TestInterpolator(object):
# 		def __init__(self, value):
# 			self.value = value
# 		def __call__(self, point):
# 			return self.value

# 	interpolators = OrderedDict([(0.0, TestInterpolator(5.0)),
# 								(1.0, TestInterpolator(4.0))])

# 	time_interpolator = TimeInterpolator(interpolators)
# 	assert time_interpolator(None, .5) == 4.5


# def test_time_interpolate_start():
# 	## This should replicate the delaunay test above
# 	tiegcm = TIEGCM(test_file)
# 	point = Point3D(z_test,  87.,  170. )
# 	variable_name ='Z'

# 	result = tiegcm.time_interpolate(point, variable_name, tiegcm.ut.min())


# def test_time_interpolate():
# 	## This should replicate the delaunay test above
# 	tiegcm = TIEGCM(test_file)
# 	point = Point3D(z_test,  87.,  170. )
# 	variable_name ='Z'

# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)
# 	variable_name = 'NE'
# 	print variable_name, point, tiegcm.time_interpolate(point, variable_name, 3.5)

# def test_time_interpolate_edge():
# 	tiegcm = TIEGCM(test_file)
# 	variable_name ='Z'

# 	point = Point3D(z_test,  87.,  tiegcm.lon.min() )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  87.,  tiegcm.lon.min() - 1 )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  87.,  tiegcm.lon.max() )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  87.,  tiegcm.lon.max() + 1)
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)


# def test_time_interpolate_2D():
# 	tiegcm = TIEGCM(test_file)
# 	point = Point2D(87.,  170. )
# 	variable_name = 'EFLUX'
# 	print variable_name, point, tiegcm.time_interpolate_2D(point, variable_name, 3.5)
# 	variable_name = 'latitude'
# 	assert isclose(point.latitude, tiegcm.time_interpolate_2D(point, variable_name, 3.5))

# def test_time_interpolate_2D_edge():
# 	tiegcm = TIEGCM(test_file)
# 	variable_name = 'longitude' # longitude has not been converted to [longitude_min, longitude_max]

# 	point = Point2D(87.,  tiegcm.lon.min() )
# 	assert isclose(point.longitude, tiegcm.time_interpolate_2D(point, variable_name, 3.5))
# 	point = Point2D(87.,  tiegcm.lon.max() + 1 )
# 	assert isclose(to_range(point[1],tiegcm.longitude_min, tiegcm.longitude_max), 
# 							tiegcm.time_interpolate_2D(point, variable_name, 3.5))
	
# def test_interpolate_3D():
# 	tiegcm = TIEGCM(test_file)
# 	point = Point3D(z_test,  87.,  170. )
# 	result = tiegcm.interpolate_3D_point(point, 'Z', 5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# def test_units():
# 	tiegcm = TIEGCM(test_file)
# 	assert tiegcm.get_variable_unit('NE') == 'cm-3'

# def test_variable_list():
# 	tiegcm = TIEGCM(test_file)

# 	varlist_2d = tiegcm.list_2d_variables()
# 	for variable_name in varlist_2d:
# 		assert len(tiegcm.rootgrp.variables[variable_name].shape) == 3

# 	varlist_3d = tiegcm.list_3d_variables()
# 	for variable_name in varlist_3d:
# 		assert len(tiegcm.rootgrp.variables[variable_name].shape) == 4



# def test_time_interpolate_speed():
# 	## This should replicate the delaunay test above
# 	tiegcm = TIEGCM(test_file)
# 	npoints = 100
# 	print 'speed test started with', npoints, 'points'
# 	t = time.time()
# 	expected = np.array(z_test)
# 	rand_seed = 0 # gives qhull facet errors Looks like edge issue
# 	# rand_seed = 1 # gives qhull index errors
# 	np.random.seed(rand_seed)
# 	try:
# 		for lat, lon in zip( np.random.uniform(-80, 80, npoints), np.random.uniform(-180, 180, npoints)):
# 			point = Point3D(z_test,  lat,  lon )
# 			variable_name ='Z'
# 			result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 			assert np.isclose(result, expected)	
# 	except:
# 		print 'test failed at', point
# 		print 'ranges:', Point2D(	[str(len(tiegcm.lat))+":", tiegcm.lat.min(), tiegcm.lat.max()],
# 									[str(len(tiegcm.lon))+":", tiegcm.lon.min(), tiegcm.lon.max()])
# 		raise
# 	dt = time.time() - t
# 	print 'speed test finished', dt, 'seconds', dt/npoints, '[sec/point]'
# 	# variable_name = 'NE'
# 	# print variable_name, point, tiegcm.time_interpolate(point, variable_name, 3.5)

# def test_time_interpolate_pole():
# 	tiegcm = TIEGCM(test_file)
# 	variable_name ='Z'

# 	point = Point3D(z_test,  88.,  tiegcm.lon.min() )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  88.,  tiegcm.lon.min() - 1 )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  88.,  tiegcm.lon.max() )
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# 	point = Point3D(z_test,  88.,  tiegcm.lon.max() + 1)
# 	result = tiegcm.time_interpolate(point, variable_name, 3.5)
# 	expected = np.array(z_test)
# 	assert isclose(result, expected)

# def test_high_altitude_triangle():
# 	"""Test precision of triangle interpolation: 
# 		lat/lon interpolation should match query point"""
# 	tiegcm = TIEGCM(test_file, outermost_layer = -1)

# 	time_index = 0
# 	z = tiegcm.z[time_index][-1]

# 	z_test = 1.1*z.max() # high altitude test
# 	p = Point3D(z_test, 20.5, .5)

# 	tree = tiegcm.get_outer_boundary_kdtree(time_index)
# 	distances, vertices = tree.query(p, p = 1, k = 3) # p = 1 for Manhattan distance!

# 	target_shape = tiegcm.lat_[-1].shape
# 	coord_indices = np.array(zip(*np.unravel_index(vertices, target_shape)))

# 	lat_indices, lon_indices = coord_indices[:,0], coord_indices[:,1]

# 	lnd_lat = LinearNDInterpolator(tree.data[vertices][:,1:], tiegcm.lat_[-1][lat_indices, lon_indices])
# 	lnd_lon = LinearNDInterpolator(tree.data[vertices][:,1:], tiegcm.lon_[-1][lat_indices, lon_indices])

# 	print 'test_high_altitude_triangle: lat,lon_indices', lat_indices, lon_indices

# 	assert (float(lnd_lat(p[1:])) == p.latitude)
# 	assert (float(lnd_lon(p[1:])) == p.longitude)


# def test_high_altitude_in_bounds():
# 	"""Test that high latitude interpolation returns something reasonable"""
# 	tiegcm = TIEGCM(test_file)
	
# 	for time_index in range(5):
# 		time_index = 0
# 		z = tiegcm.z[time_index][-1]

# 		z_test = 1.1*z.max() # high altitude test

# 		p = Point3D(z_test, 20.5, .5)

# 		variable_name = 'Z'
# 		result, variable, lat_indices, lon_indices = tiegcm.interpolate_high_altitude(p, variable_name, time_index, True)

# 		assert variable.min() <= result <= variable.max()
# 		print 'test_high_altitude_in_bounds', result


# def test_high_altitude_speed():
# 	time_index = 0
# 	tiegcm = TIEGCM(test_file)
# 	npoints = 100
# 	print 'high altitude speed test started with', npoints, 'points'
# 	t = time.time()
# 	z_test = 1.1*tiegcm.z[time_index][-1].max()

# 	rand_seed = 0 
# 	np.random.seed(rand_seed)
# 	try:
# 		for lat, lon in zip( np.random.uniform(-80, 80, npoints), np.random.uniform(-180, 180, npoints)):
# 			point = Point3D(z_test,  lat,  lon )
# 			variable_name ='Z'

# 			result, variable, lat_indices, lon_indices = tiegcm.interpolate_high_altitude(point, variable_name, time_index, True)
# 			try:
# 				assert variable.min() <= result <= variable.max()
# 			except:
# 				try:
# 					assert (True in [np.isclose(result, v) for v in variable])
# 				except:
# 					print 'not even close'
# 					raise
# 	except:
# 		print 'test failed at', point
# 		# print 'result:', result
# 		# print 'nearby variable {} values:'.format(variable_name), variable
# 		raise
# 	dt = time.time() - t
# 	print 'high altitude speed test finished', dt, 'seconds', dt/npoints, '[sec/point]'
# 	# variable_name = 'NE'
# 	# print variable_name, point, tiegcm.time_interpolate(point, variable_name, 3.5)	


# def test_time_interpolate_high_altitude():
# 	tiegcm = TIEGCM(test_file)

# 	time = 3.5

# 	z_max = tiegcm.z.max()
# 	z_test = 1.1*z_max
# 	p4 = Point4D(time, z_test, 20.5, .5)

# 	column_slicer = tiegcm.get_column_slicer_4D(p4)
# 	z_column, lat_column, lon_column = tiegcm.get_3D_column(column_slicer)

# 	time_index = column_slicer.time.start


# 	# z = tiegcm.z[time_index, -1, :, :] # topmost layer
	
# 	top_z = z_column[:,-1,:,:]


# 	p = Point3D(*p4[1:])
# 	result, (variables0, variables1), (lat_indices0, lat_indices1), (lon_indices0, lon_indices1) = tiegcm.time_interpolate_high_altitude(p, 'Z', time, True)
	
# 	lat_layers = np.stack((tiegcm.lat[lat_indices0], tiegcm.lat[lat_indices1]))
# 	assert lat_layers.min() <= p.latitude <= lat_layers.max()

# 	lon_layers = np.stack((tiegcm.lon[lon_indices0], tiegcm.lon[lon_indices1]))
# 	assert lon_layers.min() <= p.longitude <= lon_layers.max()
	
# 	z_layers = np.stack((variables0, variables1))
# 	assert z_layers.min() <= result <= z_layers.max() # result is between interpolating triangles

# 	assert top_z.min() <= result <= top_z.max() # result has position between column tops

# def test_time_interpolate_high_altitude_temperature():
# 	tiegcm = TIEGCM(test_file)

# 	time = 3.5

# 	z_max = tiegcm.z.max()
# 	z_test = 1.1*z_max
# 	p4 = Point4D(time, z_test, 20.5, .5)

# 	column_slicer = tiegcm.get_column_slicer_4D(p4)
# 	z_column, lat_column, lon_column = tiegcm.get_3D_column(column_slicer)

# 	time_index = column_slicer.time.start

# 	p = Point3D(*p4[1:])
# 	result, (variables0, variables1), (lat_indices0, lat_indices1), (lon_indices0, lon_indices1) = tiegcm.time_interpolate_high_altitude(p, 'TN', time, True)
	
# 	lat_layers = np.stack((tiegcm.lat[lat_indices0], tiegcm.lat[lat_indices1]))
# 	assert lat_layers.min() <= p.latitude <= lat_layers.max()

# 	lon_layers = np.stack((tiegcm.lon[lon_indices0], tiegcm.lon[lon_indices1]))
# 	assert lon_layers.min() <= p.longitude <= lon_layers.max()
	
# 	tn_layers = np.stack((variables0, variables1))
# 	assert tn_layers.min() <= result <= tn_layers.max() # result is between interpolating triangles

# 	## interpolation causes the masked values to get filled
# 	top_z = z_column[:,-1,:,:]
# 	variable = np.array(tiegcm.rootgrp['TN'])[column_slicer]
# 	top_tn = variable[:,-1,:,:]

# 	assert top_tn.min() <= result <= top_tn.max() # result has position between column tops

# 	assert tiegcm.rootgrp.variables['TN'].min() <= result <= tiegcm.rootgrp.variables['TN'].max() #result in bounds of available data


# def test_interpolator_high_altitude_matches():
# 	## This should replicate the delaunay test above
# 	tiegcm = TIEGCM(test_file)

# 	z_max = tiegcm.z.max()
# 	z_test = 1.1*z_max

# 	point = Point3D(z_test, -20.5, .5)

# 	variable_name = 'DEN'
# 	time = 3.5
# 	result = tiegcm.time_interpolate(point, variable_name, time)
# 	result2 = tiegcm.time_interpolate_high_altitude(point, variable_name, time)
# 	print result, result2
	
# 	assert np.isclose(result, result2)	


# def test_density_function():
# 	tiegcm = TIEGCM(test_file)
# 	xlat = -8.81183
# 	xlon = 161.96608
# 	xalt = 361.10342*1e5 #cm
# 	time = 3.5 #ut hours
# 	result = tiegcm.density(xlat, xlon, xalt, time)*1e3
# 	result2 = tiegcm.time_interpolate(Point3D(xalt, xlat, xlon), 'DEN', time)*1e3
# 	print "{} < {} [kg/m^3] ?".format(result, result2)
# 	assert result < result2



