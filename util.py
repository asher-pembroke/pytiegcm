import numpy as np
from collections import namedtuple

R_e = 6.371008e8 #cm

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


def geo_to_spherical(point3D, ):
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