# Licensed under an MIT open source license - see LICENSE

import abc
import warnings
from functools import wraps
from weakref import WeakKeyDictionary

import numpy as np

from astropy.units import Quantity
from astropy.table import Table
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.constants import si


from . import six
from .structure import Structure
from .flux import UnitMetadataWarning
from .progressbar import AnimatedProgressBar


#from streamlit import caching

__all__ = ['ppv_catalog', 'pp_catalog', 'ppp_catalog']


def memoize(func):

    # cache[instance][method args] -> method result
    # hold weakrefs to instances,
    # to stay out of the way of the garbage collector
    cache = WeakKeyDictionary()

    @wraps(func)
    def wrapper(self, *args):
        try:
            #caching.clear_cache()
            return cache[self][args]
        except KeyError:
            cache.setdefault(self, {})[args] = func(self, *args)
            return cache[self][args]
        except TypeError:
            warnings.warn("Cannot memoize inputs to %s" % func)
            return func(self, *args)

    return wrapper

def quantity_sum(quantities):
    """
    In Astropy 0.3, np.sum will do the right thing for quantities, but in the mean time we need a workaround.
    """
    return np.sum(quantities.value) * quantities.unit

class MissingMetadataWarning(UserWarning):
    pass


def _qsplit(q):
    """Split a potential astropy Quantity into unit/quantity"""
    if isinstance(1. * q, Quantity):
        return q.unit, q.value

    return 1, q


def _unit(q):
    """Return the units associated with a number, array, unit, or Quantity"""
    if q is None:
        return None
    elif isinstance(1 * q, Quantity):
        return (1 * q).unit


class ScalarStatistic(object):
    # This class does all of the heavy computation

    def __init__(self, values, indices):
        """
        Compute pixel-level statistics from a scalar field, sampled at specific
        locations.

        Parameters
        ----------
        values : 1D ndarray
            data values to use
        indices: tuple of 1D arrays
            Location of each element of values. The i-th array in the tuple
            describes the ith positional dimension
        """
        self.values = values.astype(np.float)
        self.indices = indices

    #@memoize
    def mom0(self):
        """The sum of the values"""
        return np.nansum(self.values)

    #@memoize
    def mom1(self):
        """The intensity-weighted mean position"""
        m0 = self.mom0()
        return [np.nansum(i * self.values) / m0 for i in self.indices]

    #@memoize
    def mom2(self):
        """The intensity-weighted covariance matrix"""
        mom1 = self.mom1()
        mom0 = self.mom0()
        v = self.values / mom0

        nd = len(self.indices)
        zyx = tuple(i - m for i, m in zip(self.indices, mom1))

        result = np.zeros((nd, nd))

        for i in range(nd):
            result[i, i] = np.nansum(v * zyx[i] ** 2)
            for j in range(i + 1, nd):
                result[i, j] = result[j, i] = np.nansum(v * zyx[i] * zyx[j])

        return result

    #@memoize
    def mom2_along(self, direction):
        """
        Intensity-weighted variance/covariance along 1 or more directions.

        Parameters
        ----------
        direction : array like
                  One or more set of direction vectors. Need not be normalized

        Returns
        -------
        result : array
            The variance (or co-variance matrix) of the data along the
            specified direction(s).
        """
        w = np.atleast_2d(direction).astype(np.float)
        for row in w:
            row /= np.linalg.norm(row)

        result = np.dot(np.dot(w, self.mom2()), w.T)
        if result.size == 1:
            result = np.asscalar(result)
        return result

    #@memoize
    def paxes(self):
        """
        The principal axes of the data (direction of greatest elongation)

        Returns
        -------
        result : tuple
            Ordered tuple of ndarrays

        Notes
        -----
        Each array is a normalized direction vector. The arrays
        are sorted in decreasing order of elongation of the data
        """
        mom2 = self.mom2()
        w, v = np.linalg.eig(mom2)
        order = np.argsort(w)

        return tuple(v[:, o] for o in order[::-1])

    #@memoize
    def projected_paxes(self, axes):
        """
        The principal axes of a projection of the data onto a subspace

        Paramters
        ---------
        axes : array-like, (nnew, nold)
               The projection to take. Each row defines a unit vector in
               the new coordinate system

        Returns
        --------
        result : tuple
            Tuple of arrays (nnew items)

        Notes
        -----
        The ordered principal axes in the new space
        """
        axes = tuple(axes)
        mom2 = self.mom2_along(axes)
        w, v = np.linalg.eig(mom2)
        order = np.argsort(w)

        return tuple(v[:, o] for o in order[::-1])

    #@memoize
    def count(self):
        """
        Number of elements in the dataset.
        """
        return self.values.size

    def surface_area(self):
        raise NotImplementedError

    def perimeter(self, plane=None):
        raise NotImplementedError


class VectorStatistic(object):

    def __init__(self, values_tuple, indices):
        raise NotImplementedError

    def divergence(self):
        raise NotImplementedError

    def curl(self):
        raise NotImplementedError


class Metadata(object):

    """
    A descriptor to wrap around metadata dictionaries.

    Lets classes reference self.x instead of self.metadata['x'],
    """

    _restrict_types = None

    def __init__(self, key, description, default=None, strict=False):
        """
        Parameters
        ----------
        key : str
               Metadata name.
        description : str
               What the quantity describes
        default : scalar
               Default value if metadata not provided
        strict : bool
               If True, raise KeyError if metadata not provided.
               This overrides default
        """
        if not isinstance(key, six.string_types):
            raise TypeError("Key is", key, type(key))
        self.key = key
        self.description = description or 'no description'
        self.default = default
        self.strict = strict

    def __get__(self, instance, type=None):

        if instance is None:
            return self

        try:
            value = instance.metadata[self.key]
        except KeyError:
            if self.strict:
                raise KeyError("Required metadata item not found: %s" % self)
            else:
                if self.default is not None:
                    warnings.warn("{0} ({1}) missing, defaulting to {2}".format(self.key, self.description, self.default),
                                  MissingMetadataWarning)
                value = self.default

        if value is not None and self._restrict_types is not None:
            if isinstance(value, self._restrict_types):
                return value
            else:
                raise TypeError("{0} should be an instance of {1}".format(self.key, ' or '.join([x.__name__ for x in self._restrict_types])))
        else:
            return value

    def __str__(self):
        return "%s (%s)" % (self.key, self.description)


class MetadataQuantity(Metadata):
    _restrict_types = (u.UnitBase, u.Quantity, float, int)


class MetadataWCS(Metadata):
    _restrict_types = (WCS,)


class SpatialBase(object):

    __metaclass__ = abc.ABCMeta

    wavelength = ('wavelength', 'Wavelength')
    spatial_scale = MetadataQuantity('spatial_scale', 'Pixel width/height')
    beam_major = MetadataQuantity('beam_major', 'Major FWHM of beam')
    beam_minor = MetadataQuantity('beam_minor', 'Minor FWHM of beam')
    data_unit = MetadataQuantity('data_unit', 'Units of the pixel values', strict=True)
    wcs = MetadataWCS('wcs', 'WCS object')
    distance = MetadataQuantity('distance', 'Distance of feature from set origin')
    latituder = MetadataQuantity('latitude', 'Latitude')
    longituder = MetadataQuantity('longitude', 'Longitude')


    @abc.abstractmethod
    def _sky_paxes(self):
        raise NotImplementedError()

    def _world_pos(self, pp=False):
        # World position in persecs
        xyz = self.stat.mom1()[::-1]
        xyz = [xyz[0] - 1, xyz[1] - 1, xyz[2] - 1]
        return self.wcs.all_pix2world([xyz], 0).ravel()[::-1] * u.pc

    def _world_pos_pix(self):
        # world position in pixels
        xyz = self.stat.mom1()[::-1]
        return xyz[::-1] * u.pixel

    def _sky_coord(self):
        world = self._world_pos()
        c = SkyCoord(u=world[0, 0] * u.pc, v=world[0, 1] * u.pc, w=world[0, 2] * u.pc, frame='galactic',
                     representation_type='cartesian')
        lcen = np.round(c.spherical.lon.deg, 2)  # Add these propeties to the catalog
        bcen = np.round(c.spherical.lat.deg, 2)
        dcen = c.spherical.distance.pc.astype(int)
        return lcen, bcen, dcen

    @abc.abstractproperty
    def flux(self):
        raise NotImplementedError

    @abc.abstractproperty
    def x_cen(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def y_cen(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def position_angle(self):
        raise NotImplementedError()

    @property
    def major_sigma(self):
        """
        Major axis of the projection onto the position-position (PP) plane,
        computed from the intensity weighted second moment in direction of
        greatest elongation in the PP plane.
        """
        dx = self.spatial_scale or u.pixel
        a, b = self._sky_paxes()
        # We need to multiply the second moment by two to get the major axis
        # rather than the half-major axis.
        # return dx * np.sqrt(self.stat.mom2_along(tuple(a)))

        final = dx * np.sqrt(self.stat.mom2_along(tuple(a)))
        final = d = (self.distance * final * 3600) / 206265
        return final * (u.pc / u.deg)

    @property
    def minor_sigma(self):
        """
        Minor axis of the projection onto the position-position (PP) plane,
        computed from the intensity weighted second moment perpendicular to
        the major axis in the PP plane.
        """
        dx = self.spatial_scale or u.pixel
        a, b = self._sky_paxes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        #return dx * np.sqrt(self.stat.mom2_along(tuple(b)))

        final = dx * np.sqrt(self.stat.mom2_along(tuple(b)))
        final = (self.distance * final * 3600) / 206265
        return final * (u.pc / u.deg)

    @property
    def area_ellipse(self):
        """
        The area of the ellipse defined by the second moments, where the
        semi-major and semi-minor axes used are the HWHM (half-width at
        half-maximum) derived from the moments.
        """
        return np.pi * self.major_sigma * self.minor_sigma * (2.3548 * 0.5) ** 2

    @property
    def to_mpl_ellipse(self, **kwargs):
        """
        Returns a Matplotlib ellipse representing the first and second moments
        of the structure.

        Any keyword arguments are passed to :class:`~matplotlib.patches.Ellipse`
        """
        from matplotlib.patches import Ellipse
        return Ellipse((self.x_cen.value, self.y_cen.value),
                       self.major_sigma.value * 2.3548,
                       self.minor_sigma.value * 2.3548,
                       angle=self.position_angle.value,
                       **kwargs)

    @property
    def a_sigma(self):
        """
        Ellipsoidal semi-principal axis 'a' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of greatest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the major axis
        # rather than the half-major axis.
        return dx * np.sqrt(self.stat.mom2_along(a))

    @property
    def b_sigma(self):
        """
        Ellipsoidal semi-principal axis 'b' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of second largest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(b))

    @property
    def c_sigma(self):
        """
        Ellipsoidal semi-principal axis 'c' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of smallest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(c))

    @property
    def volume_ellipsoid(self):
        """
        The volume of the ellipsoid defined by the second moments, where the
        principal axes used are the HWHM (half-width at
        half-maximum) derived from the moments.
        """
        return (4. / 3 * np.pi) * self.a_sigma * self.b_sigma * self.c_sigma  * (2.3548) ** 3


class PPVStatistic(SpatialBase):

    """
    Compute properties of structures in a position-position-velocity (PPV)
    cube.

    Parameters
    ----------
    structure : :class:`~astrodendro.structure.Structure` instance
        The structure to compute the statistics for
    metadata : dict
         Key-value pairs of metadata
    """

    velocity_scale = MetadataQuantity('velocity_scale', 'Velocity channel width')
    vaxis = Metadata('vaxis', 'Index of velocity axis (numpy convention)', default=0)

    def __init__(self, stat, metadata=None):
        print(stat)
        if isinstance(stat, Structure):
            self.stat = ScalarStatistic(stat.values(subtree=True),
                                        stat.indices(subtree=True))
        else:
            self.stat = stat
        if len(self.stat.indices) != 3:
            raise ValueError("PPVStatistic can only be used on 3-d datasets")
        self.metadata = metadata or {}

    def _sky_paxes(self):
        vaxis = self.vaxis
        ax = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        ax.pop(vaxis)
        a, b = self.stat.projected_paxes(tuple(ax))
        a = list(a)
        a.insert(0, vaxis)
        b = list(b)
        b.insert(0, vaxis)
        return tuple(a), tuple(b)

    def _sky_axes(self):
        ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        a, b, c = self.stat.projected_paxes(ax)
        a = list(a)
        b = list(b)
        c = list(c)
        return a, b, c

    @property
    def x_cen(self):
        """
        The mean position of the structure in the x direction.
        """
        p = self._world_pos()
        return p[2] if self.vaxis != 2 else p[1]

    @property
    def y_cen(self):
        """
        The mean position of the structure in the y direction.
        """
        p = self._world_pos()
        return p[1] if self.vaxis == 0 else p[0]

    @property
    def v_cen(self):
        """
        The mean velocity of the structure (where the velocity axis can be
        specified by the ``vaxis`` metadata parameter, which defaults to 0
        following the Numpy convention - the third axis in the FITS convention).
        """
        p = self._world_pos()
        return p[self.vaxis]

    @property
    def flux(self):
        """
        The integrated flux of the structure, in Jy (note that this does not
        include any kind of background subtraction, and is just a plain sum of
        the values in the structure, converted to Jy).
        """
        from .flux import compute_flux
        return compute_flux(self.stat.mom0() * self.data_unit,
                            u.Jy,
                            wavelength=self.wavelength,
                            spatial_scale=self.spatial_scale,
                            velocity_scale=self.velocity_scale,
                            beam_major=self.beam_major,
                            beam_minor=self.beam_minor)

    @property
    def v_rms(self):
        """
        Intensity-weighted second moment of velocity (where the velocity axis
        can be specified by the ``vaxis`` metadata parameter, which defaults to
        0 following the Numpy convention - the third axis in the FITS
        convention).
        """
        dv = self.velocity_scale or u.pixel
        ax = [0, 0, 0]
        ax[self.vaxis] = 1
        return dv * np.sqrt(self.stat.mom2_along(tuple(ax)))

    @property
    def position_angle(self):
        """
        The position angle of sky_maj, sky_min in degrees counter-clockwise
        from the +x axis (note that this is the +x axis in pixel coordinates,
        which is the ``-x`` axis for conventional astronomy images).
        """
        a, b = self._sky_paxes()
        a = list(a)
        a.pop(self.vaxis)
        radian = np.arctan2(a[0], a[1])
        #angle =  np.degrees(np.arctan2(a[0], a[1])) * u.degree

        # covnert angle to degrees and normalize it to be b/w 0-180, then 0-90.
        angle = (np.degrees(radian) + 180) % 180
        if angle > 90:
            angle -= 90
        print(angle)
        return angle * u.degree

    @property
    def area_exact(self):
        """
        The exact area of the structure on the sky.
        """
        dx = self.spatial_scale or u.pixel
        indices = zip(*tuple(self.stat.indices[i] for i in range(3) if i != self.vaxis))
        return len(set(indices)) * dx ** 2

    @property
    def a_sigma(self):
        """
        Ellispoidal semi-principal axis 'a' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of greatest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the major axis
        # rather than the half-major axis.
        return dx * np.sqrt(self.stat.mom2_along(a))

    @property
    def b_sigma(self):
        """
        Ellispoidal semi-principal axis 'b' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of second largest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(b))

    @property
    def c_sigma(self):
        """
        Ellispoidal semi-principal axis 'c' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of smallest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(c))

    @property
    def volume_ellipsoid(self):
        """
        The volume of the ellipsoid defined by the second moments, where the
        principal axes used are the HWHM (half-width at
        half-maximum) derived from the moments.
        """
        return 4. / 3 * np.pi * self.a_sigma * self.b_sigma * self.c_sigma #* (2.3548 * 0.5) ** 3


class PPStatistic(SpatialBase):

    """
    Compute properties of structures in a position-position (PP) cube.

    Parameters
    ----------
    structure : :class:`~astrodendro.structure.Structure` instance
        The structure to compute the statistics for
    metadata : dict
         Key-value pairs of metadata
    """

    def __init__(self, stat, metadata=None):
        if isinstance(stat, Structure):
            self.stat = ScalarStatistic(stat.values(subtree=True),
                                        stat.indices(subtree=True))
        else:
            self.stat = stat
        if len(self.stat.indices) != 2:
            raise ValueError("PPStatistic can only be used on 2-d datasets")
        self.metadata = metadata or {}

    def _sky_paxes(self):
        return self.stat.paxes()

    @property
    def flux(self):
        """
        The integrated flux of the structure, in Jy (note that this does not
        include any kind of background subtraction, and is just a plain sum of
        the values in the structure, converted to Jy).
        """
        from .flux import compute_flux
        return compute_flux(self.stat.mom0() * self.data_unit,
                            u.Jy,
                            wavelength=self.wavelength,
                            spatial_scale=self.spatial_scale,
                            beam_major=self.beam_major,
                            beam_minor=self.beam_minor)

    @property
    def radius(self):
        """
        Geometric mean of ``major_sigma`` and ``minor_sigma``.
        distance is in pc already. take distace, multiple by 1au, divide by degree in arcsec
        """
        a = _qsplit(self.major_sigma)[1]
        b = _qsplit(self.minor_sigma)[1]
        radius = np.sqrt(a * b)
        return radius
        d = (self.distance * radius * 3600) / 206265
        return d * u.pc

    @property
    def distancer(self):
        """
        Distance from the solar system to the object
        """
        return self.distance

    @property
    def longitude(self):
        """
        longitude
        """
        return self.longituder

    @property
    def latitude(self):
        """
        latitude
        """
        return self.latituder

    @property
    def position_angle(self):
        """
        The position angle of sky_maj, sky_min in degrees counter-clockwise
        from the +x axis.
        """
        a, b = self._sky_paxes()
        return np.degrees(np.arctan2(a[0], a[1])) * u.degree

    @property
    def x_cen(self):
        """
        The mean position of the structure in the x direction (in pixel
        coordinates, or in world coordinates if the WCS transformation is
        available in the meta-data).
        """
        #try _world_pos instead before just passing lat and long.
        return self._world_pos_pix()[1]

    @property
    def y_cen(self):
        """
        The mean position of the structure in the y direction (in pixel
        coordinates, or in world coordinates if the WCS transformation is
        available in the meta-data).
        """
        return self._world_pos_pix()[0]

    @property
    def area_exact(self):
        """
        The exact area of the structure on the sky.
        """
        dx = self.spatial_scale or u.pixel
        d = (self.distance * dx * 3600) / 206265 * (u.pc / u.deg)

        return self.stat.count() * d ** 2

    @property
    def mass(self):
        """
        The mass of the structre derived from the flux
        Σ/A_k ≃ 183 M⊙ pc−2 mag− where Σ = mass/area_Exact, so we get: M = 183 M_sun * Area_exact * A_K
        """
        A_k = (self.stat.mom0()) / u.pc ** 2
        #a_k = np.sum(self.spatial_scale) * self.stat.mom0() * 183 * self.area_exact

        dx = self.spatial_scale
        a_k = ((self.distance * dx * 3600) / 206265)**2 * 183 * self.stat.mom0() * (u.Msun/u.deg**2)
        return a_k

    @property
    def surface_mass_density(self):
        """
        The Surface Mass Density of the structure.
        """
        return self.mass/self.area_exact


class VolumeBase(object):

    __metaclass__ = abc.ABCMeta

    spatial_scale = MetadataQuantity('spatial_scale', 'Pixel width/height')
    data_unit = MetadataQuantity('data_unit', 'Units of the pixel values', strict=True)
    distance = MetadataQuantity('distance', 'Distance of feature from set origin')
    latitude = MetadataQuantity('latitude', 'Latitude')
    longitude = MetadataQuantity('longitude', 'Longitude')

    @abc.abstractmethod
    def _sky_axes(self):
        raise NotImplementedError()

    def _world_pos(self):
        xyz = self.stat.mom1()[::-1]
        return xyz[::-1] * u.pixel


    @abc.abstractproperty
    def mass(self):
        raise NotImplementedError

    @abc.abstractproperty
    def x_cen(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def y_cen(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def z_cen(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def azimuth(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def elevation(self):
        raise NotImplementedError()

    @property
    def a_sigma(self):
        """
        Ellispoidal semi-principal axis 'a' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of greatest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the major axis
        # rather than the half-major axis.
        return dx * np.sqrt(self.stat.mom2_along(a))

    @property
    def b_sigma(self):
        """
        Ellispoidal semi-principal axis 'b' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of second largest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(b))


    @property
    def c_sigma(self):
        """
        Ellispoidal semi-principal axis 'c' in the position-position-position
        (PPP) volume, computed from the intensity weighted second moment
        in direction of smallest elongation in the PPP volume.
        """
        dx = self.spatial_scale or u.pixel
        a, b, c = self._sky_axes()
        # We need to multiply the second moment by two to get the minor axis
        # rather than the half-minor axis.
        return dx * np.sqrt(self.stat.mom2_along(c))

    @property
    def volume_ellipsoid(self):
        """
        The volume of the ellipsoid defined by the second moments, where the
        principal axes used are the HWHM (half-width at
        half-maximum) derived from the moments.
        """
        return 4./3 * np.pi * self.a_sigma * self.b_sigma * self.c_sigma * (2.3548 * 0.5) ** 3


class PPPStatistic(SpatialBase):

    def __init__(self, stat, metadata=None):
        """
        Derive properties from PPP density.
        Parameters
        ----------
        stat : ScalarStatistic instance
        """

        if isinstance(stat, Structure):
            self.stat = ScalarStatistic(stat.values(subtree=True),
                                        stat.indices(subtree=True))
        else:
            self.stat = stat
        if len(self.stat.indices) != 3:
            raise ValueError("PPPStatistic can only be used on 3D datasets")
        self.metadata = metadata or {}


    def _sky_axes(self):
        ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        a, b, c = self.stat.projected_paxes(ax)
        a = list(a)
        b = list(b)
        c = list(c)
        return a, b, c


    def _sky_coord(self):
        world = self._world_pos()
        c = SkyCoord(u=world[0, 2] * u.pc, v=world[0, 1] * u.pc, w=world[0, 0] * u.pc, frame='galactic',
                     representation_type='cartesian')
        lcen = np.round(c.spherical.lon.deg, 2)  # Add these propeties to the catalog
        bcen = np.round(c.spherical.lat.deg, 2)
        dcen = c.spherical.distance.pc.astype(int)
        return lcen, bcen, dcen


    @property
    def peak_density(self):
        """
        Maximum density in original units.
        """
        return np.nanmax(self.stat.values) * self.data_unit

    @property
    def mass(self):
        input_quantities = self.stat.mom0() * self.data_unit
        output_unit = u.Msun
        if input_quantities.unit.is_equivalent(u.Msun):
            # Simply sum up the values and convert to output unit
            total_flux = quantity_sum(input_quantities).to(u.Msun)
        else:
            total_mass = quantity_sum(input_quantities)
            final_mass = total_mass * 1.37 * 1.6726219 * 10 ** (-24) * (3 * 10 ** (18)) ** 3
            final_mass_in_sm = final_mass / (2 * 10 ** 33) * u.Msun * u.cm ** (3)
            return final_mass_in_sm

        if not output_unit.is_equivalent(u.Msun):
            raise ValueError("output_unit has to be equivalent to Msun")
        else:
            return total_mass.to(output_unit)
        
        # from .mass import compute_mass
        # return compute_mass(self.stat.mom0() * self.data_unit,
        #                     u.Msun,
        #                     spatial_scale=self.spatial_scale)

    @property
    def volume_exact(self):
        """
        The exact volume of the structure in PPP.
        """
        dx = self.spatial_scale or u.pixel
        return self.stat.count() * dx**3

    @property
    def radius(self):
        """
        Equivalent radius of the sphere occupying the same volume
        of the structure.

        Using volume of sphere formula.
        """
        return (3. * self.volume_exact / (4. * np.pi))**(1./3)

    @property
    def surface_area(self):
        """
        Equivalent area of the circle using the equivalent
        radius estimated in PPP.
        """
        return np.pi * self.radius ** 2

    @property
    def azimuth(self):
        """
        The position angle of a_axis in degrees counter-clockwise
        from the +y axis.
        """
        a = self._sky_axes()[0]
        radian = np.arctan2(a[1], a[2])
        # covnert angle to degrees and normalize it to be b/w 0-180, then 0-90.
        angle = (np.degrees(radian) + 180) % 180
        if angle > 90:
            angle -= 90
        return angle * u.degree

    @property
    def density(self):
        """
        Average density in original units.
        """
        return np.nanmean(self.stat.values) * self.data_unit

    @property
    def x_pc(self):
        """
        The mean position of the structure in the x direction in pc.
        """
        return self._world_pos()[2]

    @property
    def y_pc(self):
        """
        The mean position of the structure in the y direction in pc.
        """
        return self._world_pos()[1]

    @property
    def z_pc(self):
        """
        The mean position of the structure in the z direction in pc.
        """
        return self._world_pos()[0]

    @property
    def elevation(self):
        """
        the height above or below the Galactic plane, defined at $z=0$ pc.
        """
        return abs(self.z_pc)

    @property
    def x_pix(self):
        """
        The mean position of the structure in the x direction (in pixel
        coordinates).
        """
        return self._world_pos_pix()[2]

    @property
    def y_pix(self):
        """
        The mean position of the structure in the y direction (in pixel
        coordinates).
        """
        return self._world_pos_pix()[1]

    @property
    def z_pix(self):
        """
        The mean position of the structure in the z direction (in pixel
        coordinates).
        """
        return self._world_pos_pix()[0]

    @property
    def mass_over_surface_area (self):
        """
        Equivalent area of the circle using the equivalent
        radius estimated in PPP over mass.
        """
        return self.mass/self.surface_area

    @property
    def longitude (self):
        """
        The angular distance of a structure eastward along the galactic equator from the galactic center.
        Calculated by converting the $X_{pc}, Y_{pc}$ and , $Z_{pc}$ positions into sky coordinates.
        """

        world = self._world_pos()
        c = SkyCoord(u=world[2] * u.pc, v=world[1] * u.pc, w=world[0] * u.pc, frame='galactic',
                     representation_type='cartesian')
        lcen = np.round(c.spherical.lon.deg, 2)
        return lcen

    @property
    def latitude (self):
        """
        The angle of a structure north or south of the midplane as viewed from Earth.
        Calculated by converting the $X_{pc}, Y_{pc}$ and , $Z_{pc}$ positions into sky coordinates.
        """
        world = self._world_pos()
        c = SkyCoord(u=world[2] * u.pc, v=world[1] * u.pc, w=world[0] * u.pc, frame='galactic',
                     representation_type='cartesian')
        bcen = np.round(c.spherical.lat.deg, 2)
        return bcen

    @property
    def distance (self):
        """
        The distance of the object from the solar system.
        """
        world = self._world_pos()
        c = SkyCoord(u=world[2] * u.pc, v=world[1] * u.pc, w=world[0] * u.pc, frame='galactic',
                     representation_type='cartesian')
        dcen = c.spherical.distance.astype(int) / u.pc
        return dcen


def _make_catalog(structures, fields, metadata, statistic, verbose=False):
    """
    Make a catalog from a list of structures
    """

    result = None

    try:
        shape_tuple = structures.data.shape
    except AttributeError:
        shape_tuple = None

    if verbose:
        print("Computing catalog for {0} structures".format(len(structures)))
        progress_bar = AnimatedProgressBar(end=max(len(structures), 1), width=40, fill='=', blank=' ')

    for struct in structures:
        print(" ", struct)
        values = struct.values(subtree=True)
        indices = np.copy(struct.indices(subtree=True))

        if shape_tuple is not None:
            for index_array, shape in zip(indices, shape_tuple):
                # catch simple cases where a structure wraps around the image boundary
                i2 = np.where(index_array < shape/2, index_array+shape, index_array)
                if i2.ptp() < index_array.ptp():  # more compact with wrapping. Use this
                    index_array[:] = i2

        stat = ScalarStatistic(values, indices)
        stat = statistic(stat, metadata)
        row = {}
        for lbl in fields:
            row[lbl] = getattr(stat, lbl)

        row = dict((lbl, getattr(stat, lbl))
                   for lbl in fields)
        row.update(_idx=struct.idx)

        # first row
        if result is None:
            sorted_row_keys = sorted(row.keys())
            try:
                result = Table(names=sorted_row_keys,
                               dtype=[int if x == '_idx' else float for x in sorted_row_keys])
            except TypeError:  # dtype was called dtypes in older versions of Astropy
                result = Table(names=sorted_row_keys,
                               dtypes=[int if x == '_idx' else float for x in sorted_row_keys])
            for k, v in row.items():
                try:  # Astropy API change
                    result[k].unit = _unit(v)
                except AttributeError:
                    result[k].units = _unit(v)

        # astropy.table.Table should in future support setting row items from
        # quantities, but for now we need to strip off the quantities
        new_row = {}
        for x in row:
            if row[x] is not None:  # in Astropy 0.3+ we no longer need to exclude None items
                if isinstance(row[x], Quantity):
                    new_row[x] = row[x].value
                else:
                    new_row[x] = row[x]
        result.add_row(new_row)

        # Print stats
        if verbose:
            progress_bar + 1
            progress_bar.show_progress()
    result.sort('_idx')

    if verbose:
        progress_bar.progress = 100  # Done
        progress_bar.show_progress()
        print("")  # newline


    return result


def ppv_catalog(structures, metadata, fields=None, verbose=True):
    """
    Iterate over a collection of position-position-velocity (PPV) structures,
    extracting several quantities from each, and building a catalog.

    Parameters
    ----------
    structures : iterable of Structures
         The structures to catalog (e.g., a dendrogram)
    metadata : dict
        The metadata used to compute the catalog
    fields : list of strings, optional
        The quantities to extract. If not provided,
        defaults to all PPV statistics
    verbose : bool, optional
        If True (the default), will generate warnings
        about missing metadata

    Returns
    -------
    table : a :class:`~astropy.table.table.Table` instance
        The resulting catalog
    """
    fields = fields or ['major_sigma', 'minor_sigma', 'radius', 'area_ellipse', 'area_exact',
                        'position_angle', 'v_rms', 'x_cen', 'y_cen', 'v_cen', 'flux', 'a_sigma', 'b_sigma',
                        'c_sigma', 'volume_ellipsoid']


    with warnings.catch_warnings():
        warnings.simplefilter("once" if verbose else 'ignore', category=MissingMetadataWarning)
        warnings.simplefilter("once" if verbose else 'ignore', category=UnitMetadataWarning)
        return _make_catalog(structures, fields, metadata, PPVStatistic, verbose)


def pp_catalog(structures, metadata, fields=None, verbose=True):
    """
    Iterate over a collection of position-position (PP) structures, extracting
    several quantities from each, and building a catalog.

    Parameters
    ----------
    structures : iterable of Structures
         The structures to catalog (e.g., a dendrogram)
    metadata : dict
        The metadata used to compute the catalog
    fields : list of strings, optional
        The quantities to extract. If not provided,
        defaults to all PPV statistics
    verbose : bool, optional
        If True (the default), will generate warnings
        about missing metadata

    Returns
    -------
    table : a :class:`~astropy.table.table.Table` instance
        The resulting catalog
    """
    fields = fields or ['major_sigma', 'minor_sigma', 'radius', 'area_ellipse', 'area_exact',
                        'position_angle', 'x_cen', 'y_cen', 'mass', 'distancer', 'longitude', 'latitude', ]
    with warnings.catch_warnings():
        warnings.simplefilter("once" if verbose else 'ignore', category=MissingMetadataWarning)
        return _make_catalog(structures, fields, metadata, PPStatistic, verbose)


def ppp_catalog(structures, metadata, fields=None, verbose=True):
    """
    Iterate over a collection of position-position-position (PPP) structures,
    extracting several quantities from each, and building a catalog.
    Parameters
    ----------
    structures : iterable of Structures
         The structures to catalog (e.g., a dendrogram)
    metadata : dict
        The metadata used to compute the catalog
    fields : list of strings, optional
        The quantities to extract. If not provided,
        defaults to all PPP statistics
    verbose : bool, optional
        If True (the default), will generate warnings
        about missing metadata
    Returns
    -------
    table : a :class:`~astropy.table.table.Table` instance
        The resulting catalog
    """

    fields = fields or ['a_sigma', 'b_sigma', 'c_sigma', 'radius', 'volume_ellipsoid', 'volume_exact',
                        'azimuth', 'elevation', 'x_pc', 'y_pc', 'z_pc', 'mass', 'peak_density', 'latitude',
                        'x_pix', 'y_pix', 'z_pix', 'surface_area', 'mass_over_surface_area', 'density',
                        'longitude', 'distance']

    # fields = fields or ['a_sigma', 'b_sigma', 'c_sigma', 'radius', 'volume_ellipsoid', 'volume_exact',
    #                     'azimuth', 'elevation', 'x_pc', 'y_pc', 'z_pc', 'mass', 'peak_density', 'latitude',
    #                     'x_pix', 'y_pix', 'z_pix', 'surface_area', 'mass_over_surface_area', 'density',
    #                     'longitude', 'distance']


    with warnings.catch_warnings():
        warnings.simplefilter("once" if verbose else 'ignore', category=MissingMetadataWarning)
        warnings.simplefilter("once" if verbose else 'ignore', category=UnitMetadataWarning)
        return _make_catalog(structures, fields, metadata, PPPStatistic, verbose)
