"""Work with Point Spread Functions"""
import astropy.units as u
import numpy as np
from astropy.io import fits

from . import PACKAGEDIR

"""Generic PSF class"""


class PSF(object):
    """Class to use PSFs"""

    def __init__(self, filename=f"{PACKAGEDIR}/data/pandora_vis_20220506.fits", transpose=False):
        """PSF class. Takes in a PSF cube fits file.

        This class will let you use an N dimensional PSF fits file, and will let you
        calculate a PRF (pixel response function) at any point in the N dimensions.
        The PRF is the PSF evaluated on a pixel grid.

        filename: str
            Filename of PSF fits file. PSF cube must have a shape such that the first
            two dimensions are the x and y extent of the PSF. Defaults to visible PSF.
        transpose: bool
            Transpose the input file, i.e. rotate 90 degrees
        """
        self.filename = filename
        hdu = fits.open(filename)
        self.pixel_size = hdu[0].header["PIXSIZE"] * u.micron / u.pix
        self.psf_flux = hdu[1].data
        # I think this makes it the right way round
        self.psf_flux = self.psf_flux.transpose([1, 0, *np.arange(2, self.psf_flux.ndim)])
        # Testing
        self.psf_flux = self.psf_flux[:, :-10]
        if transpose:
            self.psf_flux = self.psf_flux.transpose([1, 0, *np.arange(2, self.psf_flux.ndim)])
        self.shape = self.psf_flux.shape[:2]
        self.sub_pixel_size = hdu[0].header["SUBPIXSZ"] * u.micron / u.pix

        self.dimension_names = [i.name.lower() for i in hdu[2:]]
        self.ndims = len(self.dimension_names)
        self.dimension_units = [u.Unit(i.header["UNIT"]) for i in hdu[2:]]

        [
            setattr(self, i.name.lower(), i.data * u.Unit(i.header["UNIT"]))
            for i in hdu[2:]
        ]
        [
            setattr(self, f"{i.name.lower()}p", i.data * u.Unit(i.header["UNIT"]))
            for i in hdu[2:]
        ]

        if self.ndims == 1:
            setattr(
                self,
                self.dimension_names[0] + "1d",
                getattr(self, self.dimension_names[0]),
            )
            midpoint = getattr(self, self.dimension_names[0])
            midpoint = midpoint[len(midpoint) // 2]
            setattr(self, self.dimension_names[0] + "0d", midpoint)
        else:
            # Get 1D version of these grids
            dims = set(np.arange(self.ndims))
            for dim in np.arange(self.ndims):
                lp = getattr(self, self.dimension_names[dim]).transpose(
                    np.hstack([dim, list(dims - set([dim]))])
                )
                for d in range(self.ndims - 1):
                    lp = np.take(lp, 0, -1)
                setattr(self, self.dimension_names[dim] + "1d", lp)
                midpoint = getattr(self, self.dimension_names[dim] + "1d")
                midpoint = midpoint[len(midpoint) // 2]
                setattr(self, self.dimension_names[dim] + "0d", midpoint)

        self.psf_x = (
            (np.arange(self.shape[0]) - self.shape[0] // 2)
            * u.pixel
            * self.sub_pixel_size
        ) / self.pixel_size
        self.psf_y = (
            (np.arange(self.shape[1]) - self.shape[1] // 2)
            * u.pixel
            * self.sub_pixel_size
        ) / self.pixel_size
        self.midpoint = tuple(
            [getattr(self, name + "0d") for name in self.dimension_names]
        )

    def _check_bounds(self, point):
        """Check a given point has the right shape, units and values in bounds"""
        # Check length
        if len(point) != self.ndims:
            raise OutOfBoundsError(
                f"Must pass {self.ndims}D point: ({', '.join(self.dimension_names)})"
            )
        # Check units
        point = tuple(
            [u.Quantity(p, self.dimension_units[dim]) for dim, p in enumerate(point)]
        )
        # Check in bounds
        for dim, p in enumerate(point):
            lp = getattr(self, self.dimension_names[dim])
            if (p < lp.min()) | (p > lp.max()):
                raise OutOfBoundsError(
                    f"Point ({p}) out of {self.dimension_names[dim]} bounds."
                )
        return point

    def psf(self, point, freeze_dimensions=None):
        """Interpolate the PSF cube to a particular point

        Parameters
        ----------
        point: tuple
            The point in ndimensions to evaluate the PSF at. Use `self.dimension_names`
            to see what your dimensions are.

        Returns
        -------
        psf : np.ndarray of shape self.shape
            The interpolated PSF
        """
        if not isinstance(point, (list, tuple)):
            point = [point]
        if freeze_dimensions is None:
            freeze_dimensions = []
        if isinstance(freeze_dimensions, (int, str)):
            freeze_dimensions = [freeze_dimensions]
        if len(freeze_dimensions) != 0:
            point = list(point)
            for dim in freeze_dimensions:
                if isinstance(dim, int):
                    if (dim > self.ndims) | (dim < 0):
                        raise ValueError(f"No dimension `{dim}`")
                    point[dim] = self.midpoint[dim]
                elif isinstance(dim, str):
                    l = np.where(np.asarray(self.dimension_names) == dim)[0]
                    if len(l) == 0:
                        raise ValueError(f"No dimension `{dim}`")
                    point[l[0]] = self.midpoint[l[0]]
            point = tuple(point)
        point = self._check_bounds(point)
        PSF0 = self.psf_flux
        for dim, p in enumerate(point):
            PSF0 = interpfunc(
                p.value, getattr(self, self.dimension_names[dim] + "1d").value, PSF0
            )
        return PSF0

    def prf(self, point, location=None, freeze_dimensions=None):
        """
        Bins the PSF down to the pixel scale.

        Parameters
        ----------
        point: tuple
            The point to interpolate at, in `self.dimension_names`
        location: tuple
            The location in pixels on the detector. Must be two pixel values.
        freeze_dimensions: None, int, str or list
            Pass a list with the dimensions you want to "freeze", i.e. set the PSF shape to the midpoint value.
            Freezing a dimension will make this calculation faster, as it won't be interpolated. You can pass
            either integers for the dimensions, or names from `self.dimension_names`.
        """

        if location is None:
            xloc = np.where(np.in1d(self.dimension_names, "x"))[0]
            yloc = np.where(np.in1d(self.dimension_names, "y"))[0]
            if (len(xloc) == 0) | (len(yloc) == 0):
                raise ValueError("Please provide location data.")
            location = [point[xloc[0]], point[yloc[0]]]
        if len(location) != 2:
            raise ValueError("Pass a 2D location in pixels")
        location = [u.Quantity(location[0], "pixel"), u.Quantity(location[1], "pixel")]

        mod = (self.psf_x.value + location[0].value) % 1
        cyc = (self.psf_x.value + location[0].value) - mod
        xbin = np.unique(cyc)
        psf0 = self.psf(point, freeze_dimensions=freeze_dimensions)
        psf1 = np.asarray(
            [psf0[cyc == c, :].sum(axis=0) / (cyc == c).sum() for c in xbin]
        )
        mod = (self.psf_y.value + location[1].value) % 1
        cyc = (self.psf_y.value + location[1].value) - mod
        ybin = np.unique(cyc)
        psf2 = np.asarray(
            [psf1[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in ybin]
        ).T
        # We need to renormalize psf2 here
        psf2 /= np.trapz(np.trapz(psf2, ybin, axis=1), xbin)

        return xbin.astype(int), ybin.astype(int), psf2

    def __repr__(self):
        return f"{self.ndims}D PSF Model [{', '.join(self.dimension_names)}]"



def interpfunc(l, lp, PSF0):
            if l in lp:
                PSF1 = PSF0[:, :, np.where(lp == l)[0][0]]
            else:
                # Find the two closest frames
                d = np.argsort(np.abs(lp - l))[:2]
                d = d[np.argsort(lp[d])]
                # Linearly interpolate
                slope = (PSF0[:, :, d[0]] - PSF0[:, :, d[1]]) / (lp[d[0]] - lp[d[1]])
                PSF1 = PSF0[:, :, d[1]] + (slope * (l - lp[d[1]]))
            return PSF1


class OutOfBoundsError(Exception):
    """Exception raised if a point is out of bounds for this PSF"""

    pass
