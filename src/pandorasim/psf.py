"""Work with Point Spread Functions"""
# Standard library
from copy import deepcopy
from typing import Union

# Third-party
import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits

from . import PACKAGEDIR

"""Generic PSF class"""


class PSF(object):
    """Class to use PSFs"""

    def __init__(
        self,
        X,
        psf_flux,
        dimension_names,
        dimension_units,
        pixel_size,
        sub_pixel_size,
        transpose=False,
        freeze_dictionary={},
        blur_value=(0 * u.pixel, 0 * u.pixel),
    ):
        """PSF class. Takes in a PSF cube.

        This class will let you use an N dimensional PSF fits file, and will let you
        calculate a PRF (pixel response function) at any point in the N dimensions.
        The PRF is the PSF evaluated on a pixel grid.


        """
        self._psf_flux = psf_flux
        self.transpose = transpose
        if self.transpose:
            self._psf_flux = self._psf_flux.transpose(
                [1, 0, *np.arange(2, self._psf_flux.ndim)]
            )
        self.dimension_names = dimension_names
        self.dimension_units = dimension_units
        self.pixel_size = pixel_size
        self.sub_pixel_size = sub_pixel_size
        self.freeze_dictionary = freeze_dictionary
        self.blur_value = blur_value

        for name, x in zip(dimension_names, X):
            setattr(self, name, x)

        self.validate()

    @staticmethod
    def from_file(
        filename=f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
        transpose=False,
        blur_value=(0 * u.pixel, 0 * u.pixel),
    ):
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
        hdu = fits.open(filename)
        pixel_size = hdu[0].header["PIXSIZE"] * u.micron / u.pix
        sub_pixel_size = hdu[0].header["SUBPIXSZ"] * u.micron / u.pix

        #         psf_flux = hdu[1].data

        #         psf_flux = psf_flux.transpose([1, 0, *np.arange(2, psf_flux.ndim)])

        #         # # Testing
        # #        psf_flux = psf_flux[:, :-10]
        #         if transpose:
        #             psf_flux = psf_flux.transpose([1, 0, *np.arange(2, psf_flux.ndim)])
        #         dimension_names = [i.name.lower() for i in hdu[2:]]
        #         replace = {'x':'column', 'y':'row'}
        #         dimension_names = [replace[n] if n in replace else n for n in dimension_names]
        #         dimension_units = [u.Unit(i.header["UNIT"]) for i in hdu[2:]]
        #         X = [i.data * u.Unit(i.header["UNIT"]) for i in hdu[2:]]

        #         # LLNL's matlab files are COLUMN-major
        #         # This should make the array ROW-major
        replace = {"x": "column", "y": "row"}
        dimension_names = [
            replace[i.name.lower()]
            if i.name.lower() in replace
            else i.name.lower()
            for i in hdu[2:]
        ]

        if "row" in dimension_names:
            l = (
                np.where(np.asarray(dimension_names) == "row")[0][0],
                np.where(np.asarray(dimension_names) == "column")[0][0],
            )
            l = np.hstack(
                [l, list(set(list(np.arange(len(hdu) - 2))) - set(l))]
            )
        else:
            l = np.arange(len(hdu) - 2)

        psf_flux = hdu[1].data.transpose(np.hstack([1, 0, *l + 2]))
        dimension_names = [dimension_names[l1] for l1 in l]
        dimension_units = [u.Unit(hdu[l1].header["UNIT"]) for l1 in l + 2]
        X = [
            hdu[l1].data.transpose(l) * u.Unit(hdu[l1].header["UNIT"])
            for l1 in l + 2
        ]

        return PSF(
            X,
            psf_flux,
            dimension_names,
            dimension_units,
            pixel_size,
            sub_pixel_size,
            transpose=transpose,
            blur_value=blur_value,
        )

    def validate(self):
        self.shape = self._psf_flux.shape[:2]
        self.ndims = len(self.dimension_names)

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
                s = np.argsort(lp.value)
                setattr(self, self.dimension_names[dim] + "1d", lp[s])
                # We have to do this to sort the axis, having them sorted will mean it's easier to interpolate later...
                reshape = np.hstack(
                    [np.hstack([dim, list(dims - set([dim]))]) + 2, 0, 1]
                )
                deshape = [
                    np.where(reshape == idx)[0][0]
                    for idx in range(len(reshape))
                ]
                self._psf_flux = self._psf_flux.transpose(reshape)[
                    s
                ].transpose(deshape)
                midpoint = getattr(self, self.dimension_names[dim] + "1d")
                midpoint = midpoint[len(midpoint) // 2]
                setattr(self, self.dimension_names[dim] + "0d", midpoint)

        self.psf_column = (
            (np.arange(self.shape[1]) - self.shape[1] // 2)
            * u.pixel
            * self.sub_pixel_size
        ) / self.pixel_size
        self.psf_row = (
            (np.arange(self.shape[0]) - self.shape[0] // 2)
            * u.pixel
            * self.sub_pixel_size
        ) / self.pixel_size
        self.midpoint = tuple(
            [getattr(self, name + "0d") for name in self.dimension_names]
        )
        for dim, p in enumerate(self.dimension_names):
            lp = getattr(self, self.dimension_names[dim])
            setattr(self, p + "_bounds", [lp.min(), lp.max()])
        self.blur(self.blur_value)
        self._psf_flux_jitter = np.zeros_like(self._psf_flux)
        self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter

    def _get_dim(self, dim: Union[int, str]):
        """Return the numeric dimension of an input int or string"""
        if isinstance(dim, int):
            if (dim > self.ndims) | (dim < 0):
                raise ValueError(f"No dimension `{dim}`")
            l = dim
        elif isinstance(dim, str):
            l = np.where(np.asarray(self.dimension_names) == dim.lower())[0]
            if len(l) == 0:
                raise ValueError(f"No dimension `{dim}`")
            l = l[0]
        return l

    def _check_bounds(self, point):
        """Check a given point has the right shape, units and values in bounds"""
        # Check length
        if len(point) != self.ndims:
            raise OutOfBoundsError(
                f"Must pass {self.ndims}D point: ({', '.join(self.dimension_names)})"
            )
        # Check units
        point = tuple(
            [
                u.Quantity(p, self.dimension_units[dim])
                for dim, p in enumerate(point)
            ]
        )
        # Check in bounds
        for dim, p in enumerate(point):
            bounds = getattr(self, self.dimension_names[dim] + "_bounds")
            if (p < bounds[0]) | (p > bounds[1]):
                raise OutOfBoundsError(
                    f"Point ({p}) out of {self.dimension_names[dim]} bounds."
                )
        return point

    def blur(self, blur_value):
        """Blurs the PSF using a Gaussian Kernel

        Parameters:
        -----------
        blur_value: tuple of astropy quantities, must be in pixels
        """
        xstd, ystd = blur_value
        if not hasattr(xstd, "unit"):
            xstd *= u.pixel
        if not hasattr(ystd, "unit"):
            ystd *= u.pixel
        if np.any([(not (xstd.unit == u.pix)), (not (ystd.unit == u.pix))]):
            raise ValueError("Must pass `blur_value` in units of pixel.")

        xstd = ((self.pixel_size * xstd) / self.sub_pixel_size).value
        ystd = ((self.pixel_size * ystd) / self.sub_pixel_size).value
        a = deepcopy(self._psf_flux)
        if (xstd == 0) & (ystd == 0):
            self._psf_flux_blur = deepcopy(self._psf_flux)
            self._psf_flux_blur_grad = np.asarray(
                np.gradient(self._psf_flux_blur, axis=(0, 1))
            )  # [:, None]
            return
        s = a.shape
        a = a.reshape((s[0], s[1], np.product(s[2:])))
        b = np.asarray(
            [
                convolve(
                    a[:, :, idx],
                    Gaussian2DKernel(
                        xstd,
                        ystd,
                    ),
                )
                for idx in range(a.shape[2])
            ]
        ).transpose([1, 2, 0])
        b = b.reshape(s)
        b /= b.sum(axis=(0, 1))[None, None]
        self._psf_flux_blur = b
        self._psf_flux_blur_grad = np.asarray(
            np.gradient(self._psf_flux_blur, axis=(0, 1))
        )
        self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter
        return

    def jitter(self, row: npt.ArrayLike, column: npt.ArrayLike):
        def grow(ar, ndims):
            for i in range(ndims):
                ar = ar[:, None]
            return ar

        def downsample(ar, npoints):
            nbin = np.ceil(ar.shape[0] / npoints).astype(int)
            a = np.asarray(
                [ar[idx * nbin : (idx + 1) * nbin] for idx in range(npoints)]
            )
            mean, weight = np.asarray([a1.mean() for a1 in a]), np.asarray(
                [len(a1) for a1 in a]
            ).astype(float)
            weight /= weight.max()
            # print(weight)
            return mean, weight

        if len(row) > 5:
            row, row_w = downsample(row, 5)
            column, col_w = downsample(column, 5)
#        npoints = np.min([5, len(row)])
        self._psf_flux_jitter *= 0
        if (row.sum() == 0) & (column.sum() == 0):
            self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter
            return
        g1, g2 = self._psf_flux_blur_grad
        self._psf_flux_jitter = (
            g1 * grow(row * row_w, self.ndims + 2)
            + g2 * grow(column * col_w, self.ndims + 2)
        ).sum(axis=0)
        self.psf_flux = self._psf_flux_blur + self._psf_flux_jitter
        return

    def psf(self, point, check_bounds=True):
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
        if check_bounds:
            point = self._check_bounds(point)
        PSF0 = self.psf_flux
        for dim, p in enumerate(point):
            PSF0 = interpfunc(
                p.value,
                getattr(self, self.dimension_names[dim] + "1d").value,
                PSF0,
            )
        return PSF0 / PSF0.sum()

    def prf(
        self, point, location=None, check_bounds=True
    ):  # , freeze_dimensions=None):
        """
        Bins the PSF down to the pixel scale.

        Parameters
        ----------
        point: tuple
            The point to interpolate at, in `self.dimension_names`
        location: tuple
            The location in pixels on the detector. Must be two pixel values.
            THIS IS ROW MAJOR. Pass (ROW, COLUMN)
        freeze_dimensions: None, int, str or list
            Pass a list with the dimensions you want to "freeze", i.e. set the PSF shape to the midpoint value.
            Freezing a dimension will make this calculation faster, as it won't be interpolated. You can pass
            either integers for the dimensions, or names from `self.dimension_names`.

        Returns
        -------
        row: np.ndarray
            Array of integer row positions
        column: np.ndarray
            Array of integer column positions
        psf: np.ndarray
            2D array of PRF flux values with shape (nrows, ncolumns)
        """

        if location is None:
            rowloc = np.where(np.in1d(self.dimension_names, "row"))[0]
            colloc = np.where(np.in1d(self.dimension_names, "column"))[0]
            if (len(rowloc) == 0) | (len(colloc) == 0):
                raise ValueError("Please provide location data.")
            location = [point[rowloc[0]], point[colloc[0]]]
        if len(location) != 2:
            raise ValueError("Pass a 2D location in pixels")
        location = [
            u.Quantity(location[0], "pixel"),
            u.Quantity(location[1], "pixel"),
        ]

        mod = (self.psf_column.value + location[1].value) % 1
        cyc = ((self.psf_column.value + location[1].value) - mod).astype(int)
        colbin = np.unique(cyc)
        psf0 = self.psf(
            point, check_bounds=check_bounds
        )  # , freeze_dimensions=freeze_dimensions)
        psf1 = np.asarray(
            [psf0[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in colbin]
        ).T
        mod = (self.psf_row.value + location[0].value) % 1
        cyc = ((self.psf_row.value + location[0].value) - mod).astype(int)
        rowbin = np.unique(cyc)
        psf2 = np.asarray(
            [psf1[cyc == c].sum(axis=0) / (cyc == c).sum() for c in rowbin]
        )
        # We need to renormalize psf2 here
        # psf2 /= np.trapz(np.trapz(psf2, colbin, axis=1), rowbin)

        return rowbin.astype(int), colbin.astype(int), psf2 / psf2.sum()

    def __repr__(self):
        freeze_dictionary = f" ({', '.join([f'{key}: {item}' for key, item in self.freeze_dictionary.items()])})"
        return f"{self.ndims}D PSF Model [{', '.join(self.dimension_names)}]{freeze_dictionary}"

    def fix_dimension(self, **args):
        """Drop a dimension of the PSF model?"""
        freeze_dictionary = args
        dnms = self.dimension_names.copy()
        duns = self.dimension_units.copy()
        for key, point in freeze_dictionary.items():
            dim = self._get_dim(key)
            PSF0 = interpfunc(
                point.to(duns[dim]).value,
                getattr(self, dnms[dim] + "1d").value,
                reorder(self._psf_flux, dim),
            )
            dnms.pop(dim)
            duns.pop(dim)

        freeze_dictionary.update(self.freeze_dictionary)
        psf2 = PSF(
            [
                getattr(self, dnm).transpose(
                    np.hstack(
                        [dim, list(set(np.arange(self.ndims)) - set([dim]))]
                    )
                )[0]
                for dnm in dnms
            ],
            PSF0,
            dnms,
            duns,
            self.pixel_size,
            self.sub_pixel_size,
            freeze_dictionary=freeze_dictionary,
            blur_value=self.blur_value,
        )
        return psf2


def interpfunc(l, lp, PSF0):
    if l in lp:
        PSF1 = PSF0[:, :, np.where(lp == l)[0][0]]
    elif l < lp[0]:
        PSF1 = PSF0[:, :, 0]
    elif l > lp[-1]:
        PSF1 = PSF0[:, :, -1]
    else:
        # Find the two closest frames
        d = np.argsort(np.abs(lp - l))[:2]
        d = d[np.argsort(lp[d])]
        # Linearly interpolate
        slope = (PSF0[:, :, d[0]] - PSF0[:, :, d[1]]) / (lp[d[0]] - lp[d[1]])
        PSF1 = PSF0[:, :, d[1]] + (slope * (l - lp[d[1]]))
    return PSF1


def reorder(ar: np.ndarray, dim: int = 0):
    """Reorders a PSF array so that a different dimension is in the front, this helps when we interpolate."""
    if not isinstance(dim, (int, list, np.int_)):
        raise ValueError("Pass an `int` or a `list` of `int`s.")
    if not (ar.ndim - 2) > (np.max(dim)):
        raise ValueError(
            f"No dimension {[d + 2 for d in dim]} in array shape {ar.shape}"
        )
    if np.min(dim) < 0:
        raise ValueError("Can not reorder the first two dimensions.")
    if not hasattr(dim, "__iter__"):
        dim = [dim]
    cdim = [d + 2 for d in dim]
    l = set(np.arange(ar.ndim)[2:]) - set(cdim)
    return ar.transpose(np.hstack([0, 1, cdim, list(l)]))


class OutOfBoundsError(Exception):
    """Exception raised if a point is out of bounds for this PSF"""

    pass
