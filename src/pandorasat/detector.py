"""Generic Detector class"""

import abc
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from scipy.interpolate import interp2d

from . import PACKAGEDIR
from .optics import Optics
from .utils import get_jitter, load_vega, photon_energy


@dataclass
class Detector(abc.ABC):
    """Holds information on a Detector

    Attributes
    ----------

    name: str
        Name of the detector. This will determine which files are loaded, choose
        from `"visda"` or `"nirda"`
    pixel_scale: float
        The pixel scale of the detector in arcseconds/pixel
    pixel_size: float
        The pixel size in microns/mm
    gain: float, optional
        The gain in electrons per data unit
    """

    # Detector Properties
    name: str
    pixel_scale: float
    pixel_size: float
    naxis1: int
    naxis2: int
    gain: float = 2.0 * u.electron / u.DN

    def __post_init__(self):
        if self.name.lower() in ["visda", "vis", "visible", "v"]:
            self.psf_fname = f"{PACKAGEDIR}/data/Pandora_vis.fits"
        elif self.name.lower() in ["nirda", "nir", "ir"]:
            self.psf_fname = f"{PACKAGEDIR}/data/Pandora_nir.fits"
        else:
            raise ValueError(f"No such detector as {self.name}")
        self._get_psf()
        self.zeropoint = self._estimate_zeropoint()

    def __repr__(self):
        return f"Pandora {self.name} Detector"

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector
        """
        pass

    def _get_psf(self, std=0 * u.pix):
        """
        Obtain PSF from a fits file.

        Note the PSF cube should have structure (x pixel, y pixel, wavelength)

        If `std` is not 0, will convolve the PSF with a Gaussian to simulate
        high frequency noise. `std` specifies the width of the Gaussian in pixels.

        Parameters
        ----------
        std: float
            The standard deviation of the high frequency jitter noise to convolve with PSF.

        """

        hdu = fits.open(self.psf_fname)
        w, x, psf_cube = (
            hdu[1].data["wavelength"],
            hdu[2].data["pixels"],
            np.asarray([hdu[i].data for i in np.arange(3, len(hdu))]),
        )
        y = x.copy()
        w, x, y = (
            w * u.Unit(hdu[1].header["TUNIT1"]),
            x * u.Unit(hdu[2].header["TUNIT1"]),
            y * u.Unit(hdu[2].header["TUNIT1"]),
        )
        psf_cube /= np.asarray(
            [np.trapz(np.trapz(psf, x.value, axis=0), y.value) for psf in psf_cube]
        )[:, None, None]
        if std.value != 0:
            kernel = Gaussian2DKernel(
                np.median((std) / np.diff(x)).value, np.median((std) / np.diff(y))
            )
            psf_cube = np.asarray([convolve(psf, kernel) for psf in psf_cube])

        self.psf_x, self.psf_y, self.psf_wavelength, self.psf_cube = (
            x,
            y,
            w,
            psf_cube.transpose([1, 2, 0]),
        )
        # self._reinterpolate_psf()

    def reinterpolate_psf(self, pixel_resolution=4):
        """Optionally reinterpolate internal PSF to a resolution.

        Parameters
        ----------
        pixel_resolution: int
            Number of sub-pixels per pixel in the PSF."""
        dp = 1 / pixel_resolution
        x = np.arange(-40, 40, dp)
        y = np.arange(-40, 40, dp)

        psf_cube = np.asarray(
            [interp2d(self.psf_x, self.psf_y, psf)(x, y) for psf in self.psf_cube.T]
        ).T
        psf_cube /= np.asarray(
            [np.trapz(np.trapz(psf, x, axis=0), y) for psf in psf_cube.T]
        )[None, None, :]
        self.psf_x, self.psf_y, self.psf_cube = x * u.pixel, y * u.pixel, psf_cube
        self.psf_pixel_resolution = pixel_resolution

    def _bin_prf(self, wavelength, center=(0, 0)):
        """
        Bins the PSF down to the pixel scale.
        """
        mod = (self.psf_x.value + center[0]) % 1
        cyc = (self.psf_x.value + center[0]) - mod
        xbin = np.unique(cyc)
        psf0 = self.psf(wavelength)
        psf1 = np.asarray(
            [psf0[cyc == c, :].sum(axis=0) / (cyc == c).sum() for c in xbin]
        )
        mod = (self.psf_y.value + center[1]) % 1
        cyc = (self.psf_y.value + center[1]) - mod
        ybin = np.unique(cyc)
        psf2 = np.asarray(
            [psf1[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in ybin]
        )
        # We need to renormalize psf2 here
        psf2 /= np.trapz(np.trapz(psf2, xbin, axis=1), ybin)

        return xbin.astype(int), ybin.astype(int), psf2

    def throughput(self, wavelength):
        pass

    def sensitivity(self, wavelength):
        sed = 1 * u.erg / u.s / u.cm**2 / u.angstrom
        E = photon_energy(wavelength)
        telescope_area = np.pi * (Optics.mirror_diameter / 2) ** 2
        photon_flux_density = (
            (telescope_area * sed * self.throughput(wavelength) / E).to(
                u.photon / u.second / u.angstrom
            )
            * self.qe(wavelength)
            * self.gain
        )
        photon_flux = photon_flux_density
        sensitivity = photon_flux / sed
        return sensitivity

    def prf(
        self,
        wavelength,
        center=(0, 0),
        seed=42,
        xstd=4.5,
        ystd=1,
        tstd=3,
        exptime=10 * u.second,
        obs_duration=30 * u.second,
    ):
        """
        Make a Pixel Response Function

        Parameters
        ----------
        xstd: np.ndarray
            Standard deviation of the jitter in X dimension
        ystd: np.ndarray
            Standard deviation of the jitter in Y dimension
        tstd: np.ndarray
            Width of Gaussian convolution in the time dimension. Higher
            values will give longer time correlations in jitter.
        """
        nsubtimes = np.ceil((exptime / (0.2 * u.second)).value).astype(int)
        ncadences = np.ceil((obs_duration / exptime).value).astype(int)
        jitter_x, jitter_y = get_jitter(
            nsubtimes=nsubtimes * ncadences, xstd=xstd, ystd=ystd, tstd=tstd, seed=seed
        )
        jitter_x += center[0]
        jitter_y += center[1]
        xs, ys, prfs = [], [], []
        for jx, jy in zip(jitter_x, jitter_y):
            x, y, prf = self._bin_prf(wavelength=wavelength, center=(jx, jy))
            xs.append(x)
            ys.append(y)
            prfs.append(prf)
        xmin = np.hstack(xs).min()
        ymin = np.hstack(ys).min()
        res = np.zeros(
            (
                np.hstack(xs).max() - xmin + 1,
                np.hstack(ys).max() - ymin + 1,
                ncadences,
                *prfs[0].shape[2:],
            )
        )

        for tdx, x, y, prf in zip(range(len(xs)), xs, ys, prfs):
            X, Y = np.asarray(np.meshgrid(x - xmin, y - ymin))
            res[X, Y, np.floor(tdx / nsubtimes).astype(int)] += prf
        res /= len(prfs)
        x, y = (
            np.arange(np.hstack(xs).max() - xmin + 1) + xmin,
            np.arange(np.hstack(ys).max() - ymin + 1) + ymin,
        )

        res /= np.asarray(
            [np.sum(res[:, :, idx], axis=(0, 1)) for idx in range(res.shape[2])]
        )[None, None, :]
        return (
            x,
            y,
            res,
        )

    @property
    def midpoint(self):
        """Mid point of the sensitivity function"""
        w = np.arange(0.1, 3, 0.005) * u.micron
        return np.average(w, weights=self.sensitivity(w))

    def _estimate_zeropoint(self):
        """Use Vega SED to estimate the zeropoint of the detector"""
        wavelength, spectrum = load_vega()
        sens = self.sensitivity(wavelength)
        zeropoint = np.trapz(spectrum * sens, wavelength) / np.trapz(sens, wavelength)
        return zeropoint

    def mag_from_flux(self, flux):
        return -2.5 * np.log10(flux / self.zeropoint)

    def flux_from_mag(self, mag):
        return self.zeropoint * 10 ** (-mag / 2.5)

    def psf(self, wavelength):
        """Get the PSF at a certain wavelength, interpolated from self.psf_cube"""
        return interp_psf_cube(
            wavelength.to(u.micron), self.psf_wavelength, self.psf_cube
        )

    def wavelength_to_pixel(self, wavelength):
        if not hasattr(self, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self._dispersion_df
        return np.interp(
            wavelength,
            np.asarray(df.Wavelength) * u.micron,
            np.asarray(df.Pixel) * u.pixel,
            left=np.nan,
            right=np.nan,
        )

    def pixel_to_wavelength(self, pixel):
        if not hasattr(self, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self._dispersion_df
        return np.interp(
            pixel,
            np.asarray(df.Pixel) * u.pixel,
            np.asarray(df.Wavelength) * u.micron,
            left=np.nan,
            right=np.nan,
        )


def interp_psf_cube(w, wp, fp):
    if w in wp:
        return fp[:, :, np.where(wp == w)[0][0]]
    l = np.where(wp - w > 0)[0]
    if len(l) == 0:
        interp_fp = fp[:, :, 0].copy() * np.nan
    else:
        if l[0] != 0:
            idx = l[0]
            slope = (fp[:, :, idx] - fp[:, :, idx - 1]) / (wp[idx] - wp[idx - 1]).value
            interp_fp = fp[:, :, idx - 1] + (slope * (w - wp[idx - 1]).value)
        else:
            interp_fp = fp[:, :, 0].copy() * np.nan
    return interp_fp
