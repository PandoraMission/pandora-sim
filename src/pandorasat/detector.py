"""Generic Detector class..."""

import abc
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from astropy.io import fits

from . import PACKAGEDIR
from .optics import Optics
from .utils import load_vega, photon_energy


@dataclass
class Detector(abc.ABC):
    """Holds information on the Visible Detector"""

    # Detector Properties
    name: str
    pixel_scale: float
    pixel_size: float
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

        Parameters
        ----------
        std: float
            The standard deviation of the high frequency jitter noise to convolve with PSF
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

    def _bin_prf(self, wavelength, center=(0, 0)):
        mod = (self.psf_x.value + center[0]) % 1
        cyc = (self.psf_x.value + center[0]) - mod
        xbin = np.unique(cyc)
        psf1 = np.asarray(
            [
                self.psf(wavelength)[cyc == c, :].sum(axis=0) / (cyc == c).sum()
                for c in xbin
            ]
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

    def _jitter(self, xstd=4, ystd=1.5, tstd=3, nsubtimes=50, seed=42):
        """Returns the jitter inside a cadence

        This is a dumb placeholder function.
        """
        np.random.seed(seed)
        jitter_x = (
            convolve(np.random.normal(0, xstd, size=nsubtimes), Gaussian1DKernel(tstd))
            * tstd**0.5
            * xstd**0.5
        )
        np.random.seed(seed + 1)
        jitter_y = (
            convolve(
                np.random.normal(0, ystd, size=nsubtimes),
                Gaussian1DKernel(tstd),
            )
            * tstd**0.5
            * ystd**0.5
        )
        return jitter_x, jitter_y

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
        jitter_x, jitter_y = self._jitter(
            nsubtimes=nsubtimes * ncadences, xstd=xstd, ystd=ystd, tstd=tstd, seed=seed
        )
        jitter_x += center[0]
        jitter_y += center[1]
        xs, ys, prfs = [], [], []
        for jx, jy in zip(jitter_x, jitter_y):
            x, y, prf = self._bin_prf(center=(jx, jy))
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
        w = np.arange(0.1, 3, 0.005) * u.micron
        return np.average(w, weights=self.sensitivity(w))

    def _estimate_zeropoint(self):
        wavelength, spectrum = load_vega()
        sens = self.sensitivity(wavelength)
        zeropoint = np.trapz(spectrum * sens, wavelength) / np.trapz(sens, wavelength)
        return zeropoint

    def mag_from_flux(self, flux):
        return -2.5 * np.log10(flux / self.zeropoint)

    def flux_from_mag(self, mag):
        return self.zeropoint * 10 ** (-mag / 2.5)

    def psf(self, wavelength):
        return interp_psf_cube(wavelength, self.psf_wavelength, self.psf_cube)


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
