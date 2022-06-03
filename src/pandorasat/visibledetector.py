"""Holds metadata and methods on Pandora VISDA"""

from dataclasses import dataclass
import astropy.units as u
import numpy as np
from scipy.io import loadmat
from scipy import interpolate
from . import PACKAGEDIR

from astropy.convolution import convolve, Gaussian1DKernel


@dataclass
class VisibleDetector:
    """Holds information on the Visible Detector"""

    # Detector Properties
    npix_column: int = 2048 * u.pixel
    npix_row: int = 2048 * u.pixel
    pixel_scale: float = 0.78 * u.arcsec / u.pixel
    pixel_size: float = 6.5 * u.um / u.pixel

    def __post_init__(self):
        self._get_psf()

    def __repr__(self):
        return "Pandora Visible Detector"

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector
        """
        raise NotImplementedError

    def jitter(self, sigma=5, resolution=50):
        """Returns the jitter inside a cadence

        This is a dumb placeholder function.
        """
        jitter_x = convolve(
            np.random.normal(0, sigma, size=resolution), Gaussian1DKernel(3)
        )
        jitter_y = convolve(
            np.random.normal(0, sigma, size=resolution), Gaussian1DKernel(3)
        )
        return jitter_x, jitter_y

    def _get_psf(self):
        data = loadmat(f"{PACKAGEDIR}/data/Pandora_vis_20210602.mat")
        psf = data["psf"]
        # This is from Tom, I'm assuming the units are pixels, should check with him
        x = np.arange(-256, 257) * np.ravel(data["dx"]) / 6.5 * u.pixel
        y = np.arange(-256, 257) * np.ravel(data["dx"]) / 6.5 * u.pixel
        psf /= np.trapz(np.trapz(psf, x.value, axis=0), y.value)

        xnew = np.arange(-33.5, 33.5, 0.04)
        ynew = np.arange(-33.5, 33.5, 0.04)
        f = interpolate.interp2d(x, y, np.log(psf), kind="cubic")
        psf = np.exp(f(xnew, ynew))
        # Making PSF wavelength dependent in a fake way for now
        wavelength = ((50 * (np.arange(0, 11) - 5)) + 450) * u.nm
        w = wavelength.to(u.nm).value / 450 - 1
        dx, dy = np.gradient(psf)
        psf_cube = psf[:, :, None] + (dx[:, :, None] * w) + (dy[:, :, None] * w)
        psf_cube[psf_cube <= 0] = data["psf"].min()
        self.psf_x, self.psf_y, self.psf_wavelength, self.psf = (
            xnew * u.pixel,
            ynew * u.pixel,
            wavelength,
            psf_cube,
        )

    def _bin_prf(self, center=(0, 0)):
        mod = (self.psf_x.value + center[0]) % 1
        cyc = (self.psf_x.value + center[0]) - mod
        xbin = np.unique(cyc)
        psf1 = np.asarray(
            [self.psf[cyc == c, :].sum(axis=0) / (cyc == c).sum() for c in xbin]
        )
        mod = (self.psf_y.value + center[1]) % 1
        cyc = (self.psf_y.value + center[1]) - mod
        ybin = np.unique(cyc)
        psf2 = np.asarray(
            [psf1[:, cyc == c].sum(axis=1) / (cyc == c).sum() for c in ybin]
        )
        return xbin.astype(int), ybin.astype(int), psf2

    def prf(self, center=(0, 0), sigma=5):
        jitter_x, jitter_y = self.jitter(sigma=sigma)
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
                *prfs[0].shape[2:],
            )
        )

        for x, y, prf in zip(xs, ys, prfs):
            X, Y = np.asarray(np.meshgrid(x - xmin, y - ymin))
            res[X, Y] += prf
        res /= len(prfs)
        return (
            np.arange(np.hstack(xs).max() - xmin + 1) + xmin,
            np.arange(np.hstack(ys).max() - ymin + 1) + ymin,
            res,
        )
