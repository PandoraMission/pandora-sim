"""Holds metadata and methods on Pandora VISDA"""

from dataclasses import dataclass
import astropy.units as u
import numpy as np
from scipy.io import loadmat
from scipy import interpolate
from . import PACKAGEDIR
from .filters import Throughput
from .utils import photon_energy
from .optics import Optics
from tqdm import tqdm

from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel


@dataclass
class VisibleDetector:
    """Holds information on the Visible Detector"""

    # Detector Properties
    npix_column: int = 2048 * u.pixel
    npix_row: int = 2048 * u.pixel
    pixel_scale: float = 0.78 * u.arcsec / u.pixel
    pixel_size: float = 6.5 * u.um / u.pixel
    gain: float = 2.0 * u.electron / u.DN

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
        return wavelength.value**0 * 0.7 * u.DN / u.photon

    def jitter(self, xstd=4, ystd=1.5, tstd=3, nsubtimes=50, seed=42):
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

    def _get_psf(self, std=0.5 * u.pix):
        """

        Parameters
        ----------
        std: float
            The standard deviation of the high frequency jitter noise to convolve with PSF
        """
        data = loadmat(f"{PACKAGEDIR}/data/Pandora_vis_20210602.mat")
        psf = data["psf"]
        # This is from Tom, I'm assuming the units are pixels, should check with him
        x = np.arange(-256, 257) * np.ravel(data["dx"]) / 6.5 * u.pixel
        y = np.arange(-256, 257) * np.ravel(data["dx"]) / 6.5 * u.pixel
        kernel = Gaussian2DKernel(
            np.median((std) / np.diff(x)).value, np.median((std) / np.diff(y))
        )
        psf = convolve(psf, kernel)

        # Tom thinks this step shoudl be done later when it's at the "science" level...
        psf /= np.trapz(np.trapz(psf, x.value, axis=1), y.value)

        # xnew = np.arange(-33.5, 33.5, 0.04)
        # ynew = np.arange(-33.5, 33.5, 0.04)
        # f = interpolate.interp2d(x, y, np.log(psf), kind="cubic")
        # psf = np.exp(f(xnew, ynew))

        # This should be convolved with a Gaussian for high frequency noise.

        # Making PSF wavelength dependent in a fake way for now
        wavelength = ((50 * (np.arange(0, 11) - 5)) + 450) * u.nm
        w = wavelength.to(u.nm).value / 450 - 1
        dx, dy = np.gradient(psf)
        psf_cube = psf[:, :, None] + (dx[:, :, None] * w) + (dy[:, :, None] * w)
        psf_cube[psf_cube <= 0] = data["psf"].min()
        self.psf_x, self.psf_y, self.psf_wavelength, self.psf = (
            x,
            y,
            wavelength,
            psf_cube,
        )
        # The PSF should probably be convolved with a Gaussian to account for high frequency noise.

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

        # We need to renormalize psf2 here
        psf2 /= np.asarray(
            [
                np.trapz(np.trapz(psf2[:, :, idx], xbin, axis=1), ybin)
                for idx in range(psf2.shape[2])
            ]
        )[None, None, :]
        return xbin.astype(int), ybin.astype(int), psf2

    def throughput(self, wavelength):
        return Throughput("Pandora_Visible").transmission(wavelength)

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
        photon_flux = photon_flux_density * np.gradient(wavelength)
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
        jitter_x, jitter_y = self.jitter(
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
