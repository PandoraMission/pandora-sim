"""Holds metadata and methods on Pandora NIRDA"""

from dataclasses import dataclass

import astropy.units as u

import numpy as np
from numpy import typing as npt
from scipy.io import loadmat
from . import PACKAGEDIR

from .optics import Optics
from .filters import Throughput
from .utils import photon_energy

from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel


@dataclass
class IRDetector:
    """Holds information on the IR Detector

    Args:
        darkcurrent_T110K (float): Detector dark current at 110K
        gain (float): Gain of the NIRDA in electrons/DN per PANDORA_JWST_NIRCam_detector_parts.pdf
        thermal_var (float): RMS of detector thermal variation in mK
        npix_column (int): Number of pixels for NIRDA in column dimension
        npix_row (int): Number of pixels for NIRDA in row dimension
        nreads (int): Number of reads up the ramp
    """

    # Detector Properties
    darkcurrent_T110K: float = 1.0 * u.electron / u.second / u.pixel
    thermal_var: float = 5 * u.mK
    gain: float = 2.0 * u.electron / u.DN
    npix_column: int = 2048 * u.pixel
    npix_row: int = 2048 * u.pixel
    pixel_scale: float = 1.19 * u.arcsec / u.pixel
    pixel_size: float = 18.0 * u.um / u.pixel

    # Readout Properties
    nreads: int = 4  # assumed number of non-destructive reads per integration
    # I think we need a function here. Simulations required to build it.

    # We will need these details to calculate sensitivity functions, PRFs...etc
    _Optics = Optics()

    def __post_init__(self):
        self._get_psf()

    def _get_psf(self, std=0.5 * u.pix):
        """

        Parameters
        ----------
        std: float
            The standard deviation of the high frequency jitter noise to convolve with PSF
        """
        data = loadmat(f"{PACKAGEDIR}/data/Pandora_nir_20210602.mat")
        psf = data["psf"]
        # This is from Tom, I'm assuming the units are pixels, should check with him
        x = np.arange(-256, 257) * np.ravel(data["dx"]) / 18 * u.pixel
        y = np.arange(-256, 257) * np.ravel(data["dx"]) / 18 * u.pixel
        kernel = Gaussian2DKernel(
            np.median((std) / np.diff(x)).value, np.median((std) / np.diff(y))
        )
        psf = convolve(psf, kernel)

        # Tom thinks this step should be done later when it's at the "science" level...
        psf /= np.trapz(np.trapz(psf, x.value, axis=1), y.value)

        # xnew = np.arange(-33.5, 33.5, 0.04)
        # ynew = np.arange(-33.5, 33.5, 0.04)
        # f = interpolate.interp2d(x, y, np.log(psf), kind="cubic")
        # psf = np.exp(f(xnew, ynew))

        self.psf_x, self.psf_y, self.psf = (
            x,
            y,
            psf,
        )

    def __repr__(self):
        return "Pandora IR Detector"

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector from the JWST NIRCam models.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector

        """
        if not hasattr(wavelength, "unit"):
            raise ValueError("Pass a wavelength with units")
        wavelength = np.atleast_1d(wavelength)
        sw_coeffs = np.array([0.65830, -0.05668, 0.25580, -0.08350])
        sw_exponential = 100.0
        sw_wavecut_red = 1.69  # changed from 2.38 for Pandora
        sw_wavecut_blue = 0.75  # new for Pandora

        sw_qe = (
            sw_coeffs[0]
            + sw_coeffs[1] * wavelength.to(u.micron).value
            + sw_coeffs[2] * wavelength.to(u.micron).value ** 2
            + sw_coeffs[3] * wavelength.to(u.micron).value ** 3
        )

        sw_qe = np.where(
            wavelength.to(u.micron).value > sw_wavecut_red,
            sw_qe
            * np.exp((sw_wavecut_red - wavelength.to(u.micron).value) * sw_exponential),
            sw_qe,
        )

        sw_qe = np.where(
            wavelength.to(u.micron).value < sw_wavecut_blue,
            sw_qe
            * np.exp(
                -(sw_wavecut_blue - wavelength.to(u.micron).value)
                * (sw_exponential / 1.5)
            ),
            sw_qe,
        )
        sw_qe[sw_qe < 1e-5] = 0
        return sw_qe * u.DN / u.photon

    def counts_from_jmag(self, jmag: float) -> float:
        """Calculates the counts from a target based on j magnitude

        Parameters:
            jmag (float): j band magnitude

        Returns:
            counts (float): Recorded detector counts
        """
        # NOTE: If counts is > than some limit this should raise a warning to the user.
        raise NotImplementedError

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.7

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

    def PRF(self) -> object:
        """Uses the PSF from the `Optics` class to make a PRF object
        Should returns a PRF object that has plot and downsample methods."""
        raise NotImplementedError
