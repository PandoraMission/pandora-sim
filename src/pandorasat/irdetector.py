"""Holds metadata and methods on Pandora NIRDA"""

from dataclasses import dataclass
import astropy.units as u
from numpy import typing as npt
import numpy as np
from .optics import Optics


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
    gain: float = 2.7 * u.electron / u.DN
    npix_column: int = 2048 * u.pixel
    npix_row: int = 2048 * u.pixel

    # Readout Properties
    nreads: int = 4  # assumed number of non-destructive reads per integration

    # We will need these details to calculate sensitivity functions, PRFs...etc
    _Optics = Optics()

    def __repr__(self):
        return "Pandora IR Detector"

    def qe(self, wavelength: npt.NDArray) -> npt.NDArray:
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
        sw_wavecut = 2.38

        sw_qe = (
            sw_coeffs[0]
            + sw_coeffs[1] * wavelength.to(u.micron).value
            + sw_coeffs[2] * wavelength.to(u.micron).value ** 2
            + sw_coeffs[3] * wavelength.to(u.micron).value ** 3
        )

        sw_qe = sw_qe * np.exp(
            (sw_wavecut - wavelength.to(u.micron).value) * sw_exponential
        )
        sw_qe[wavelength.to(u.micron).value < sw_wavecut] = np.nan
        return sw_qe * u.dimensionless_unscaled

    def counts_from_jmag(self, jmag: float) -> float:
        """Calculates the counts from a target based on j magnitude

        Parameters:
            jmag (float): j band magnitude

        Returns:
            counts (float): Recorded detector counts
        """
        # NOTE: If counts is > than some limit this should raise a warning to the user.
        raise NotImplementedError

    def sensitivity(self, wavelength: npt.NDArray) -> npt.NDArray:
        """
        Calculate the detector sensitivity as a function of wavelength, encorporating
        the properties of the optics of the telescope

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            sensitivity (npt.NDArray): Array of the sensitivity of the detector as a function of wavelength

        """
        # Use self._Optics to get access to the telescope parameters
        raise NotImplementedError

    def PRF(self) -> object:
        """Uses the PSF from the `Optics` class to make a PRF object
        Should returns a PRF object that has plot and downsample methods."""
        raise NotImplementedError
