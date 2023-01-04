"""Test class for a theoretical UV detector..."""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
from astropy.wcs import WCS

from . import PACKAGEDIR
from .detector import Detector
from .psf import OutOfBoundsError


class UVDetector(Detector):

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector
        """
        return (
            np.interp(wavelength.to(u.nm), self.qe_wav, self.qe_transmission, left=0, right=0) * u.DN / u.photon
        )

    def throughput(self, wavelength):
        # Assume there are 5 aluminium coated reflective surfaces, with a hard cut off at 400nm...
        return (
            np.interp(wavelength.to(u.nm), self.throughput_wav, self.throughput_reflectance**5, left=0, right=0)
        ) * (wavelength.to(u.nm).value < 320)

            
    @property
    def read_noise(self):
        return 2.1 * u.electron

    @property
    def dark(self):
        return 2 * u.electron / u.second