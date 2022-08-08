"""Holds metadata and methods on Pandora VISDA"""
import astropy.units as u
import numpy as np
from astropy.io import votable

from . import PACKAGEDIR
from .detector import Detector


class VisibleDetector(Detector):
    """Pandora Visible Detector"""

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector
        """
        df = (
            votable.parse(f"{PACKAGEDIR}/data/Pandora.Pandora.Visible.xml")
            .get_first_table()
            .to_table()
            .to_pandas()
        )
        wav, transmission = np.asarray(df.Wavelength) * u.angstrom, np.asarray(
            df.Transmission
        )
        return (
            np.interp(wavelength, wav, transmission, left=0, right=0) * u.DN / u.photon
        )

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.816

    def wavelength_to_pixel(self, wavelength):
        if self.sensitivity(wavelength).value == 0:
            return np.nan
        else:
            return 0

    def pixel_to_wavelength(self, pixel):
        raise ValueError("No unique solution exists")
