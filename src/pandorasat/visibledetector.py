"""Holds metadata and methods on Pandora VISDA"""
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import votable

from . import PACKAGEDIR
from .detector import Detector


class VisibleDetector(Detector):
    """Pandora Visible Detector"""

    @property
    def _dispersion_df(self):
        return pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength_vis.csv")

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

    @property
    def dark(self):
        return 2 * u.electron / u.second
   
    @property
    def read_noise(self):
        return 2.1 * u.electron

    @property
    def bias(self):
        return 100 * u.electron

    @property
    def integration_time(self):
        return 0.2 * u.second

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.816