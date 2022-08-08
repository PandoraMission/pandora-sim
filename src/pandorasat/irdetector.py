"""Holds metadata and methods on Pandora NIRDA"""

import astropy.units as u
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import PACKAGEDIR
from .detector import Detector


class NIRDetector(Detector):
    def throughput(self, wavelength):
        return wavelength.value**0 * 0.714

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
        with np.errstate(invalid="ignore", over="ignore"):
            sw_qe = (
                sw_coeffs[0]
                + sw_coeffs[1] * wavelength.to(u.micron).value
                + sw_coeffs[2] * wavelength.to(u.micron).value ** 2
                + sw_coeffs[3] * wavelength.to(u.micron).value ** 3
            )

            sw_qe = np.where(
                wavelength.to(u.micron).value > sw_wavecut_red,
                sw_qe
                * np.exp(
                    (sw_wavecut_red - wavelength.to(u.micron).value) * sw_exponential
                ),
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

    def wavelength_dispersion(self, pixel):
        df = pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength.csv")
        return (
            np.interp(pixel, df.Pixel, df.Wavelength, left=np.nan, right=np.nan)
            * u.micron
        )

    def get_trace(
        self,
        target,
        pixel_resolution=2,
        subarray_size=(40, 300),
        target_center=(20, 200),
    ):
        """Calculates the electrons per second from a source in a subarray"""
        dp = 1 / pixel_resolution
        pix = np.arange(-200, 100, dp) + dp / 2
        wav = self.wavelength_dispersion(pix)
        ar = np.zeros(subarray_size)
        yc, xc = target_center

        wavelength = np.linspace(0.1, 2, 2000) * u.micron
        sed = target.model_spectrum(wavelength)
        sensitivity = self.sensitivity(wavelength)

        pix_edges = np.vstack([pix - dp / 2, pix + dp / 2]).T
        wav_edges = self.wavelength_dispersion(pix_edges)
        # Iterate every pixel, integrate the SED
        for pdx in tqdm(range(len(pix))):
            # Find the value in each pixel
            k = (wavelength > wav_edges[pdx][0]) & (wavelength < wav_edges[pdx][1])
            wp = np.hstack(
                [
                    wav_edges[pdx][0] + 1e-10 * u.AA,
                    wavelength[k],
                    wav_edges[pdx][1] - 1e-10 * u.AA,
                ]
            )
            sp = np.interp(wp, wavelength, sed * sensitivity)
            integral = (
                np.trapz(
                    np.hstack([0, sp, 0]),
                    np.hstack([wav_edges[pdx][0], wp, wav_edges[pdx][1]]),
                )
            ).to(u.electron / u.s)

            # Build the PRF at this wavelength
            x, y, prf = self._bin_prf(wavelength=wav[pdx], center=(pix[pdx], 0))
            # Assign to each pixel
            X, Y = np.meshgrid(x + xc, y + yc)
            k = (X > 0) & (Y > 0) & (X < ar.shape[1]) & (Y < ar.shape[0])
            ar[Y[k], X[k]] += np.nan_to_num(prf[k] * integral.value)
        ar *= u.electron / u.second
        return ar
