"""Holds metadata and methods on Pandora NIRDA"""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
import warnings
from astropy.wcs import WCS

from . import PACKAGEDIR
from .detector import Detector
from .psf import OutOfBoundsError


class NIRDetector(Detector):
    @property
    def _dispersion_df(self):
        return pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength.csv")

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

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.61

    def wcs(self, target_ra, target_dec):
            # This is where we'd build or use a WCS.
            # Here we're assuming no distortions, no rotations.
            hdu = fits.PrimaryHDU()
            hdu.header['CTYPE1'] = 'RA---TAN'
            hdu.header['CTYPE2'] = 'DEC--TAN'
            hdu.header['CRVAL1'] = target_ra
            hdu.header['CRVAL2'] = target_dec
            hdu.header['CRPIX1'] = 2048 - 1024 + 40# + 0.5
            hdu.header['CRPIX2'] = 2048# - 0.5
            hdu.header['NAXIS1'] = self.naxis1.value
            hdu.header['NAXIS2'] = self.naxis2.value
            hdu.header['CDELT1'] = -self.pixel_scale.to(u.deg/u.pixel).value
            hdu.header['CDELT2'] = self.pixel_scale.to(u.deg/u.pixel).value
            ## We're not doing any rotation and scaling right now... but those go in PC1_1, PC1_2, PC1_2, PC2_2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                wcs = WCS(hdu.header)
            return wcs

    def get_trace(
        self,
        target,
        pixel_resolution=2,
        subarray_size=(40, 300),
        target_center=(20, 200),
        temperature=-10*u.deg_C
        
    ):
        """Calculates the electrons per second from a source in a subarray

        Parameters
        ----------
        target: ps.target.Target
            A target with the method to get an SED as a function of wavelength
        pixel_resolution: float
            The number of subpixels to use when building the trace.
            Higher numbers will take longer to calculate.
        subarray_size: Tuple
            Size of the subarray to calculate
        target_center: tuple
            Center of the target within the subarray.
        """
        dp = 1 / pixel_resolution
        pix = np.arange(-200, 100, dp) + dp / 2
        wav = self.pixel_to_wavelength(pix * u.pixel)
        ar = np.zeros(subarray_size)
        yc, xc = target_center

        wavelength = np.linspace(0.1, 2, 2000) * u.micron
        sed = target.model_spectrum(wavelength)
        sensitivity = self.sensitivity(wavelength)

        pix_edges = np.vstack([pix - dp / 2, pix + dp / 2]).T
        wav_edges = self.pixel_to_wavelength(pix_edges * u.pixel)
        # Iterate every pixel, integrate the SED
        for pdx in range(len(pix)):
            if ~np.isfinite(wav[pdx]):
                continue
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
            #            x, y, prf = self._bin_prf(wavelength=wav[pdx], center=(pix[pdx], 0))
            try:
                x, y, prf = self.psf.prf([wav[pdx], temperature], location=(pix[pdx], 0))
            except OutOfBoundsError:
                continue
            # Assign to each pixel
            Y, X = np.meshgrid(y + yc, x + xc)
            k = (X > 0) & (Y > 0) & (X < ar.shape[1]) & (Y < ar.shape[0])
            ar[Y[k], X[k]] += np.nan_to_num(prf[k] * integral.value)
        ar *= u.electron / u.second
        return ar
