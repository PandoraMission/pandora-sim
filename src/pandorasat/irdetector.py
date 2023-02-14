"""Holds metadata and methods on Pandora NIRDA"""

# Standard library
import warnings

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm

from . import PACKAGEDIR
from .detector import Detector
from .psf import OutOfBoundsError
from .utils import get_jitter
from .wcs import get_wcs


class NIRDetector(Detector):
    @property
    def _dispersion_df(self):
        return pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength.csv")

    @property
    def subarray_size(self):
        return (80, 400)

    @property
    def pixel_read_time(self):
        return 1e-5 * u.second / u.pixel

    @property
    def frame_time(self):
        return np.product(self.subarray_size) * u.pixel * self.pixel_read_time

    @property
    def dark(self):
        return 1 * u.electron / u.second

    @property
    def read_noise(self):
        raise ValueError("Not Set")

    @property
    def saturation_limit(self):
        raise ValueError("Not Set")

    @property
    def non_linearity(self):
        raise ValueError("Not Set")

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
                    (sw_wavecut_red - wavelength.to(u.micron).value)
                    * sw_exponential
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

    # def wcs(self, target_ra, target_dec):
    #     # This is where we'd build or use a WCS.
    #     # Here we're assuming no distortions, no rotations.
    #     hdu = fits.PrimaryHDU()
    #     hdu.header["CTYPE1"] = "RA---TAN"
    #     hdu.header["CTYPE2"] = "DEC--TAN"
    #     hdu.header["CRVAL1"] = target_ra
    #     hdu.header["CRVAL2"] = target_dec
    #     hdu.header["CRPIX1"] = 2048 - 1024 + 40  # + 0.5
    #     hdu.header["CRPIX2"] = 2048  # - 0.5
    #     hdu.header["NAXIS1"] = self.naxis1.value
    #     hdu.header["NAXIS2"] = self.naxis2.value
    #     hdu.header["CDELT1"] = -self.pixel_scale.to(u.deg / u.pixel).value
    #     hdu.header["CDELT2"] = self.pixel_scale.to(u.deg / u.pixel).value
    #     # We're not doing any rotation and scaling right now... but those go in PC1_1, PC1_2, PC1_2, PC2_2
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         wcs = WCS(hdu.header)
    #     return wcs

    def wcs(
            self, target_ra: u.Quantity, target_dec: u.Quantity, theta: u.Quantity, distortion: bool=True,
        ):
            """Get the World Coordinate System for a detector

            Parameters:
            -----------
            target_ra: astropy.units.Quantity
                The target RA in degrees
            target_dec: astropy.units.Quantity
                The target Dec in degrees
            theta: astropy.units.Quantity
                The observatory angle in degrees
            distortion_file: str
                Optional file path to a distortion CSV file. See `wcs.read_distortion_file`
            """
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wcs = get_wcs(
                    self,
                    target_ra=target_ra,
                    target_dec=target_dec,
                    theta=theta,
                    crpix1=2048-40,
                    distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv" if distortion else None,
                )
            return wcs

    def diagnose(
        self, n=4, npixels=20, image_type="psf", temperature=10 * u.deg_C
    ):
        wavs = np.linspace(
            self.psf.wavelength1d.min(), self.psf.wavelength1d.max(), n**2
        )
        m = npixels // 2
        fig, ax = plt.subplots(
            n,
            n,
            figsize=(n * 2 + 2, n * 2),
            sharex=True,
            sharey=True,
            facecolor="white",
        )
        for ndx in np.arange(n**2):
            jdx = ndx % n
            idx = (ndx - jdx) // n
            if image_type.lower() == "psf":
                x, y, f = (
                    self.psf.psf_x.value,
                    self.psf.psf_y.value,
                    self.psf.psf([wavs[ndx], temperature]),
                )
                ax[idx, jdx].set(xticklabels=[], yticklabels=[])
            elif image_type.lower() == "prf":
                x, y, f = self.psf.prf(
                    [wavs[ndx], temperature], location=[0, 0]
                )
            im = ax[idx, jdx].pcolormesh(
                x,
                y,
                f.T,
                vmin=0,
                vmax=[0.1 if image_type.lower() == "prf" else 0.01][0],
            )
            ax[idx, jdx].set(
                xlim=(-m, m),
                ylim=(-m, m),
                xticks=np.arange(-(m - 1), m, 2),
                yticks=np.arange(-(m - 1), m, 2),
                title=f"{wavs[ndx]:0.2} $\mu$m",
            )
            ax[idx, jdx].grid(True, ls="-", color="white", lw=0.5, alpha=0.5)
            plt.subplots_adjust(wspace=0, hspace=0.2)
        for jdx in range(n):
            ax[n - 1, jdx].set(xlabel="Pixels")
        for idx in range(n):
            ax[idx, 0].set(ylabel="Pixels")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Normalized PSF Value")
        fig.suptitle(
            f"NIRDA {image_type.upper()} Across Wavelength", fontsize=15
        )
        return fig

    def get_trace(
        self,
        wavelength,
        spectrum,
        pixel_resolution=2,
        target_center=(40, 250),
        temperature=-10 * u.deg_C,
    ):
        """Calculates the electrons per second from a source in a subarray

        Parameters
        ----------
        target: ps.target.Target
            A target with the method to get an SED as a function of wavelength
        pixel_resolution: float
            The number of subpixels to use when building the trace.
            Higher numbers will take longer to calculate.
        target_center: tuple
            Center of the target within the subarray.
        """
        dp = 1 / pixel_resolution
        pix = np.arange(-200, 100, dp) + dp / 2
        wav = self.pixel_to_wavelength(pix * u.pixel)
        if not (
            np.nanmin(np.diff(wav)).to(u.micron).value
            > np.nanmin(np.diff(wavelength)).to(u.micron).value
        ):
            raise ValueError("Model spectrum must be higher resolution.")
        ar = np.zeros(self.subarray_size)
        yc, xc = target_center

        sensitivity = self.sensitivity(wavelength)

        pix_edges = np.vstack([pix - dp / 2, pix + dp / 2]).T
        wav_edges = self.pixel_to_wavelength(pix_edges * u.pixel)

        unit_convert = (1 * wav.unit * spectrum.unit * sensitivity.unit).to(
            u.DN / u.second
        )
        # Iterate every pixel, integrate the SED
        for pdx in range(len(pix)):
            if ~np.isfinite(wav[pdx]):
                continue
            # Find the value in each pixel
            k = (wavelength > wav_edges[pdx][0]) & (
                wavelength < wav_edges[pdx][1]
            )
            wp = np.hstack(
                [
                    wav_edges[pdx][0] + 1e-12 * u.AA,
                    wavelength[k],
                    wav_edges[pdx][1] - 1e-12 * u.AA,
                ]
            )
            sp = np.interp(wp, wavelength, spectrum * sensitivity)
            integral = (
                np.trapz(
                    np.hstack([0, sp.value, 0]),
                    np.hstack(
                        [
                            wav_edges[pdx][0].value,
                            wp.value,
                            wav_edges[pdx][1].value,
                        ]
                    ),
                )
            ) * unit_convert

            # Build the PRF at this wavelength
            #            x, y, prf = self._bin_prf(wavelength=wav[pdx], center=(pix[pdx], 0))
            try:
                x, y, prf = self.psf.prf(
                    [wav[pdx], temperature], location=(pix[pdx] + xc, yc)
                )
            except OutOfBoundsError:
                continue
            # Assign to each pixel
            Y, X = np.meshgrid(y, x)
            k = (X > 0) & (Y > 0) & (X < ar.shape[1]) & (Y < ar.shape[0])
            ar[Y[k], X[k]] += np.nan_to_num(prf[k] * integral.value)
        ar = self.apply_gain(ar * u.DN)
        ar *= 1 / u.second
        return ar

    def get_frames(
        self,
        wavelength,
        spectrum,
        nframes=20,
        target_center=(40, 250),
        pixel_resolution=2,
        temperature=-10 * u.deg_C,
        seed=None,
    ):
        """Calculates the frames  from a source in a subarray

        Parameters
        ----------
        pixel_resolution: float
            The number of subpixels to use when building the trace.
            Higher numbers will take longer to calculate.
        target_center: tuple
            Center of the target within the subarray.
        """
        x1, y1 = np.asarray(get_jitter(nframes=nframes, seed=seed))
        traces = (
            np.asarray(
                [
                    self.get_trace(
                        wavelength,
                        spectrum,
                        target_center=[
                            target_center[0] + y1[idx],
                            target_center[1] + x1[idx],
                        ],
                        temperature=temperature,
                        pixel_resolution=pixel_resolution,
                    )
                    for idx in tqdm(range(nframes), leave=True, position=0)
                ]
            )
            * u.electron
            / u.second
        )
        traces *= self.frame_time
        dark_noise = (
            np.asarray(
                [
                    np.random.poisson(
                        lam=(self.dark * self.frame_time * idx).value,
                        size=self.subarray_size,
                    )
                    * u.electron
                    for idx in range(nframes)
                ]
            )
            * u.electron
        )
        return np.cumsum(traces, axis=0) + dark_noise

    def get_integration(
        self,
        wavelength,
        spectrum,
        nframes,
        target_center=(40, 250),
        pixel_resolution=2,
        temperature=-10 * u.deg_C,
        seed=None,
    ):
        if nframes < 8:
            raise ValueError("Too few frames to do Fowler sampling.")
        frames = self.get_frames(
            wavelength,
            spectrum,
            nframes=nframes,
            target_center=target_center,
            pixel_resolution=pixel_resolution,
            temperature=temperature,
            seed=seed,
        )

        # Fowler sampling
        integration = frames[-4:].mean(axis=0) - frames[:4].mean(axis=0)
        return integration

    def apply_gain(self, values: u.Quantity):
        """Applies a single gain value"""
        return values * 0.5 *u.electron/u.DN
