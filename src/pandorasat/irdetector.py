"""Holds metadata and methods on Pandora NIRDA"""

# Standard library
import warnings
from glob import glob

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.patches import Rectangle
from tqdm import tqdm

from . import PACKAGEDIR
from .detector import Detector
from .psf import PSF, OutOfBoundsError
from .utils import get_jitter
from .wcs import get_wcs


class NIRDetector(Detector):
    def _setup(self):
        self.shape = (2048, 512)
        """Some detector specific functions to run on initialization"""
        self.psf = PSF.from_file(
            f"{PACKAGEDIR}/data/pandora_nir_20220506.fits",
            transpose=self.transpose_psf,
        )
        self.flat = fits.open(
            np.sort(
                np.atleast_1d(glob(f"{PACKAGEDIR}/data/flatfield_NIRDA*.fits"))
            )[-1]
        )[1].data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.wcs = get_wcs(
                self,
                target_ra=self.ra,
                target_dec=self.dec,
                theta=self.theta,
                crpix1=80,
                distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv",
            )

        # ROW COLUMN JUST LIKE PYTHON
        self.subarray_size = (400, 80)
        self.subarray_center = (self.wcs.wcs.crpix[1], self.wcs.wcs.crpix[0])
        crpix = self.wcs.wcs.crpix
        self.subarray_corner = (
            crpix[1] - self.subarray_size[0] / 2,
            crpix[0] - self.subarray_size[1] / 2,
        )
        # COLUMN, ROW
        self.subarray_row, self.subarray_column = np.meshgrid(
            crpix[1]
            + np.arange(self.subarray_size[0])
            - self.subarray_size[0] / 2,
            crpix[0]
            + np.arange(self.subarray_size[1])
            - self.subarray_size[1] / 2,
            indexing="ij",
        )
        self.trace_range = [-200, 100]

    @property
    def _dispersion_df(self):
        return pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength.csv")

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

    # @property
    # def frame_time(self):
    #     return (
    #         15
    #         * u.microsecond
    #         / u.pixel
    #         * np.product(self.subarray_size)
    #         * u.pixel
    #     ).to(u.second)

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

    # def wcs(
    #     self,
    #     target_ra: u.Quantity,
    #     target_dec: u.Quantity,
    #     theta: u.Quantity,
    #     distortion: bool = True,
    # ):
    #     """Get the World Coordinate System for a detector

    #     Parameters:
    #     -----------
    #     target_ra: astropy.units.Quantity
    #         The target RA in degrees
    #     target_dec: astropy.units.Quantity
    #         The target Dec in degrees
    #     theta: astropy.units.Quantity
    #         The observatory angle in degrees
    #     distortion_file: str
    #         Optional file path to a distortion CSV file. See `wcs.read_distortion_file`
    #     """
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")
    #         wcs = get_wcs(
    #             self,
    #             target_ra=target_ra,
    #             target_dec=target_dec,
    #             theta=theta,
    #             crpix1=2048 - 40,
    #             distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv"
    #             if distortion
    #             else None,
    #         )
    #     return wcs

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
                y, x, f = (
                    self.psf.psf_row.value,
                    self.psf.psf_column.value,
                    self.psf.psf([wavs[ndx], temperature]),
                )
            #                ax[idx, jdx].set(xticklabels=[], yticklabels=[])
            elif image_type.lower() == "prf":
                y, x, f = self.psf.prf(
                    [wavs[ndx], temperature], location=[0, 0]
                )
            im = ax[idx, jdx].pcolormesh(
                x,
                y,
                f,
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
        target_center=(250, 40),
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
                y, x, prf = self.psf.prf(
                    [wav[pdx], temperature],
                    location=(pix[pdx] + yc, xc),
                )
            except OutOfBoundsError:
                continue
            # Assign to each pixel
            Y, X = np.meshgrid(y, x, indexing="ij")
            k = (X > 0) & (Y > 0) & (X < ar.shape[1]) & (Y < ar.shape[0])
            ar[Y[k], X[k]] += np.nan_to_num(prf[k] * integral.value)
        #        ar = self.apply_gain(ar * u.DN)
        #        ar *= 1 / u.second
        return (
            wav_edges[np.isfinite(wav_edges).all(axis=1)].value,
            ar * u.DN / u.second,
        )

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
        return values * 0.5 * u.electron / u.DN

    def get_background_light_estimate(self, ra, dec, duration, shape=None):
        """Placeholder, will estimate the background light at different locations?
        Background in one integration...!
        """
        # This is an approximate value assuming a zodi of ~22 Vmag
        bkg_rate = 4 * u.electron / u.second
        if shape is None:
            shape = self.shape
        bkg = u.Quantity(
            np.random.poisson(
                lam=(bkg_rate * duration).to(u.electron).value, size=shape
            ).astype(int),
            unit="electron",
            dtype="int",
        )
        return bkg

    def get_trace_positions(self, ra, dec, pixel_resolution=2, plot=False):
        """Finds the position of a trace, accounting for WCS distortions

        Parameters
        ----------


        Returns
        -------


        """

        # ROW COLUMN
        # Includes the distortion to the POSITION of the trace due to the WCS
        # p1_0, p2_0 = self.wcs.all_world2pix([[ra.value, dec.value]], 0)[0].T
        p1_0, p2_0 = self.world_to_pixel(ra, dec)
        dp = 1 / pixel_resolution
        p_range = (
            np.arange(self.trace_range[0], self.trace_range[1], dp) + dp / 2
        )

        # CH: This code was when we thought the WCS distortion applied to the trace
        # Is there any distortion from the prism?
        # # positions of trace
        # p1, p2 = p1_0 + p_range, p2_0 + np.zeros_like(p_range)
        # # Focal plane positions of trace
        # f1, f2 = p1 - self.wcs.wcs.crpix[0], p2 - self.wcs.wcs.crpix[1]
        # # Distorted pixel positions of traces
        # l1, l2 = self.wcs.sip_foc2pix(np.vstack([f1, f2]).T, 0).T

        # positions of trace
        l1, l2 = p1_0 + p_range, p2_0 + np.zeros_like(p_range)

        # Exagerate
        #        l2 = (8*(l2 - self.wcs.wcs.crpix[1])) + self.wcs.wcs.crpix[1]
        if plot:
            fig, ax = plt.subplots()
            ax.plot(l2, l1, zorder=10, c="k", label="Trace")
            ax.scatter(p2_0, p1_0, zorder=10, c="k", label="Source Position")
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    2048,
                    2048,
                    edgecolor="k",
                    facecolor="w",
                    label="Detector Edge",
                )
            )
            ax.add_patch(
                Rectangle(
                    self.subarray_corner[::-1],
                    *self.subarray_size[::-1],
                    alpha=0.4,
                    color="C3",
                    label="Subarray",
                )
            )
            ax.set_aspect(1)
            ax.set(
                xlabel="Pixel Column",
                ylabel="Pixel Row",
                xlim=(-2 * self.subarray_size[1], 3 * self.subarray_size[1]),
                ylim=(
                    -self.subarray_size[0] / 2 + self.subarray_corner[0],
                    1.5 * self.subarray_size[0] + self.subarray_corner[0],
                ),
                title="Trace Plot",
            )
            ax.legend(frameon=True)
        return np.asarray([l1, l2]).T

    def get_fasttrace(
        self,
        ra: u.Quantity = None,
        dec: u.Quantity = None,
        npix: int = 2,
        temperature: u.Quantity = 10 * u.deg_C,
        sub_res: int = 3,
    ):
        """Returns a function to evaluate the trace on the IR channel -FAST-

        This trace will be fixed to the WCS solution at the given RA and Dec, but can be evaluated anywhere on the detector.

        Parameters
        ----------

        ra: astropy.units.Quantity
            The RA to build the WCS solution at in degrees
        dec: astropy.units.Quantity
            The Declination to build the WCS solution at in degrees
        npix: int
            The number of PRFs to evaluate per pixel. Higher numbers are slower, but more accurate.
        temperature: astropy.units.Quantity
            Temperature to use for the PRF in degrees C
        sub_res: int
            The number of sub-pixel resolution elements in the returned PRF model. Higher numbers are slower, but more accurate.

        Returns
        -------

        fastprf: function
            A function to evaluate the PRF at any point on the detector.
        """

        if ra is None:
            ra = self.ra
        if dec is None:
            dec = self.dec

        # Distorted pixel positions
        l = (
            self.get_trace_positions(ra, dec, pixel_resolution=npix)
            - self.world_to_pixel(ra, dec).T
        ).T

        # Wavelengths
        wav = self.pixel_to_wavelength(
            np.arange(
                self.trace_range[0],
                self.trace_range[1],
                1 / npix,
            )
            * u.pixel
        ).value

        wav_edges = np.vstack(
            [
                self.pixel_to_wavelength(
                    np.arange(
                        self.trace_range[0] - 0.5,
                        self.trace_range[1] - 0.5,
                        1 / (npix),
                    )
                    * u.pixel
                ).value,
                self.pixel_to_wavelength(
                    np.arange(
                        self.trace_range[0] + 0.5,
                        self.trace_range[1] + 0.5,
                        1 / (npix),
                    )
                    * u.pixel
                ).value,
            ]
        ).T

        k = (
            np.isfinite(wav)
            & (wav > self.psf.wavelength1d.min().value)
            & (wav < self.psf.wavelength1d.max().value)
        )
        wav, wav_edges, l = wav[k], wav_edges[k], l[:, k]

        xs = np.arange(
            int(np.floor(self.psf.psf_column.min().value + l[1].min())) - 1,
            int(np.ceil(self.psf.psf_column.max().value + l[1].max())) + 2,
            1,
        )
        ys = np.arange(
            int(np.floor(self.psf.psf_row.min().value) + l[0].min()) - 1,
            int(np.ceil(self.psf.psf_row.max().value) + l[0].max()) + 2,
            1,
        )
        xp, yp = np.arange(0, 1, 1 / sub_res), np.arange(0, 1, 1 / sub_res)

        grid = np.zeros(
            (sub_res, sub_res, wav.shape[0], ys.shape[0], xs.shape[0])
        )
        jdx, kdx = 0, 0
        for kdx in tqdm(
            range(yp.shape[0]),
            total=sub_res,
            desc="Building fasttrace",
            leave=True,
            position=0,
        ):
            for jdx in range(xp.shape[0]):
                for idx, w in enumerate(wav):
                    y1, x1, ar = self.psf.prf(
                        (w, temperature),
                        location=(yp[jdx] + l[0][idx], xp[kdx] + l[1][idx]),
                    )
                    ar /= ar.sum()
                    k = np.asarray(
                        np.meshgrid(
                            np.in1d(ys, y1), np.in1d(xs, x1), indexing="ij"
                        )
                    ).all(axis=0)
                    # need to integrate here to get correct units! self.sensitivity(w*u.micron)/npix
                    grid[kdx, jdx, idx, k] = ar[
                        np.asarray(
                            np.meshgrid(
                                np.in1d(y1, ys), np.in1d(x1, xs), indexing="ij"
                            )
                        ).all(axis=0)
                    ]  # * self.sensitivity(w*u.micron)/npix

        # ADD wavelength and spectrum here to take care of integration properly?
        def fasttrace(rowloc: float, colloc: float):
            """Function to get the PRF.

            Parameters
            ----------
            rowloc: float
                y location on the detector
            colloc: float
                x location on the detector

            Returns
            -------
            w: np.ndarray
                wavelength of every PRF in the trace stack
            row: np.ndarray
                y location on the detector
            col: np.ndarray
                x location on the detector
            prf : np.ndarray
                3D flux of the PRF on the detector. Sum down first axis to create a summed trace.
            """
            return (
                ys + (rowloc - (rowloc % 1)),
                xs + (colloc - (colloc % 1)),
                grid[
                    np.argmin(np.abs(yp - (rowloc % 1))),
                    np.argmin(np.abs(xp - (colloc % 1))),
                    :,
                    :,
                    :,
                ],
            )

        return wav_edges, fasttrace

    def get_integrated_spectrum(self, wav, spec, wav_edges, plot=False):
        """Given an input spectrum will get the integrated spectrum

        Pass wav_edges to define the bounds of the integration"""
        spectrum = spec.to(u.erg / (u.micron * u.second * u.cm**2)).value
        sensitivity = self.sensitivity(wav).value
        wavelength = wav.to(u.micron).value
        unit_convert = (
            1 * wav.unit * spec.unit * self.sensitivity(wav[0]).unit
        ).to(u.DN / u.second)
        integral = np.zeros(wav_edges.shape[0])
        for pdx in range(len(wav_edges)):
            k = (wavelength > wav_edges[pdx][0]) & (
                wavelength < wav_edges[pdx][1]
            )
            wp = np.hstack(
                [
                    wav_edges[pdx][0] + 1e-12,
                    wavelength[k],
                    wav_edges[pdx][1] - 1e-12,
                ]
            )

            sp = np.interp(wp, wavelength, spectrum * sensitivity)
            integral[pdx] = np.trapz(
                np.hstack([0, sp, 0]),
                np.hstack(
                    [
                        wav_edges[pdx][0],
                        wp,
                        wav_edges[pdx][1],
                    ]
                ),
            )
        integral = integral * unit_convert
        return integral

    # def get_sky_catalog(self, jmagnitude_range=(-3, 18)):
    #     cat = get_sky_catalog(
    #         self.ra,
    #         self.dec,
    #         columns="ra, dec, Teff, logg, jmag",
    #         jmagnitude_range=jmagnitude_range,
    #     )
    #     cat[["nir_row", "nir_column"]] = self.world_to_pixel(cat.ra, cat.dec).T
    #     r1 = cat.nir_row - self.subarray_corner[0]
    #     c1 = cat.nir_column - self.subarray_corner[1]
    #     k = (
    #         (r1 > -self.trace_range[1])
    #         & (r1 < (self.subarray_size[0] - self.trace_range[0]))
    #         & (c1 > -5)
    #         & (c1 < (self.subarray_size[1] + 5))
    #     )
    #     return cat[k].reset_index(drop=True)
