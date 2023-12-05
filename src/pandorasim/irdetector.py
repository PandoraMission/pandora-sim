"""Holds metadata and methods on Pandora NIRDA"""

# Standard library
import warnings

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm

from pandorasat.irdetector import NIRDetector as nirda

from . import PACKAGEDIR
from .psf import PSF, OutOfBoundsError
from .utils import get_jitter
from .wcs import get_wcs


class NIRDetector(nirda):
    """Holds methods for simulating data from the NIR Detector on Pandora.

    Attributes
    ----------

    ra: float
        Right Ascension of the pointing
    dec: float
        Declination of the pointing
    theta: float
        Roll angle of the pointing
    transpose_psf : bool
        Transpose the LLNL input PSF file, i.e. rotate 90 degrees
    """

    def __init__(
            self,
            ra: u.Quantity,
            dec: u.Quantity,
            theta: u.Quantity,
            transpose_psf: bool = False,
            ):
        self.ra, self.dec, self.theta, = (ra, dec, theta)

        self.frame_dict = {"reset": 1, "read": 2, "drop": 4}

        """Some detector specific functions to run on initialization"""
        self.psf = PSF.from_file(
            f"{PACKAGEDIR}/data/pandora_nir_20220506.fits",
            transpose=transpose_psf,
        )
        self.psf.blur(blur_value=(0.25 * u.pixel, 0.25 * u.pixel))

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

    def world_to_pixel(self, ra, dec, distortion=True):
        """Helper function. This function ensures we keep the row-major convention in pandora-sim.

        Parameters
        ----------
        ra : float
            Right Ascension to be converted to pixel position.
        dec : float
            Declination to be converted to pixel position.
        distortion : bool
            Flag whether to account for the distortion in the WCS when converting from RA/Dec
            to pixel position. Default is True.

        Returns
        -------
        np.ndarray
            Row and column positions of each provided RA and Dec.
        """
        coords = np.vstack(
            [
                ra.to(u.deg).value if isinstance(ra, u.Quantity) else ra,
                dec.to(u.deg).value if isinstance(dec, u.Quantity) else dec,
            ]
        ).T
        if distortion:
            column, row = self.wcs.all_world2pix(coords, 0).T
        else:
            column, row = self.wcs.wcs_world2pix(coords, 0).T
        return np.vstack([row, column])

    def pixel_to_world(self, row, column, distortion=True):
        """Helper function. This function ensures we keep the row-major convention in pandora-sim.

        Parameters
        ----------
        row : float
            Pixel row position to be converted to sky coordinates.
        column : float
            Pixel column position to be converted to sky coordinates.
        distortion : bool
            Flag whether to account for the distortion in the WCS when converting from pixel position
            to sky coordinates. Default is True.

        Returns
        -------
        np.ndarray
            RA and Dec of input pixel positions.
        """
        coords = np.vstack(
            [
                column.to(u.pixel).value
                if isinstance(column, u.Quantity)
                else column,
                row.to(u.pixel).value if isinstance(row, u.Quantity) else row,
            ]
        ).T
        if distortion:
            return self.wcs.all_pix2world(coords, 0).T * u.deg
        else:
            return self.wcs.wcs_pix2world(coords, 0).T * u.deg

    def wavelength_to_pixel(self, wavelength):
        """Provides position on the NIRDA that a given wavelength will be dispersed to.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths to be converted in microns.

        Returns
        -------
        pixel : np.ndarray
            Pixel position that wavelengths are dispersed to on NIRDA.
        """
        if not hasattr(self, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self._dispersion_df
        return np.interp(
            wavelength,
            np.asarray(df.Wavelength) * u.micron,
            np.asarray(df.Pixel) * u.pixel,
            left=np.nan,
            right=np.nan,
        )

    def pixel_to_wavelength(self, pixel):
        """Provides the wavelength that is dispersed to a given pixel position on NIRDA.

        Parameters
        ----------
        pixel : np.ndarray
            Pixel positions.

        Returns
        -------
        wavelength : float or np.nadarray
            Wavelengths dispersed to the given NIRDA pixels.
        """
        if not hasattr(self, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self._dispersion_df
        return np.interp(
            pixel,
            np.asarray(df.Pixel) * u.pixel,
            np.asarray(df.Wavelength) * u.micron,
            left=np.nan,
            right=np.nan,
        )

    def diagnose(
        self, n=4, npixels=20, image_type="psf", temperature=10 * u.deg_C
    ):
        """Plots diagnostic plots of the NIRDA PSF and PRF as they appear on the detector across
        multiple wavelengths.

        Parameters
        ----------
        n : int
            Determines number of subplots (and therefore number of wavelengths to sample) will be
            plotted. n x n plots will be plotted in a square arrangement. Default is 4.
        npixels : int
            Number of pixels to plot in each subplot. Each subplot will be npixels x npixels.
            Default is 20.
        image_type : str
            Specifies whether the PSF or PRF will be plotted. Options are 'psf' or 'prf'. Default
            is 'psf'.
        temperature : float
            Temperature of the detector in degrees Celsius. Default is 10.

        Returns
        -------
        fig : plt.figure
            The output figure.
        """
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
        wavelength : np.ndarray
            Wavelengths to get trace for.
        spectrum : np.ndarray
            Flux values at each wavelength value.
        pixel_resolution : float
            The number of subpixels to use when building the trace. Higher numbers will
            take longer to calculate. Default is 2.
        target_center : tuple
            Center of the target within the subarray. Default is (250, 40).
        temperature : u.Quantity
            Temperature of the detector in Celsius. Default is -10.

        Returns
        -------
        trace : np.ndarray
            Counts values across NIRDA.
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
        """Calculates the frames from a source in a subarray.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths of the target that the frames will be evaluated at.
        spectrum : np.ndarray
            Flux of the target at the wavelengths specified.
        nframes : int
            Number of frames to be coadded Default is 20.
        target_center: tuple
            Center of the target within the subarray. Default is (40, 250).
        pixel_resolution: float
            The number of subpixels to use when building the trace. Higher numbers
            will take longer to calculate. Default is 2
        temperature : u.Quantity
            Temperature of the detector in Celsius. Default is -10.
        seed : int or None
            Seed value to be passed tot the get_jitter function.

        Returns
        -------
        traces : np.ndarray
            Coadded frame of NIRDA observations of the target.
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
        """Integrates the frames of a simulated NIRDA observation and performs
        Fowler sampling on them.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelengths of the target that the frames will be evaluated at.
        spectrum : np.ndarray
            Flux of the target at the wavelengths specified.
        nframes : int
            Number of frames used to generate the integration.
        target_center: tuple
            Center of the target within the subarray. Default is (40, 250).
        pixel_resolution: float
            The number of subpixels to use when building the trace. Higher numbers
            will take longer to calculate. Default is 2
        temperature : u.Quantity
            Temperature of the detector in Celsius. Default is -10.
        seed : int or None
            Seed value to be passed tot the get_jitter function.

        Returns
        -------
        integration : np.ndarray
            Fowler-sampled integration of NIRDA frames.
        """
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
        """Finds the position of a trace, accounting for WCS distortions.

        Parameters
        ----------
        ra : float
            Right Ascension of the target in decimal degrees.
        dec : float
            Declination of the target in decimal degrees.
        pixel_resolution : int
            Resolution of the pixel position sampling. The pixel positions are sampled as
            1 / pixel_resoultion so higher values for pixel_resolution will result in higher
            resolution.

        Returns
        -------
        trace : array
            Pixel positions of trace.

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
        This trace will be fixed to the WCS solution at the given RA and Dec,
        but can be evaluated anywhere on the detector.

        Parameters
        ----------

        ra: astropy.units.Quantity
            The RA to build the WCS solution at in degrees
        dec: astropy.units.Quantity
            The Declination to build the WCS solution at in degrees
        npix: int
            The number of PRFs to evaluate per pixel. Higher numbers are slower, but more
            accurate.
        temperature: astropy.units.Quantity
            Temperature to use for the PRF in degrees C
        sub_res: int
            The number of sub-pixel resolution elements in the returned PRF model. Higher
            numbers are slower, but more accurate.

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
        """Given an input spectrum will get the integrated spectrum. Pass wav_edges
        to define the bounds of the integration.

        Parameters
        ----------
        wav : np.ndarray
            Wavelength values of the input spectrum
        spec : np.ndarray
            Flux density of the input spectrum at each wavelength
        wav_edges : np.ndarray
            A two-element array containing the minimum and maximum wavelengths to
            integrate between.
        plot : bool
            Flag to specify whether to plot the integrated spectrum. Default is False.

        Returns
        -------
        integral : float
            Integrated flux density
        """
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
