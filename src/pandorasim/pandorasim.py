"""Holds methods on Pandora"""

# Standard library
from copy import deepcopy

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from pandorasat.hardware import Hardware
from pandorasat.orbit import Orbit

from . import PANDORASTYLE, __version__
from .irdetector import NIRDetector
from .utils import get_jitter, get_sky_catalog, save_to_FITS
from .visibledetector import VisibleDetector


class PandoraSim(object):
    """Holds methods for simulating Pandora data.

    Attributes
    ----------
    ra : u.Quantity
        Right Ascension of the target.
    dec : u.Quantity
        Declination of the target.
    theta : u.Quantity
        Roll angle of the observatory.
    obstime : Time
        Time of observation. Default is Time.now().
    duration : u.Quantity
        Observation duration. Default is 60 minutes.
    rowjitter_1sigma : u.Quantity
        1 sigma jitter in the pixel row direction. Default is 0.2 pixels.
    coljitter_1sigma : u.Quantity
        1 sigma jitter in the pixel column direction. Default is 0.2 pixels.
    thetajitter_1sigma : u.Quantity
        1 sigma jitter in the spacecraft roll. Default is 0.0005 degrees.
    jitter_timescale : u.Quantity
        Timescale on which jitter occurs. Default is 60 seconds.
    """

    def __init__(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        theta: u.Quantity,
        obstime: Time = Time.now(),
        duration: u.Quantity = 60 * u.minute,
        rowjitter_1sigma: u.Quantity = 0.2 * u.pixel,
        coljitter_1sigma: u.Quantity = 0.2 * u.pixel,
        thetajitter_1sigma: u.Quantity = 0.0005 * u.deg,
        jitter_timescale: u.Quantity = 60 * u.second,
    ):
        self.ra, self.dec, self.theta, self.obstime, self.duration = (
            ra,
            dec,
            theta,
            obstime,
            duration,
        )
        self.Orbit = Orbit()
        self.Hardware = Hardware()
        self.NIRDA = NIRDetector(
            ra,
            dec,
            theta,
        )
        self.VISDA = VisibleDetector(
            ra,
            dec,
            theta,
        )
        self.rowjitter_1sigma = rowjitter_1sigma
        self.coljitter_1sigma = coljitter_1sigma
        self.thetajitter_1sigma = thetajitter_1sigma
        self.jitter_timescale = jitter_timescale
        self._get_jitter()
        self.SkyCatalog = self.get_sky_catalog()

        nints = (duration.to(u.s) / self.VISDA.integration_time).value
        self.VISDA.time = (
            self.obstime.jd + np.arange(0, nints) / nints * duration.to(u.day).value
        )
        self.VISDA.rowj, self.VISDA.colj, self.VISDA.thetaj = (  # noqa
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.rowj),
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.colj),
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.thetaj),
        )

        self.VISDA.corners, self.VISDA.Catalogs = self.get_vis_stars()
        self.NIRDA.Catalog = self.get_nirda_stars()

    def __repr__(self):
        return f"Pandora Observatory (RA: {np.round(self.ra, 3)}, Dec: {np.round(self.dec, 3)}, theta: {np.round(self.theta, 3)})"

    def _repr_html_(self):
        return f"Pandora Observatory (RA: {self.ra._repr_latex_()},  Dec:{self.dec._repr_latex_()}, theta: {self.theta._repr_latex_()})"

    def _get_jitter(self):
        self.jitter = pd.DataFrame(
            np.asarray(
                get_jitter(
                    self.rowjitter_1sigma.value,
                    self.coljitter_1sigma.value,
                    self.thetajitter_1sigma.value,
                    correlation_time=self.jitter_timescale,
                    nframes=int(
                        (
                            5
                            * self.duration.to(u.second)
                            / self.jitter_timescale.to(u.second)
                        ).value
                    ),
                    frame_time=self.jitter_timescale / 5,
                )
            ).T,
            columns=["time", "rowj", "colj", "thetaj"],
        )
        self.jitter.time = (self.jitter.time * (u.second).to(u.day)) + self.obstime.jd

    def get_sky_catalog(
        self,
        distortion: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Gets the source catalog of an input target

        Parameters
        ----------
        target_name : str
            Target name to obtain catalog for.
        distortion : bool
            Whether to apply a distortion to the WCS when mapping RA and Dec of catalog targets
            to detector pixels. Default is True.
        **kwargs
            Additional arguments passed to the utils.get_sky_catalog function.

        Returns
        -------
        sky_catalog: pd.DataFrame
            Pandas dataframe of all the sources near the target
        """

        # This is fixed for visda, for now
        radius = np.min(
            [
                (2 * self.VISDA.fieldstop_radius.to(u.deg).value ** 2) ** 0.5,
                (2 * ((2048 * u.pix * self.VISDA.pixel_scale).to(u.deg).value / 2) ** 2)
                ** 0.5,
            ]
        )

        # Get location and magnitude data
        cat = get_sky_catalog(self.ra, self.dec, radius=radius * u.deg, **kwargs)

        ra, dec, mag = cat["coords"].ra.deg, cat["coords"].dec.deg, cat["bmag"]
        vis_pix_coords = self.VISDA.world_to_pixel(ra, dec, distortion=distortion)
        nir_pix_coords = self.NIRDA.world_to_pixel(ra, dec, distortion=distortion)
        k = (
            np.abs(vis_pix_coords[0] - self.VISDA.shape[0] / 2)
            < self.VISDA.shape[0] / 2
        ) & (
            np.abs(vis_pix_coords[1] - self.VISDA.shape[1] / 2)
            < self.VISDA.shape[1] / 2
        )
        new_cat = deepcopy(cat)
        for key, item in new_cat.items():
            new_cat[key] = item[k]
        vis_pix_coords, nir_pix_coords, ra, dec, mag = (
            vis_pix_coords[:, k],
            nir_pix_coords[:, k],
            ra[k],
            dec[k],
            mag[k],
        )
        # k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
        # ra, dec, mag = ra[k], dec[k], mag[k]
        # vis_pix_coords = self.VISDA.wcs(
        #     target_ra, target_dec, theta, distortion=distortion
        # ).all_world2pix(np.vstack([ra, dec]).T, 1)
        # nir_pix_coords = self.NIRDA.wcs(
        #     target_ra, target_dec, theta, distortion=distortion
        # ).all_world2pix(np.vstack([ra, dec]).T, 1)
        # ra, dec, mag = cat['coords'].ra.deg, cat['coords'].dec.deg, cat['bmag']
        # vis_pix_coords = self.VISDA.world_to_pixel(
        #     ra, dec, distortion=distortion
        # )
        # nir_pix_coords = self.NIRDA.world_to_pixel(
        #     ra, dec, distortion=distortion
        # )

        # we're assuming that Gaia B mag is very close to the Pandora visible magnitude
        vis_counts = np.zeros_like(mag)
        vis_flux = np.zeros_like(mag)
        wav = np.arange(100, 1000) * u.nm
        s = np.trapz(self.VISDA.sensitivity(wav), wav)
        for idx, m in enumerate(mag):
            f = self.VISDA.flux_from_mag(m)
            vis_flux[idx] = f.value
            vis_counts[idx] = (f * s).to(u.DN / u.second).value
        source_catalog = pd.DataFrame(
            np.vstack(
                [
                    ra,
                    dec,
                    mag,
                    *vis_pix_coords,
                    *nir_pix_coords,
                    vis_counts,
                    vis_flux,
                    new_cat["jmag"],
                    new_cat["teff"].value,
                    new_cat["logg"],
                    new_cat["RUWE"],
                    new_cat["ang_sep"].value,
                ]
            ).T,
            columns=[
                "ra",
                "dec",
                "mag",
                "vis_row",
                "vis_column",
                "nir_row",
                "nir_column",
                "vis_counts",
                "vis_flux",
                "jmag",
                "teff",
                "logg",
                "ruwe",
                "ang_sep",
            ],
        )
        return source_catalog

    def get_nirda_stars(self) -> pd.DataFrame:
        """Gets all of the stars contained within the NIR detector for the given pointing.

        Returns
        -------
        cat : pd.DataFrame
            Catalog of stars contained within the NIR detector for the given pointing.
        """
        cat = self.SkyCatalog.copy()
        cat[["nir_row", "nir_column"]] = self.NIRDA.world_to_pixel(cat.ra, cat.dec).T
        r1 = cat.nir_row - self.NIRDA.subarray_corner[0]
        c1 = cat.nir_column - self.NIRDA.subarray_corner[1]
        k = (
            (r1 > -self.NIRDA.trace_range[1])
            & (r1 < (self.NIRDA.subarray_size[0] - self.NIRDA.trace_range[0]))
            & (c1 > -5)
            & (c1 < (self.NIRDA.subarray_size[1] + 5))
        )
        return cat[k].reset_index(drop=True)

    def get_vis_stars(self):
        """Gets all of the stars contained within the visible detector for the given pointing.

        Returns
        -------
        corners : np.ndarray
            An array containing the row and column of the corners of the subarrays of the stars
            contained within the visible detector. Currenly only includes the target star and
            eight of the brightest background stars.
        minicats : list
            List containing astropy.Table objects for each of the stars contained within the visible
            detector. Currently only includes the target star and eight of the brightest background
            stars.
        """
        dist = self.SkyCatalog["ang_sep"] > (self.VISDA.pixel_scale * 50 * u.pixel).to(
            u.deg
        )
        ruwe = self.SkyCatalog["ruwe"] < 1.2
        edge = (
            (self.SkyCatalog["vis_row"] > 50)
            & (self.SkyCatalog["vis_row"] < (2048 - 51))
            & (self.SkyCatalog["vis_column"] > 50)
            & (self.SkyCatalog["vis_column"] < (2048 - 51))
        )
        cat = (
            self.SkyCatalog[dist & ruwe & edge]
            .sort_values("vis_counts", ascending=False)
            .head(8)
            .reset_index(drop=True)
        )
        corners = np.vstack(
            [
                np.asarray(cat.vis_row, dtype=int) - self.VISDA.subarray_size[0] // 2,
                np.asarray(cat.vis_column, dtype=int)
                - self.VISDA.subarray_size[1] // 2,
            ]
        ).T
        corners = np.vstack(
            [
                np.asarray(
                    [
                        self.VISDA.shape[0] // 2 - self.VISDA.subarray_size[0] // 2,
                        self.VISDA.shape[1] // 2 - self.VISDA.subarray_size[1] // 2,
                    ]
                ),
                corners,
            ]
        )

        minicats = []
        for corner in corners:
            minicat = self.SkyCatalog[
                (self.SkyCatalog.vis_row > (corner[0] - 10))
                & (
                    self.SkyCatalog.vis_row
                    < (corner[0] + self.VISDA.subarray_size[0] + 10)
                )
                & (self.SkyCatalog.vis_column > (corner[1] - 10))
                & (
                    self.SkyCatalog.vis_column
                    < (corner[1] + self.VISDA.subarray_size[1] + 10)
                )
            ].copy()
            minicat["subarray_vis_row"] = minicat["vis_row"] - corner[0]
            minicat["subarray_vis_column"] = minicat["vis_column"] - corner[1]
            minicats.append(
                minicat.sort_values("vis_counts", ascending=False).reset_index(
                    drop=True
                )
            )
        return corners, minicats

    def plot_footprint(self, ax=None) -> plt.axis:
        """Plots the footprint of VISDA and NIRDA at the given pointing.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Figure axis to plot the footprint.

        Returns
        -------
        ax : matplotlib.pyplot.axis
            Figure axis with footprints plotted.
        """
        with plt.style.context(PANDORASTYLE):
            if ax is None:
                fig, ax = plt.subplots()
            ax.scatter(
                self.ra.value,
                self.dec.value,
                edgecolor="k",
                facecolor="None",
                zorder=10,
                label="Target",
                marker="*",
                s=50,
            )
            if hasattr(self, "SkyCatalog"):
                ax.scatter(
                    self.SkyCatalog.ra,
                    self.SkyCatalog.dec,
                    s=0.1,
                    c="grey",
                    label="Background Stars",
                )
            ax.add_patch(
                PathPatch(
                    Path(self.VISDA.wcs.calc_footprint(undistort=False)),
                    color="C2",
                    alpha=0.5,
                    label="VISDA",
                )
            )
            ax.add_patch(
                PathPatch(
                    Path(self.NIRDA.wcs.calc_footprint(undistort=False)),
                    color="C3",
                    alpha=0.5,
                    label="NIRDA",
                )
            )
            ax.set_xlim(self.ra.value - 0.5, self.ra.value + 0.5)
            ax.set_ylim(self.dec.value - 0.5, self.dec.value + 0.5)
            ax.legend(frameon=True)
            ax.set_aspect(1)
            ax.set(
                xlabel="RA",
                ylabel="Dec",
                title=f"RA:{np.round(self.ra, 3)} Dec:{np.round(self.dec, 3)}, theta:{np.round(self.theta, 3)}",
            )
        return ax

    def get_FFIs(
        self,
        nframes: int = 1,
        nreads: int = 10,
        include_cosmics: bool = False,
        include_noise: bool = True,
        make_subarrays: bool = True,
        **kwargs,
    ):
        """Wrapper function for VISDA.get_FFIs to allow user access at the `PandoraSim` class level"""
        self.ffi_nframes = nframes
        self.ffi_nreads = nreads

        _, self.ffis = self.VISDA.get_FFIs(
            self.SkyCatalog,
            nframes=nframes,
            nreads=nreads,
            include_cosmics=include_cosmics,
            include_noise=include_noise,
            **kwargs,
        )

        if make_subarrays:
            subarrays = []

            for idx in range(len(self.VISDA.Catalogs)):
                _, f, _ = self.VISDA.get_subarray(
                    self.VISDA.Catalogs[idx],
                    self.VISDA.corners[idx],
                    nreads=nreads,
                    nframes=nframes,
                    quiet=True,
                    include_noise=include_noise,
                )

                subarrays.append(f)

            self.subarrays = np.array(subarrays)

    def plot_FFI(
        self,
        nreads: int = 10,
        include_cosmics: bool = False,
        include_noise: bool = True,
        figsize: tuple = (10, 8),
        subarrays: bool = True,
        max_subarrays=8,
        **kwargs,
    ) -> plt.figure:
        """Plots the simulated FFIs of the initialized target in the visible camera.

        Parameters
        ----------
        nreads : int
            Number of detector reads to include in the integration. Default is 10.
        include_cosmics : bool
            Flag to determine whether cosmic rays are simulated on the detector. Default is True.
        include_noise : bool
            Flag to determine whether background noise is simulated on the detector. Defautl is
            True.
        figsize : tuple
            Output figure size to be passed to matplotlib.pyplot. Default is (10, 8).
        subarrays : bool
            Flag to determine whether visible detector subarrays centered on the target and
            background stars will be included in the plot. Default is True.
        **kwargs
            Additional arguments to be passed to the plot_corner function. This function is
            responsible for plotting the subarrays on the detector image.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Output figure of the visible detector.
        """
        if not hasattr(self, "ffis"):
            # _, self.ffis = self.VISDA.get_FFIs(
            #     self.SkyCatalog,
            #     nreads=nreads,
            #     nframes=1,
            #     include_cosmics=include_cosmics,
            #     include_noise=include_noise,
            #     #            freeze_dimensions=freeze_dimensions,
            # )
            raise AttributeError("Please create FFIs first with .get_FFIs() command!")

        with plt.style.context(PANDORASTYLE):
            vmin = kwargs.pop("vmin", self.VISDA.bias.value)
            vmax = kwargs.pop("vmax", self.VISDA.bias.value + 20)
            cmap = kwargs.pop("cmap", "Greys_r")
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                self.ffis[0],
                origin="lower",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Counts [e$^-$]", fontsize=12)
            ax.set_title(
                f"Visible Channel RA:{np.round(self.ra, 3)} "
                + f"Dec:{np.round(self.dec, 3)}, "
                + f"theta:{np.round(self.theta, 3)}, "
                + f"{self.VISDA.integration_time*nreads} Integration",
                fontsize=15,
            )
            ax.set_xlabel("Column Pixel", fontsize=13)
            ax.set_ylabel("Row Pixel", fontsize=13)
            ax.set_xlim(0, self.VISDA.shape[0])
            ax.set_ylim(0, self.VISDA.shape[1])

            if subarrays:

                def plot_corner(c, ax, **kwargs):
                    ax.plot(
                        [
                            c[1],
                            c[1] + self.VISDA.subarray_size[1],
                            c[1] + self.VISDA.subarray_size[1],
                            c[1],
                            c[1],
                        ],
                        [
                            c[0],
                            c[0],
                            c[0] + self.VISDA.subarray_size[0],
                            c[0] + self.VISDA.subarray_size[0],
                            c[0],
                        ],
                        **kwargs,
                    )

                [plot_corner(c, ax, color="r") for c in self.VISDA.corners]
        return fig

    def save_visda(
        self,
        outfile: str = "pandora_"
        + Time.now().strftime("%Y-%m-%dT%H:%M:%S")
        + "_l1_visda.fits",
        rois: bool = False,
        overwrite: bool = True,
    ):
        """Function to save FFIs in the FITS format"""
        if not hasattr(self, "ffis"):
            raise AttributeError("Please create FFIs first with .get_FFIs() command!")

        corstime = int(
            np.floor((self.obstime - Time("2000-01-01T12:00:00", scale="utc")).sec)
        )
        finetime = int(corstime % 1 * 10**9 // 1)

        primary_kwds = {
            "EXTNAME": ("PRIMARY", "name of extension"),
            "NEXTEND": (2, "number of standard extensions"),
            "SIMDATA": (True, "simulated data"),
            "SCIDATA": (False, "science data"),
            "TELESCOP": ("NASA Pandora", "telescope"),
            "INSTRMNT": ("VISDA", "instrument"),
            "CREATOR": ("Pandora DPC", "creator of this product"),
            "CRSOFTV": ("v" + str(__version__), "creator software version"),
            "TARG_RA": (self.ra.value, "target right ascension [deg]"),
            "TARG_DEC": (self.dec.value, "target declination [deg]"),
            "FRMSREQD": (self.ffi_nframes, "number of frames requested"),
            "FRMSCLCT": (self.ffi_nframes, "number of frames collected"),
            "NUMCOAD": (1, "number of frames coadded"),
            "FRMTIME": (
                self.ffi_nreads * self.VISDA.integration_time.value,
                "time in each frame [s]",
            ),
            "EXPDELAY": (-1, "exposure time delay [ms]"),
            "RICEX": (-1, "bit noise parameter for Rice compression"),
            "RICEY": (-1, "bit noise parameter for Rice compression"),
            "CORSTIME": (
                corstime,
                "seconds since the TAI Epoch (12PM Jan 1, 2000)",
            ),
            "FINETIME": (finetime, "nanoseconds added to CORSTIME seconds"),
        }

        if rois:
            n_arrs, frames, nrows, ncols = self.subarrays.shape

            # Find the next largest perfect square from the number of subarrays given
            next_square = int(np.ceil(np.sqrt(n_arrs)) ** 2)
            sq_sides = int(np.sqrt(next_square))

            # Pad the subarrays with addtional subarrays full of zeros up to the next perfect square
            subarrays = self.subarrays
            padding = np.zeros((next_square - n_arrs, frames, nrows, ncols), dtype=int)
            subarrays = np.append(subarrays, padding, axis=0)

            image_data = (
                subarrays.reshape(frames, sq_sides, sq_sides, nrows, ncols)
                .swapaxes(2, 3)
                .reshape(frames, sq_sides * nrows, sq_sides * ncols)
            )

            roi_data = Table(self.VISDA.corners)

            roitable_kwds = {
                "NAXIS": (2, "number of array dimensions"),
                "NAXIS1": (
                    len(self.VISDA.corners[0]),
                    "length of dimension 1",
                ),
                "NAXIS2": (len(self.VISDA.corners), "length of dimension 2"),
                "PCOUNT": (0, "number of group parameters"),
                "GCOUNT": (1, "number of groups"),
                "TFIELDS": (2, "number of table fields"),
                "TTYPE1": ("Column", "table field 1 type"),
                "TFORM1": ("I21", "table field 1 format"),
                "TUNIT1": ("pix", "table field 1 unit"),
                "TBCOL1": (1, ""),
                "TTYPE2": ("Row", "table field 2 type"),
                "TFORM2": ("I21", "table field 2 format"),
                "TUNIT2": ("pix", "table field 2 unit"),
                "TBCOL2": (22, ""),
                "EXTNAME": ("ROITABLE", "name of extension"),
                "NROI": (
                    len(self.VISDA.corners),
                    "number of regions of interest",
                ),
                "ROISTRTX": (
                    -1,
                    "region of interest origin position in column",
                ),
                "ROISTRTY": (-1, "region of interest origin position in row"),
                "ROISIZEX": (-1, "region of interest size in column"),
                "ROISIZEY": (-1, "region of interest size in row"),
            }
        else:
            image_data = self.ffis

        image_kwds = {
            "NAXIS": (3, "number of array dimensions"),
            "NAXIS1": (
                image_data.shape[1],
                "first axis size",
            ),  # might need to change these
            "NAXIS2": (image_data.shape[2], "second axis size"),
            "NAXIS3": (image_data.shape[0], "third axis size"),
            "EXTNAME": ("SCIENCE", "extension name"),
            "TTYPE1": ("COUNTS", "data title: raw pixel counts"),
            "TFORM1": ("J", "data format: images of unsigned 32-bit integers"),
            "TUNIT1": ("count", "data units: count"),
        }

        if rois:
            save_to_FITS(
                image_data,
                outfile,
                primary_kwds,
                image_kwds,
                roitable=True,
                roitable_kwds=roitable_kwds,
                roi_data=roi_data,
                overwrite=overwrite,
            )
        else:
            save_to_FITS(
                image_data,
                outfile,
                primary_kwds,
                image_kwds,
                overwrite=overwrite,
            )
