"""Holds methods on Pandora"""

# Standard library
from copy import deepcopy

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from matplotlib.patches import PathPatch
from matplotlib.path import Path

# from pandorasat.irdetector import NIRDetector
# from pandorasat.visibledetector import VisibleDetector
from pandorasat.hardware import Hardware
from pandorasat.orbit import Orbit

from . import PANDORASTYLE
from .irdetector import NIRDetector
# from .optics import Optics
# from .orbit import Orbit
from .utils import get_jitter, get_sky_catalog
from .visibledetector import VisibleDetector


# @dataclass
class PandoraSim(object):
    """Holds methods for simulating the full Pandora system.

    Args:
        NIRDA (IRDetector): Class of the NIRDA properties
        VISDA (IRDetector): Class of the VISDA properties
        Optics (IRDetector): Class of the Optics properties
        Orbit (IRDetector): Class of the Orbit properties
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
            # "NIR",
            ra,
            dec,
            theta,
            # 1.19 * u.arcsec / u.pixel,
            # 18.0 * u.um / u.pixel,
            False,
        )
        self.VISDA = VisibleDetector(
            # "Visible",
            ra,
            dec,
            theta,
            # 0.78 * u.arcsec / u.pixel,
            # 6.5 * u.um / u.pixel,
        )
        self.rowjitter_1sigma = rowjitter_1sigma
        self.coljitter_1sigma = coljitter_1sigma
        self.thetajitter_1sigma = thetajitter_1sigma
        self.jitter_timescale = jitter_timescale
        self._get_jitter()
        self.SkyCatalog = self.get_sky_catalog()

        nints = (duration.to(u.s) / self.VISDA.integration_time).value
        self.VISDA.time = (
            self.obstime.jd
            + np.arange(0, nints) / nints * duration.to(u.day).value
        )
        self.VISDA.rowj, self.VISDA.colj, self.VISDA.thetaj = (  # noqa
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.rowj),
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.colj),
            np.interp(self.VISDA.time, self.jitter.time, self.jitter.thetaj),
        )

        self.VISDA.corners, self.VISDA.Catalogs = self.get_vis_stars()
        self.NIRDA.Catalog = self.get_nirda_stars()

        # self.targetlist = pd.read_csv(f"{PACKAGEDIR}/data/targets.csv")

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
        self.jitter.time = (
            self.jitter.time * (u.second).to(u.day)
        ) + self.obstime.jd

    def get_sky_catalog(
        self,
        distortion: bool = True,
    ):
        """Gets the source catalog of an input target

        Parameters
        ----------
        target_name : str
            Target name to obtain catalog for.

        Returns
        -------
        sky_catalog: pd.DataFrame
            Pandas dataframe of all the sources near the target
        """

        # This is fixed for visda, for now
        radius = np.min(
            [
                (2 * self.VISDA.fieldstop_radius.to(u.deg).value ** 2) ** 0.5,
                (
                    2
                    * (
                        (2048 * u.pix * self.VISDA.pixel_scale).to(u.deg).value
                        / 2
                    )
                    ** 2
                )
                ** 0.5,
            ]
        )
        # catalog_data = Catalogs.query_object(target_name, radius=radius, catalog="TIC")
        # target_ra, target_dec = catalog_data[0][['ra', 'dec']].values()

        # Get location and magnitude data
        cat = get_sky_catalog(self.ra, self.dec, radius=radius * u.deg)

        ra, dec, mag = cat["coords"].ra.deg, cat["coords"].dec.deg, cat["bmag"]
        vis_pix_coords = self.VISDA.world_to_pixel(
            ra, dec, distortion=distortion
        )
        nir_pix_coords = self.NIRDA.world_to_pixel(
            ra, dec, distortion=distortion
        )
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

    def get_nirda_stars(self):
        cat = self.SkyCatalog.copy()
        cat[["nir_row", "nir_column"]] = self.NIRDA.world_to_pixel(
            cat.ra, cat.dec
        ).T
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
        dist = self.SkyCatalog["ang_sep"] > (
            self.VISDA.pixel_scale * 50 * u.pixel
        ).to(u.deg)
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
                np.asarray(cat.vis_row, dtype=int)
                - self.VISDA.subarray_size[0] // 2,
                np.asarray(cat.vis_column, dtype=int)
                - self.VISDA.subarray_size[1] // 2,
            ]
        ).T
        corners = np.vstack(
            [
                np.asarray(
                    [
                        self.VISDA.shape[0] // 2
                        - self.VISDA.subarray_size[0] // 2,
                        self.VISDA.shape[1] // 2
                        - self.VISDA.subarray_size[1] // 2,
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

    # def get_FFIs(
    #     self,
    #     prf_func,
    #     nreads=10,
    #     nt=40,
    #     include_noise=True,
    #     cosmic_rays=True,
    # ):
    #     # if prf_func is None:
    #     #     prf_func = self.VISDA.get_fastPRF(wavelength, temperature)

    #     # Spacecraft jitter motion
    #     time, rowj, colj, thetaj = get_jitter(
    #         self.rowjitter_1sigma.value,
    #         self.coljitter_1sigma.value,
    #         self.thetajitter_1sigma.value,
    #         correlation_time=self.jitter_timescale,
    #         nframes=nt * nreads,
    #         frame_time=self.VISDA.integration_time,
    #     )
    #     science_image = u.Quantity(
    #         np.zeros(
    #             (
    #                 nt,
    #                 *self.VISDA.shape,
    #             ),
    #             dtype=int,
    #         ),
    #         unit="electron",
    #         dtype="int",
    #     )
    #     vis_y, vis_x = self.VISDA.world_to_pixel(
    #         self.SkyCatalog.ra, self.SkyCatalog.dec
    #     )

    #     for jdx in tqdm(range(nt * nreads)):
    #         # Update the positions via the new WCS in each frame
    #         # vis_x, vis_y = (
    #         #     self.VISDA.wcs(
    #         #         target_ra + (xj[jdx] * self.VISDA.pixel_scale).to(u.deg),
    #         #         target_dec + (yj[jdx] * self.VISDA.pixel_scale).to(u.deg),
    #         #         theta + thetaj[jdx],
    #         #         distortion=distortion,
    #         #     )
    #         #     .all_world2pix(
    #         #         np.vstack([self.SkyCatalog.ra, self.SkyCatalog.dec]).T, 1
    #         #     )
    #         #     .T
    #         # )

    #         for idx, s in self.SkyCatalog.iterrows():
    #             #            for idx in range(len(self.SkyCatalog)):
    #             if (
    #                 (vis_x[idx] < 0)
    #                 | (vis_x[idx] > self.VISDA.shape[1])
    #                 | (vis_y[idx] < 0)
    #                 | (vis_y[idx] > self.VISDA.shape[0])
    #             ):
    #                 continue

    #             x, y = (
    #                 colj[jdx].value + vis_x[idx] - self.VISDA.shape[1] // 2,
    #                 rowj[jdx].value + vis_y[idx] - self.VISDA.shape[0] // 2,
    #             )

    #             y1, x1, f = prf_func(y, x)
    #             Y, X = np.asarray(
    #                 np.meshgrid(
    #                     y1 + self.VISDA.shape[0] // 2,
    #                     x1 + self.VISDA.shape[1] // 2,
    #                     indexing="ij",
    #                 )
    #             ).astype(int)
    #             #                print(X.mean(), Y.mean())
    #             k = (
    #                 (X >= 0)
    #                 & (X < self.VISDA.shape[1])
    #                 & (Y >= 0)
    #                 & (Y < self.VISDA.shape[0])
    #             )
    #             science_image[jdx // nreads, Y[k], X[k]] += u.Quantity(
    #                 np.random.poisson(
    #                     self.VISDA.apply_gain(
    #                         f[k]
    #                         * (
    #                             (s.vis_counts * u.DN / u.second)
    #                             * self.VISDA.integration_time.to(u.second)
    #                         )
    #                     ).value
    #                 ),
    #                 unit="electron",
    #                 dtype=int,
    #             )

    #     if cosmic_rays:
    #         # This is the worst rate we expect, from the SAA
    #         cosmic_ray_rate = 1000 / (u.second * u.cm**2)
    #         cosmic_ray_expectation = (
    #             cosmic_ray_rate
    #             * ((self.VISDA.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
    #             * self.VISDA.integration_time
    #         ).value

    #         for jdx in range(nt):
    #             science_image[jdx] += get_simple_cosmic_ray_image(
    #                 cosmic_ray_expectation=cosmic_ray_expectation,
    #                 gain_function=self.VISDA.apply_gain,
    #                 image_shape=self.VISDA.shape,
    #             )

    #     # Apply the flat-field
    #     science_image = u.Quantity(
    #         science_image.value.astype(float) * self.VISDA.flat[None, :, :],
    #         dtype=int,
    #         unit="electron",
    #     )

    #     # electrons in a read
    #     #        science_image *= u.electron

    #     if include_noise:
    #         # # background light?
    #         science_image += self.VISDA.get_background_light_estimate(
    #             self.SkyCatalog.loc[0, "ra"], self.SkyCatalog.loc[0, "dec"]
    #         )

    #     # time integrate
    #     science_image *= nreads

    #     # fieldstop
    #     science_image *= self.VISDA.fieldstop.astype(int)

    #     if include_noise:
    #         # noise
    #         for jdx in range(nt):
    #             noise = np.zeros(self.VISDA.shape, int)
    #             noise = np.random.normal(
    #                 loc=self.VISDA.bias.value,
    #                 scale=self.VISDA.read_noise.value,
    #                 size=self.VISDA.shape,
    #             )
    #             noise += np.random.poisson(
    #                 lam=(self.VISDA.dark * self.VISDA.integration_time).value,
    #                 size=self.VISDA.shape,
    #             )

    #             science_image[jdx] += u.Quantity(
    #                 noise, unit="electron", dtype=int
    #             )
    #     return time, rowj, colj, thetaj, science_image

    def plot_footprint(self, ax=None):
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

    def plot_FFI(
        self,
        nreads=10,
        include_cosmics=False,
        include_noise=True,
        figsize=(10, 8),
        subarrays=True,
        **kwargs,
    ):
        _, ffis = self.VISDA.get_FFIs(
            self.SkyCatalog,
            nreads=nreads,
            nframes=1,
            include_cosmics=include_cosmics,
            include_noise=include_noise,
            #            freeze_dimensions=freeze_dimensions,
        )
        with plt.style.context(PANDORASTYLE):
            vmin = kwargs.pop("vmin", self.VISDA.bias.value)
            vmax = kwargs.pop("vmax", self.VISDA.bias.value + 20)
            cmap = kwargs.pop("cmap", "Greys_r")
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                ffis[0],
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
