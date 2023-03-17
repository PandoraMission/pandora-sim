"""Holds metadata and methods on Pandora"""

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from . import PACKAGEDIR
from .irdetector import NIRDetector
from .optics import Optics
from .orbit import Orbit
from .utils import get_jitter, get_simple_cosmic_ray_image, get_sky_catalog
from .visibledetector import VisibleDetector


# @dataclass
class PandoraSat(object):
    """Holds information and methods for the full Pandora system.

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
    ):
        self.ra, self.dec, self.theta = ra, dec, theta
        self.Orbit = Orbit()
        self.Optics = Optics()
        self.NIRDA = NIRDetector(
            "NIR",
            ra,
            dec,
            theta,
            1.19 * u.arcsec / u.pixel,
            18.0 * u.um / u.pixel,
            False,
        )
        self.VISDA = VisibleDetector(
            "Visible",
            ra,
            dec,
            theta,
            0.78 * u.arcsec / u.pixel,
            6.5 * u.um / u.pixel,
        )
        self.targetlist = pd.read_csv(f"{PACKAGEDIR}/data/targets.csv")

    def __repr__(self):
        return f"Pandora Sat (RA: {np.round(self.ra, 3)}, Dec: {np.round(self.dec, 3)}, theta: {np.round(self.theta, 3)})"

    def get_sky_catalog(
        self,
        target_ra: u.Quantity,
        target_dec: u.Quantity,
        theta: u.Quantity,
        magnitude_range: tuple = (-3, 16),
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
        ra, dec, mag = np.asarray(
            get_sky_catalog(
                target_ra.value
                if isinstance(target_ra, u.Quantity)
                else target_ra,
                target_dec.value
                if isinstance(target_dec, u.Quantity)
                else target_dec,
                radius=radius,
                magnitude_range=magnitude_range,
            )
        ).T
        k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
        ra, dec, mag = ra[k], dec[k], mag[k]
        # vis_pix_coords = self.VISDA.wcs(
        #     target_ra, target_dec, theta, distortion=distortion
        # ).all_world2pix(np.vstack([ra, dec]).T, 1)
        # nir_pix_coords = self.NIRDA.wcs(
        #     target_ra, target_dec, theta, distortion=distortion
        # ).all_world2pix(np.vstack([ra, dec]).T, 1)
        vis_pix_coords = self.VISDA.world_to_pixel(
            ra, dec, distortion=distortion
        )
        nir_pix_coords = self.NIRDA.world_to_pixel(
            ra, dec, distortion=distortion
        )

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
            ],
        )
        return source_catalog

    def get_sky_images(
        self,
        target_ra: u.Quantity,
        target_dec: u.Quantity,
        theta: u.Quantity,
        magnitude_range: tuple = (-3, 16),
        distortion: bool = True,
        wavelength=0.54 * u.micron,
        temperature=10 * u.deg_C,
        nreads=10,
        nt=40,
        rowjitter_1sigma=2 * u.pixel,
        coljitter_1sigma=2 * u.pixel,
        thetajitter_1sigma=0.0005 * u.deg,
        jitter_timescale=1 * u.second,
        include_noise=True,
        prf_func=None,
        cosmic_rays=True,
    ):
        source_catalog = self.get_sky_catalog(
            target_ra=target_ra,
            target_dec=target_dec,
            theta=theta,
            magnitude_range=magnitude_range,
            distortion=distortion,
        )
        if prf_func is None:
            prf_func = self.VISDA.get_fastPRF(wavelength, temperature)

        # Spacecraft jitter motion
        time, rowj, colj, thetaj = get_jitter(
            rowjitter_1sigma.value,
            coljitter_1sigma.value,
            thetajitter_1sigma.value,
            correlation_time=jitter_timescale,
            nframes=nt * nreads,
            frame_time=self.VISDA.integration_time,
        )
        science_image = u.Quantity(
            np.zeros(
                (
                    nt,
                    *self.VISDA.shape,
                ),
                dtype=int,
            ),
            unit="electron",
            dtype="int",
        )
        vis_y, vis_x = self.VISDA.world_to_pixel(
            source_catalog.ra, source_catalog.dec
        )

        for jdx in range(nt * nreads):
            # Update the positions via the new WCS in each frame
            # vis_x, vis_y = (
            #     self.VISDA.wcs(
            #         target_ra + (xj[jdx] * self.VISDA.pixel_scale).to(u.deg),
            #         target_dec + (yj[jdx] * self.VISDA.pixel_scale).to(u.deg),
            #         theta + thetaj[jdx],
            #         distortion=distortion,
            #     )
            #     .all_world2pix(
            #         np.vstack([source_catalog.ra, source_catalog.dec]).T, 1
            #     )
            #     .T
            # )

            for idx, s in source_catalog.iterrows():
                #            for idx in range(len(source_catalog)):
                if (
                    (vis_x[idx] < 0)
                    | (vis_x[idx] > self.VISDA.shape[1])
                    | (vis_y[idx] < 0)
                    | (vis_y[idx] > self.VISDA.shape[0])
                ):
                    continue

                x, y = (
                    colj[jdx].value + vis_x[idx] - self.VISDA.shape[1] // 2,
                    rowj[jdx].value + vis_y[idx] - self.VISDA.shape[0] // 2,
                )

                y1, x1, f = prf_func(y, x)
                Y, X = np.asarray(
                    np.meshgrid(
                        y1 + self.VISDA.shape[0] // 2,
                        x1 + self.VISDA.shape[1] // 2,
                        indexing="ij",
                    )
                ).astype(int)
                #                print(X.mean(), Y.mean())
                k = (
                    (X >= 0)
                    & (X < self.VISDA.shape[1])
                    & (Y >= 0)
                    & (Y < self.VISDA.shape[0])
                )
                science_image[jdx // nreads, Y[k], X[k]] += u.Quantity(
                    np.random.poisson(
                        self.VISDA.apply_gain(
                            f[k]
                            * (
                                (s.vis_counts * u.DN / u.second)
                                * self.VISDA.integration_time.to(u.second)
                            )
                        ).value
                    ),
                    unit="electron",
                    dtype=int,
                )

        if cosmic_rays:
            # This is the worst rate we expect, from the SAA
            cosmic_ray_rate = 1000 / (u.second * u.cm**2)
            cosmic_ray_expectation = (
                cosmic_ray_rate
                * ((self.VISDA.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
                * self.VISDA.integration_time
            ).value

            for jdx in range(nt):
                science_image[jdx] += get_simple_cosmic_ray_image(
                    cosmic_ray_expectation=cosmic_ray_expectation,
                    gain_function=self.VISDA.apply_gain,
                    image_shape=self.VISDA.shape,
                )

        # Apply the flat-field
        science_image = u.Quantity(
            science_image.value.astype(float) * self.VISDA.flat[None, :, :],
            dtype=int,
            unit="electron",
        )

        # electrons in a read
        #        science_image *= u.electron

        if include_noise:
            # # background light?
            science_image += self.VISDA.get_background_light_estimate(
                source_catalog.loc[0, "ra"], source_catalog.loc[0, "dec"]
            )

        # time integrate
        science_image *= nreads

        # fieldstop
        science_image *= self.VISDA.fieldstop.astype(int)

        if include_noise:
            # noise
            for jdx in range(nt):
                noise = np.zeros(self.VISDA.shape, int)
                noise = np.random.normal(
                    loc=self.VISDA.bias.value,
                    scale=self.VISDA.read_noise.value,
                    size=self.VISDA.shape,
                )
                noise += np.random.poisson(
                    lam=(self.VISDA.dark * self.VISDA.integration_time).value,
                    size=self.VISDA.shape,
                )

                science_image[jdx] += u.Quantity(
                    noise, unit="electron", dtype=int
                )
        return time, rowj, colj, thetaj, science_image

    def footprint_plot(self, fig=None):
        if fig is None:
            fig, ax = plt.subplots()
        ax.scatter(
            self.ra.value,
            self.dec.value,
            edgecolor="k",
            facecolor="None",
            zorder=10,
            label="Target",
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
        ax.set_xlim(self.ra.value - 1, self.ra.value + 1)
        ax.set_ylim(self.dec.value - 1, self.dec.value + 1)
        ax.legend(frameon=True)
        ax.set_aspect(1)
        ax.set(
            xlabel="RA",
            ylabel="Dec",
            title=f"RA:{self.ra} Dec:{self.dec}, theta:{self.theta}",
        )
        return fig
