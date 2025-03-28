"""Abstract base class for a Simulator object"""

from abc import ABC, abstractmethod
from copy import deepcopy

import astropy.units as u
import numpy as np
import pandas as pd
import pandorapsf as pp
import pandorasat as ps

from .docstrings import add_docstring
from . import logger

__all__ = ["Sim"]


class Sim(ABC):
    def __repr__(self):
        if hasattr(self, "ra"):
            return f"{self.detector.name} Simulation [({self.ra:.3f}, {self.dec:.3f})]"
        else:
            return f"{self.detector.name} Simulation [no pointing information]"

    def __init__(self, detector):
        self.detector = detector
        # logger.start_spinner("Loading PSF..")
        self.psf = pp.PSF.from_name(self.detector.name)
        # logger.stop_spinner()

    @add_docstring("ra", "dec", "theta")
    @abstractmethod
    def point(self, ra: u.Quantity, dec: u.Quantity, roll: u.Quantity):
        self.ra, self.dec, self.roll = ra, dec, roll
        self.wcs = self.detector.get_wcs(self.ra, self.dec, theta=self.roll)

        # logger.start_spinner("Finding nearby sources...")
        self.source_catalog = self._get_source_catalog()
        self.locations = np.asarray(
            [self.source_catalog.row, self.source_catalog.column]
        ).T
        # logger.stop_spinner()

    @add_docstring("source_catalog")
    @abstractmethod
    def from_source_catalog(self, source_catalog: pd.DataFrame):
        self.ra, self.dec, self.roll = 0 * u.deg, 0 * u.deg, 0 * u.deg
        self.wcs = self.detector.get_wcs(0 * u.deg, 0 * u.deg, theta=0 * u.deg)
        self.source_catalog = source_catalog
        self.locations = np.asarray(
            [self.source_catalog.row, self.source_catalog.column]
        ).T

    def world_to_pixel(self, ra, dec, distortion=True):
        """Helper function. This function ensures we keep the row-major convention in pandora-sim.

        Parameters:
        -----------
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

        Parameters:
        -----------
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
                column.to(u.pixel).value if isinstance(column, u.Quantity) else column,
                row.to(u.pixel).value if isinstance(row, u.Quantity) else row,
            ]
        ).T
        if distortion:
            return self.wcs.all_pix2world(coords, 0).T * u.deg
        else:
            return self.wcs.wcs_pix2world(coords, 0).T * u.deg

    def _get_source_catalog(self, distortion: bool = True, **kwargs) -> pd.DataFrame:
        """Gets the source catalog of an input target

        Parameters
        ----------
        target_name : str
            Target name to obtain catalog for.
        distortion : bool
            Whether to apply a distortion to the WCS when mapping RA and Dec of catalog targets
            to detector pixels. Default is True.
        **kwargs
            Additional arguments passed to the ps.utils.get_sky_catalog function.

        Returns
        -------
        sky_catalog: pd.DataFrame
            Pandas dataframe of all the sources near the target
        """

        # This is fixed for visda, for now
        if hasattr(self.detector, "subarray_size"):
            shape = self.detector.subarray_size
        else:
            shape = self.detector.shape
        radius = np.hypot(*np.asarray(shape) // 2)
        radius = ((radius * u.pixel) * self.detector.pixel_scale).to(u.deg).value

        # If there is a fieldstop, we can stop finding sources at that radius
        # if hasattr(self.detector, "fieldstop_radius"):
        #     fieldstop_radius = (
        #         (
        #             (self.detector.fieldstop_radius / self.detector.pixel_size)
        #             * self.detector.pixel_scale
        #         )
        #         .to(u.deg)
        #         .value
        #     )
        #     if fieldstop_radius < radius:
        #         radius = fieldstop_radius

        # Get location and magnitude data
        cat = ps.utils.get_sky_catalog(
            self.ra, self.dec, radius=radius * u.deg, **kwargs
        )
        ra, dec, mag = cat["coords"].ra.deg, cat["coords"].dec.deg, cat["bmag"]
        pix_coords = self.world_to_pixel(ra, dec, distortion=distortion)

        k = (
            np.abs(pix_coords[0] - shape[0] / 2)
            < (shape[0] / 2 + self.psf_shape[0] / 2)
        ) & (
            np.abs(pix_coords[1] - shape[1] / 2)
            < (shape[1] / 2 + self.psf_shape[1] / 2)
        )

        new_cat = deepcopy(cat)
        for key, item in new_cat.items():
            new_cat[key] = item[k]
        pix_coords, ra, dec, mag = (
            pix_coords[:, k],
            ra[k],
            dec[k],
            mag[k],
        )
        source_catalog = (
            pd.DataFrame(
                np.vstack(
                    [
                        ra,
                        dec,
                        mag,
                        *pix_coords,
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
                    "row",
                    "column",
                    "jmag",
                    "teff",
                    "logg",
                    "ruwe",
                    "ang_sep",
                ],
            )
            .drop_duplicates(["ra", "dec", "mag"])
            .reset_index(drop=True)
        )
        return source_catalog

    @property
    def psf_shape(self):
        p = self.detector.trace_pixel.value
        p = p.max() - p.min()
        return (p + 6, 6)

    def get_simple_cosmic_ray_image(
        self,
        cosmic_ray_rate=1000 / (u.second * u.cm**2),
        average_cosmic_ray_flux: u.Quantity = u.Quantity(1e3, unit="DN"),
        cosmic_ray_distance: u.Quantity = u.Quantity(0.01, unit=u.pixel / u.DN),
        image_shape=(2048, 2048),
    ):
        """Function to get a simple cosmic ray image

        This function has no basis in physics at all.
        The true rate of cosmic rays, the energy deposited, sampling distributions.
        All this function can do is put down fairly reasonable "tracks" that mimic the impact of cosmic rays, with some tuneable parameters to change the properties.

        """
        cosmic_ray_expectation = (
            cosmic_ray_rate
            * ((self.detector.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
            * self.detector.integration_time
        ).value
        ncosmics = np.random.poisson(cosmic_ray_expectation)
        im = np.zeros(image_shape, dtype=int)

        for ray in np.arange(ncosmics):
            # Random flux drawn from some exponential...
            cosmic_ray_counts = (
                np.random.exponential(average_cosmic_ray_flux.value)
                * average_cosmic_ray_flux.unit
            )

            # Random location
            xloc = np.random.uniform(0, image_shape[0])
            yloc = np.random.uniform(0, image_shape[1])
            # Random angle into the detector
            theta = np.random.uniform(
                -0.5 * np.pi, 0.5 * np.pi
            )  # radians from the top of the sensor?
            # Random angle around
            phi = np.random.uniform(0, 2 * np.pi)

            r = np.sin(theta) * (cosmic_ray_distance * cosmic_ray_counts).value

            x1, x2, y1, y2 = (
                xloc,
                xloc + (r * np.cos(phi)),
                yloc,
                yloc + (r * np.sin(phi)),
            )
            m = (y2 - y1) / (x2 - x1)
            c = y1 - (m * x1)

            xs, ys = np.sort([x1, x2]).astype(int), np.sort([y1, y2]).astype(int)
            xs, ys = (
                [xs[0], xs[1] if np.diff(xs) > 0 else xs[1] + 1],
                [
                    ys[0],
                    ys[1] if np.diff(ys) > 0 else ys[1] + 1,
                ],
            )

            coords = np.vstack(
                [
                    np.round(np.arange(*xs, 0.005)).astype(int),
                    np.round(m * np.arange(*xs, 0.005) + c).astype(int),
                ]
            ).T
            coords = coords[(coords[:, 1] >= ys[0]) & (coords[:, 1] <= ys[1])]
            if len(coords) == 0:
                continue
            fper_element = cosmic_ray_counts / len(coords)
            coords = coords[
                (
                    (coords[:, 0] >= 0)
                    & (coords[:, 0] < image_shape[0])
                    & (coords[:, 1] >= 0)
                    & (coords[:, 1] < image_shape[1])
                )
            ]
            coords, wcoords = np.unique(coords, return_counts=True, axis=0)
            im[coords[:, 0], coords[:, 1]] = np.random.poisson(
                (wcoords * fper_element).value
            )
        return u.Quantity(im, dtype=int, unit=u.DN)
