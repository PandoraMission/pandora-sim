"""Abstract base class for a Simulator object"""
from abc import ABC, abstractmethod
from copy import deepcopy

import astropy.units as u
import numpy as np
import pandas as pd
import pandorapsf as pp
import pandorasat as ps
from pandorasat import get_logger

from .docstrings import add_docstring

__all__ = ["Sim"]

logger = get_logger("pandora-sim")


class Sim(ABC):
    def __repr__(self):
        if hasattr(self, "ra"):
            return f"{self.detector.name} Simulation [({self.ra:.3f}, {self.dec:.3f})]"
        else:
            return f"{self.detector.name} Simulation [no pointing information]"

    def __init__(self, detector):
        self.detector = detector
        logger.start_spinner("Loading PSF..")
        self.psf = pp.PSF.from_name(self.detector.name)
        logger.stop_spinner()

    @add_docstring("ra", "dec", "theta")
    @abstractmethod
    def point(self, ra: u.Quantity, dec: u.Quantity, roll: u.Quantity):
        self.ra, self.dec, self.roll = ra, dec, roll
        self.wcs = self.detector.get_wcs(self.ra, self.dec)

        logger.start_spinner("Finding nearby sources...")
        self.source_catalog = self._get_source_catalog()
        self.locations = np.asarray(
            [self.source_catalog.row, self.source_catalog.column]
        ).T
        logger.stop_spinner()

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
