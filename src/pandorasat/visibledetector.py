"""Holds metadata and methods on Pandora VISDA"""
# Standard library
import warnings

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import votable, fits

from . import PACKAGEDIR
from .detector import Detector
from .psf import interpfunc
from .wcs import get_wcs

from . import PACKAGEDIR
from .psf import PSF
from glob import glob


class VisibleDetector(Detector):
    """Pandora Visible Detector"""

    def _setup(self):
        """Some detector specific functions to run on initialization"""
        self.psf = PSF.from_file(
            f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
            transpose=self.transpose_psf,
        )
        self.flat = fits.open(
            np.sort(
                np.atleast_1d(glob(f"{PACKAGEDIR}/data/flatfield_VISDA*.fits"))
            )[-1]
        )[1].data[:, :1024]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.wcs = get_wcs(
                self,
                target_ra=self.ra,
                target_dec=self.dec,
                theta=self.theta,
                distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv"
            )

    @property
    def _dispersion_df(self):
        return pd.read_csv(f"{PACKAGEDIR}/data/pixel_vs_wavelength_vis.csv")

    def qe(self, wavelength):
        """
        Calculate the quantum efficiency of the detector.

        Parameters:
            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

        Returns:
            qe (npt.NDArray): Array of the quantum efficiency of the detector
        """
        df = (
            votable.parse(f"{PACKAGEDIR}/data/Pandora.Pandora.Visible.xml")
            .get_first_table()
            .to_table()
            .to_pandas()
        )
        wav, transmission = np.asarray(df.Wavelength) * u.angstrom, np.asarray(
            df.Transmission
        )
        return (
            np.interp(wavelength, wav, transmission, left=0, right=0)
            * u.DN
            / u.photon
        )

    @property
    def dark(self):
        return 1 * u.electron / u.second

    @property
    def read_noise(self):
        return 1.5 * u.electron

    @property
    def bias(self):
        return 100 * u.electron

    @property
    def integration_time(self):
        return 0.2 * u.second

    @property
    def fieldstop_radius(self):
        return 0.3 * u.deg

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.714

    def diagnose(
        self,
        n=3,
        image_type="PSF",
        wavelength=0.54 * u.micron,
        temperature=-10 * u.deg_C,
    ):
        if not (n % 2) == 1:
            n += 1
        fig, ax = plt.subplots(n, n, figsize=(n * 2, n * 2))
        for x1, y1 in (
            np.asarray(((np.mgrid[:n, :n] - n // 2) * (600 / (n // 2))))
            .reshape((2, n**2))
            .T
        ):
            jdx = int(x1 // (600 / (n // 2)) + n // 2)
            idx = int(-y1 // (600 / (n // 2)) + n // 2)
            point = (x1, y1, 0.5, 10)
            if image_type.lower() == "psf":
                y, x, f = (
                    self.psf.psf_row.value,
                    self.psf.psf_column.value,
                    self.psf.psf(point),
                )
                ax[idx, jdx].set(xticklabels=[], yticklabels=[])
            elif image_type.lower() == "prf":
                y, x, f = self.psf.prf(point)
                if idx < (n - 1):
                    ax[idx, jdx].set(xticklabels=[])
                if jdx >= 1:
                    ax[idx, jdx].set(yticklabels=[])
            #             if jdx >= 1:
            #                 ax[idx, jdx].set(yticklabels=[])
            else:
                raise ValueError(
                    "No such image type. Choose from `'PSF'`, or `'PRF'.`"
                )
            ax[idx, jdx].pcolormesh(
                x,
                y,
                f,
                vmin=0,
                vmax=[0.1 if image_type.lower() == "prf" else 0.005][0],
            )
        ax[n // 2, 0].set(ylabel="Y Pixel")
        ax[n - 1, n // 2].set(xlabel="X Pixel")
        ax[0, n // 2].set(title=image_type.upper())
        return fig

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
    #             distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv"
    #             if distortion
    #             else None,
    #         )
    #     return wcs

    def get_fastPRF(self, wavelength, temperature, res=7, sub_res=5):
        """Returns a function which will evaluate the PRF on a grid, fast.

        Parameters
        ----------
        wavelength : float with astropy.units
            The wavelength to evaluate the PRF at
        temperature : float with astropy.units
            The temperature to evaluate the PRF at
        res: int
            The resolution of the interpolation across the detector.
        sub_res:
            The number of samples in sub pixel space

        Returns
        -------
        fastPRF: function
            Function that will return the PRF
        """

        # Grid of Pixel Locations
        xs = np.arange(
            int(np.floor(self.psf.psf_column.min().value)) - 1,
            int(np.ceil(self.psf.psf_column.max().value)) + 2,
            1,
        )
        ys = np.arange(
            int(np.floor(self.psf.psf_row.min().value)) - 1,
            int(np.ceil(self.psf.psf_row.max().value)) + 2,
            1,
        )

        def get_grid(prf_point, res=5):
            grid = np.zeros((res, res, ys.shape[0], xs.shape[0]))
            for idx, col in enumerate(np.arange(0, 1, 1 / res)):
                for jdx, row in enumerate(np.arange(0, 1, 1 / res)):
                    y1, x1, ar = self.psf.prf(prf_point, location=(row, col))
                    k = np.asarray(
                        np.meshgrid(np.in1d(ys, y1), np.in1d(xs, x1), indexing='ij')
                    ).all(axis=0)
                    grid[jdx, idx, k] = ar.ravel()
            return grid

        # Positions on detector
        x = np.linspace(
            int(np.ceil(self.psf.column1d.min().value)),
            int(np.floor(self.psf.column1d.max().value)),
            res,
        )
        y = np.linspace(
            int(np.ceil(self.psf.row1d.min().value)),
            int(np.floor(self.psf.row1d.max().value)),
            res,
        )

        # Sub Pixels
        xp, yp = np.arange(0, 1, 1 / sub_res), np.arange(0, 1, 1 / sub_res)

        # Build a grid of evaluated PRFs at each location
        grid = np.zeros((sub_res, sub_res, res, res, ys.shape[0], xs.shape[0]))
        for idx, col in enumerate(x):
            for jdx, row in enumerate(y):
                prf_point = (row, col, wavelength, temperature)
                grid[:, :, jdx, idx, :] = get_grid(prf_point, sub_res)

        grid = grid.transpose([4, 5, 2, 3, 0, 1])

        # Function to get PRF at any given location
        # Will interpolate across the detector, but will return the closest match in subpixel space

        def fastPRF(rowloc: float, colloc: float):
            """Function to get the PRF.

            Parameters
            ----------
            rowloc: float
                row/y location on the detector
            colloc: float
                column/x location on the detector

            Returns
            -------
            row: np.ndarray
                row/y location on the detector
            column: np.ndarray
                column/x location on the detector
            z : np.ndarray
                2D flux of the PRF on the detector
            """
            return (
                ys + (rowloc - (rowloc % 1)),
                xs + (colloc - (colloc % 1)),
                interpfunc(
                    colloc,
                    x,
                    interpfunc(
                        rowloc,
                        y,
                        grid[
                            :,
                            :,
                            :,
                            :,
                            np.argmin(np.abs(yp - (rowloc % 1))),
                            np.argmin(np.abs(xp - (colloc % 1))),
                        ],
                    ),
                ),
            )

        return fastPRF

    def get_background_light_estimate(self, ra, dec):
        """Placeholder, will estimate the background light at different locations?
        Background in one integration...!
        """
        bkg = u.Quantity(
            np.zeros(self.shape, int), unit="electron", dtype="int"
        )
        bkg[self.fieldstop] = u.Quantity(
            np.random.poisson(lam=2, size=self.fieldstop.sum()).astype(int),
            unit="electron",
            dtype="int",
        )
        return bkg

    def apply_gain(self, values: u.Quantity):
        """Applies a piecewise gain function"""
        x = np.atleast_1d(values)
        masks = np.asarray(
            [
                (x >= 0 * u.DN) & (x < 1e3 * u.DN),
                (x >= 1e3 * u.DN) & (x < 5e3 * u.DN),
                (x >= 5e3 * u.DN) & (x < 2.8e4 * u.DN),
                (x >= 2.8e4 * u.DN),
            ]
        )
        gain = np.asarray([0.52, 0.6, 0.61, 0.67]) * u.electron / u.DN
        return (masks * x[None, :] * gain[:, None]).sum(axis=0)
