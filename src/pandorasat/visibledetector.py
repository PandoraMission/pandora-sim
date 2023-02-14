"""Holds metadata and methods on Pandora VISDA"""
# Standard library
import warnings

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits, votable
from astropy.wcs import WCS

from . import PACKAGEDIR
from .detector import Detector
from .psf import interpfunc
from .wcs import get_wcs


class VisibleDetector(Detector):
    """Pandora Visible Detector"""

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
                x, y, f = (
                    self.psf.psf_x.value,
                    self.psf.psf_y.value,
                    self.psf.psf(point),
                )
                ax[idx, jdx].set(xticklabels=[], yticklabels=[])
            elif image_type.lower() == "prf":
                x, y, f = self.psf.prf(point)
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
                f.T,
                vmin=0,
                vmax=[0.1 if image_type.lower() == "prf" else 0.005][0],
            )
        ax[n // 2, 0].set(ylabel="Y Pixel")
        ax[n - 1, n // 2].set(xlabel="X Pixel")
        ax[0, n // 2].set(title=image_type.upper())
        return fig

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
                distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv" if distortion else None,
            )
        return wcs

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

        def get_grid(prf_point, res=5):
            grid = np.zeros((res, res, xs.shape[0], ys.shape[0]))
            for idx, col in enumerate(np.arange(0, 1, 1 / res)):
                for jdx, row in enumerate(np.arange(0, 1, 1 / res)):
                    x1, y1, ar = self.psf.prf(prf_point, location=(col, row))
                    k = np.asarray(
                        np.meshgrid(np.in1d(ys, y1), np.in1d(xs, x1))
                    ).all(axis=0)
                    grid[idx, jdx, k] = ar.ravel()
            return grid

        # Grid of Pixel Locations
        xs = np.arange(
            int(np.floor(self.psf.psf_x.min().value)) - 1,
            int(np.ceil(self.psf.psf_x.max().value)) + 2,
            1,
        )
        ys = np.arange(
            int(np.floor(self.psf.psf_y.min().value)) - 1,
            int(np.ceil(self.psf.psf_y.max().value)) + 2,
            1,
        )

        # Positions on detector
        x = np.linspace(
            int(np.ceil(self.psf.x1d.min().value)),
            int(np.floor(self.psf.x1d.max().value)),
            res,
        )
        y = np.linspace(
            int(np.ceil(self.psf.y1d.min().value)),
            int(np.floor(self.psf.y1d.max().value)),
            res,
        )

        # Sub Pixels
        xp, yp = np.arange(0, 1, 1 / sub_res), np.arange(0, 1, 1 / sub_res)

        # Build a grid of evaluated PRFs at each location
        grid = np.zeros((sub_res, sub_res, res, res, xs.shape[0], ys.shape[0]))
        for idx, col in enumerate(x):
            for jdx, row in enumerate(y):
                prf_point = (col, row, wavelength, temperature)
                grid[:, :, idx, jdx, :] = get_grid(prf_point, sub_res)

        grid = grid.transpose([4, 5, 2, 3, 0, 1])

        # Function to get PRF at any given location
        # Will interpolate across the detector, but will return the closest match in subpixel space

        def fastPRF(xloc: float, yloc: float):
            """Function to get the PRF.

            Parameters
            ----------
            xloc: float
                x location on the detector
            yloc: float
                y location on the detector

            Returns
            -------
            x: np.ndarray
                x location on the detector
            y: np.ndarray
                y location on the detector
            z : np.ndarray
                2D flux of the PRF on the detector
            """
            return (
                xs + (xloc - (xloc % 1)),
                ys + (yloc - (yloc % 1)),
                interpfunc(
                    yloc,
                    y,
                    interpfunc(
                        xloc,
                        x,
                        grid[
                            :,
                            :,
                            :,
                            :,
                            np.argmin(np.abs(xp - (xloc % 1))),
                            np.argmin(np.abs(yp - (yloc % 1))),
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
        masks = np.asarray([(x >= 0 * u.DN) & (x < 1e3 * u.DN),
                (x >= 1e3 * u.DN) & (x < 5e3 * u.DN),
                (x >= 5e3 * u.DN) & (x < 2.8e4 * u.DN),
                (x >= 2.8e4 * u.DN)])
        gain = np.asarray([0.52, 0.6, 0.61, 0.67])*u.electron/u.DN
        return (masks * x[None, :] * gain[:, None]).sum(axis=0)
