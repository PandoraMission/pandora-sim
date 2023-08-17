"""Holds metadata and methods on Pandora VISDA"""
# Standard library
import warnings
from glob import glob

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits, votable
from tqdm import tqdm

from . import PACKAGEDIR, PANDORASTYLE
from .detector import Detector
from .psf import PSF, interpfunc
from .utils import get_simple_cosmic_ray_image
from .wcs import get_wcs


class VisibleDetector(Detector):
    """Pandora Visible Detector"""

    def _setup(self):
        """Some detector specific functions to run on initialization"""
        self.shape = (2048, 2048)
        self.psf = PSF.from_file(
            f"{PACKAGEDIR}/data/pandora_vis_20220506.fits",
            transpose=self.transpose_psf,
        )
        self.flat = fits.open(
            np.sort(
                np.atleast_1d(glob(f"{PACKAGEDIR}/data/flatfield_VISDA*.fits"))
            )[-1]
        )[1].data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.wcs = get_wcs(
                self,
                target_ra=self.ra,
                target_dec=self.dec,
                theta=self.theta,
                distortion_file=f"{PACKAGEDIR}/data/fov_distortion.csv",
            )
        if hasattr(self, "fieldstop_radius"):
            C, R = (
                np.mgrid[
                    : self.shape[0],
                    : self.shape[1],
                ]
                - np.hstack(
                    [
                        self.shape[0],
                        self.shape[1],
                    ]
                )[:, None, None]
                / 2
            )
            r = (self.fieldstop_radius / self.pixel_scale).to(u.pix).value
            self.fieldstop = ~((np.abs(C) >= r) | (np.abs(R) >= r))

        # Blur our the PSF
        # We assume that there's low level jitter that "blurs" the PSF to some extent.
        s = self.psf.psf_flux.shape
        self.psf.psf_flux = np.asarray(
            [
                [
                    [
                        [
                            convolve(
                                self.psf.psf_flux[:, :, idx, jdx, kdx, ldx],
                                Gaussian2DKernel(1),
                            )
                            for idx in range(s[2])
                        ]
                        for jdx in range(s[3])
                    ]
                    for kdx in range(s[4])
                ]
                for ldx in range(s[5])
            ]
        ).transpose([4, 5, 3, 2, 1, 0])

        # ROW COLUMN JUST LIKE PYTHON
        self.subarray_size = (50, 51)
        # COLUMN, ROW
        self.subarray_row, self.subarray_column = np.meshgrid(
            +np.arange(self.subarray_size[0]),
            +np.arange(self.subarray_size[1]),
            indexing="ij",
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
            # elif image_type.lower() == "prf":
            #     y, x, f = self.psf.prf(point)
            #     if idx < (n - 1):
            #         ax[idx, jdx].set(xticklabels=[])
            #     if jdx >= 1:
            #         ax[idx, jdx].set(yticklabels=[])
            # #             if jdx >= 1:
            # #                 ax[idx, jdx].set(yticklabels=[])
            else:
                raise ValueError("No such image type. Choose from `'PSF'`.")
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
                        np.meshgrid(
                            np.in1d(ys, y1), np.in1d(xs, x1), indexing="ij"
                        )
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

        grid /= np.trapz(np.trapz(grid, ys, axis=0), axis=0)[
            None, None, :, :, :, :
        ]

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

    def get_background_light_estimate(self, ra, dec, duration, shape=None):
        """Placeholder, will estimate the background light at different locations?
        Background in one integration...!
        """
        # bkg = u.Quantity(
        #     np.zeros(shape, int), unit="electron", dtype="int"
        # )
        # bkg[self.fieldstop] = u.Quantity(
        #     np.random.poisson(lam=2, size=self.fieldstop.sum()).astype(int),
        #     unit="electron",
        #     dtype="int",
        # )
        # This is an approximate value assuming a zodi of ~22 Vmag
        bkg_rate = 1.2*u.electron/u.second
        if shape is None:
            shape = self.shape
        bkg = u.Quantity(
            np.random.poisson(lam=(bkg_rate*duration).to(u.electron).value, size=shape).astype(int),
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
        if values.ndim == 1:
            gain = gain[:, None]
        if values.ndim == 2:
            gain = gain[:, None, None]
        return u.Quantity(
            (masks * x[None, :] * gain).sum(axis=0), dtype=int, unit=u.electron
        )

    def prf(
        self,
        row=1024,
        col=1024,
        wavelength=None,
        temperature=None,
        shape=None,
        corner=(1004, 1004),
        freeze_dimensions=[2, 3],
        return_locs=False,
    ):
        if shape is None:
            shape = self.subarray_size

        # If outside the interp edges just return the edge
        point = (
            row - self.shape[0] // 2,
            col - self.shape[1] // 2,
            wavelength,
            temperature,
        )
        dr, dc = 0, 0
        if point[0] > self.psf.row1d[-1].value:
            dr = point[0] - self.psf.row1d[-1].value + 1
        if point[0] < self.psf.row1d[0].value:
            dr = point[0] - self.psf.row1d[0].value - 1
        if point[1] > self.psf.column1d[-1].value:
            dc = point[1] - self.psf.column1d[-1].value + 1
        if point[1] < self.psf.column1d[0].value:
            dc = point[1] - self.psf.column1d[0].value - 1
        point = (point[0] - dr, point[1] - dc, point[2], point[3])

        r1, c1 = self.psf.psf_row.value + row, self.psf.psf_column.value + col
        r = np.arange(0, shape[0]) + corner[0]
        c = np.arange(0, shape[1]) + corner[1]
        if (
            (row < (r[0] - 20))
            | (row > (r[-1] + 20))
            | (col < (c[0] - 20))
            | (col > (c[-1] + 20))
        ):
            return r, c, np.zeros(shape)

        f = self.psf.psf(point, freeze_dimensions=freeze_dimensions)
        s1 = np.asarray(
            [np.interp(r, r1, f[:, idx]) for idx in range(f.shape[1])]
        )
        s2 = np.asarray(
            [np.interp(c, c1, s1[:, idx]) for idx in range(s1.shape[1])]
        )
        s2 /= s2.sum()
        if return_locs:
            return r, c, s2
        return s2

    def get_FFIs(
        self,
        catalog,
        nframes: int,
        nreads: int,
        include_noise=True,
        include_cosmics=True,
        wavelength=None,
        temperature=None,
        freeze_dimensions=[2, 3],
        jitter=None,
        quiet=True,
    ):
        """Get Full Frame Images"""
        shape = self.shape
        catalog = catalog[
            (catalog.vis_column > 0)
            | (catalog.vis_column < shape[1])
            | (catalog.vis_row > 0)
            | (catalog.vis_row < shape[0])
        ].reset_index(drop=True)

        time = np.linspace(
            0,
            self.integration_time.value * nreads * nframes,
            nreads * nframes + 1,
        )[:-1]

        if jitter is not None:
            rowj, colj, thetaj = (  # noqa
                np.interp(time, jitter.time - jitter.time[0], jitter.rowj),
                np.interp(time, jitter.time - jitter.time[0], jitter.colj),
                np.interp(time, jitter.time - jitter.time[0], jitter.thetaj),
            )

        science_image = np.zeros(
            (
                nframes,
                *shape,
            ),
            dtype=float,
        )

        for idx, s in tqdm(
            catalog.iterrows(),
            total=catalog.shape[0],
            desc="Target",
            leave=True,
            position=0,
            disable=quiet,
        ):
            for tdx in range(nreads * nframes):
                if jitter is not None:
                    x, y = (
                        colj[tdx] + s.vis_column,  # - self.shape[1] // 2,
                        rowj[tdx] + s.vis_row,  # - self.shape[0] // 2,
                    )
                    y1, x1, prf = self.prf(
                        row=y,
                        col=x,
                        wavelength=wavelength,
                        temperature=temperature,
                        freeze_dimensions=freeze_dimensions,
                        corner=(y - 30, x - 30),
                        shape=(60, 61),
                        return_locs=True,
                    )
                else:
                    if tdx == 0:
                        x, y = (s.vis_column, s.vis_row)
                        y1, x1, prf = self.prf(
                            row=y,
                            col=x,
                            wavelength=wavelength,
                            temperature=temperature,
                            freeze_dimensions=freeze_dimensions,
                            corner=(y - 30, x - 30),
                            shape=(60, 61),
                            return_locs=True,
                        )

                Y, X = np.asarray(
                    np.meshgrid(
                        y1,
                        x1,
                        indexing="ij",
                    )
                ).astype(int)
                k = (
                    (X >= 0)
                    & (X < self.shape[1])
                    & (Y >= 0)
                    & (Y < self.shape[0])
                )
                science_image[tdx // nreads, Y[k], X[k]] += self.apply_gain(
                    u.Quantity(
                        np.random.poisson(
                            prf[k]
                            * s.vis_counts
                            * self.integration_time.to(u.second).value
                        ),
                        dtype=int,
                        unit=u.DN,
                    )
                ).value

        if include_noise:
            # # background light?
            for tdx in range(science_image.shape[0]):
                science_image[tdx] += self.get_background_light_estimate(
                    catalog.loc[0, "ra"], catalog.loc[0, "dec"], nreads * self.integration_time, shape
                ).value.astype(int)
                science_image[tdx] += np.random.normal(
                    loc=self.bias.value,
                    scale=self.read_noise.value,
                    size=shape,
                ).astype(int)
                science_image[tdx] += np.random.poisson(
                    lam=(self.dark * self.integration_time).value,
                    size=shape,
                ).astype(int)

        if include_cosmics:
            # This is the worst rate we expect, from the SAA
            cosmic_ray_rate = 1000 / (u.second * u.cm**2)
            cosmic_ray_expectation = (
                cosmic_ray_rate
                * ((self.pixel_size * 2048 * u.pix) ** 2).to(u.cm**2)
                * self.integration_time
            ).value

            for tdx in range(science_image.shape[0]):
                science_image[tdx] += get_simple_cosmic_ray_image(
                    cosmic_ray_expectation=cosmic_ray_expectation,
                    gain_function=self.apply_gain,
                    image_shape=self.shape,
                ).value

        time = np.asarray([time[idx::nreads] for idx in range(nreads)]).mean(
            axis=0
        )
        return time, science_image

    def get_subarray(
        self,
        cat,
        corner,
        nreads=50,
        nframes=10,
        freeze_dimensions=[2, 3],
        quiet=False,
        time_series_generators=None,
        include_noise=True
    ):
        f = np.zeros((nframes, *self.subarray_size), dtype=int)
        time = self.time[: nreads * nframes]
        time = np.asarray([time[idx::nreads] for idx in range(nreads)]).mean(
            axis=0
        )
        for idx, m in cat.iterrows():
            if time_series_generators is None:
                tsgenerator = lambda x: 1  # noqa
            else:
                tsgenerator = time_series_generators[idx]
            if tsgenerator is None:
                tsgenerator = lambda x: 1  # noqa

            prf = self.prf(row=m.vis_row, col=m.vis_column, corner=corner, freeze_dimensions=freeze_dimensions)
            for tdx in tqdm(
                range(nframes),
                desc="Times",
                position=0,
                leave=True,
                disable=quiet,
            ):
                f[tdx] += self.apply_gain(
                    u.Quantity(
                        np.random.poisson(
                            prf
                            * tsgenerator(time[tdx])
                            * m.vis_counts
                            * nreads
                            * self.integration_time.to(u.second).value
                        ),
                        dtype=int,
                        unit=u.DN,
                    )
                ).value
        if include_noise:
            for tdx in range(nframes):
                f[tdx] += (
                    self.get_background_light_estimate(
                        cat.loc[0, "ra"], cat.loc[0, "dec"], nreads*self.integration_time, self.subarray_size
                    )
                ).value.astype(int)
                f[tdx] += np.random.normal(
                    loc=self.bias.value,
                    scale=self.read_noise.value,
                    size=self.subarray_size,
                ).astype(int)
                f[tdx] += np.random.poisson(
                    lam=(self.dark * self.integration_time * nreads).value,
                    size=self.subarray_size,
                ).astype(int)

        apers = np.asarray(
            [
                self.get_aper(row=m.vis_row, col=m.vis_column, corner=corner, freeze_dimensions=freeze_dimensions)
                for idx, m in cat.iterrows()
            ]
        )
        return time, f, apers

    def plot_TPFs(self, nreads=50, include_noise=False, **kwargs):
        with plt.style.context(PANDORASTYLE):
            fig, ax = plt.subplots(
                1, len(self.Catalogs), figsize=(len(self.Catalogs) * 2, 2)
            )
            vmin = kwargs.pop("vmin", self.bias.value)
            vmax = kwargs.pop("vmax", self.bias.value + 10 * nreads)
            cmap = kwargs.pop("cmap", "viridis")

            for idx in range(len(self.Catalogs)):
                _, f, _ = self.get_subarray(
                    self.Catalogs[idx],
                    self.corners[idx],
                    nreads=nreads,
                    nframes=1,
                    quiet=True,
                    include_noise=include_noise,
                )
                _ = ax[idx].imshow(
                    f[0],
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    **kwargs,
                )
                ax[idx].set_title(f"BKG star {idx}")
                ax[idx].set(xticks=[], yticks=[])
            ax[0].set_title("Target")

        return fig

    def get_aper(self, row=0, col=0, corner=(0, 0), shape=None, freeze_dimensions=[2, 3]):
        if shape is None:
            shape = self.subarray_size
        aper = np.zeros(shape)
        for idx in np.arange(0, 1, 0.1):
            for jdx in np.arange(0, 1, 0.1):
                aper += self.prf(
                    row=row + idx, col=col + jdx, corner=corner, shape=shape, freeze_dimensions=freeze_dimensions,
                )
        aper /= 100
        return aper > 0.005

    def get_target_timeseries(self, ts_func=None, nreads=50, subarray=0, freeze_dimensions=[2, 3]):
        cat = self.Catalogs[subarray]
        corner = self.corners[subarray]
        time_series_generators = np.hstack(
            [ts_func, [None for idx in range(len(cat) - 1)]]
        )
        time, f, apers = self.get_subarray(
            cat,
            corner,
            nreads=nreads,
            nframes=len(self.time) // nreads,
            time_series_generators=time_series_generators,
            freeze_dimensions=freeze_dimensions,
            quiet=True,
        )
        return time, f[:, apers[0]].sum(axis=1)
