"""Holds metadata and methods on Pandora VISDA"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import votable, fits
from astropy.wcs import WCS
from astroquery.mast import Catalogs

from . import PACKAGEDIR
from .utils import get_sky_catalog
from .detector import Detector
from .psf import interpfunc
import warnings


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
            np.interp(wavelength, wav, transmission, left=0, right=0) * u.DN / u.photon
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
        return 0.15 * u.deg

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.714

    def diagnose(self, n=3, image_type='PSF', wavelength=0.54 * u.micron, temperature=-10 * u.deg_C):
        if not (n % 2) == 1:
            n += 1
        fig, ax = plt.subplots(n, n, figsize=(n*2, n*2))
        for x1, y1 in np.asarray(((np.mgrid[:n, :n] - n//2) * (600/(n//2)))).reshape((2, n**2)).T:
            jdx = int(x1//(600/(n//2)) + n//2)
            idx = int(-y1//(600/(n//2)) + n//2)
            point = (x1, y1, 0.5, 10)
            if image_type.lower() == 'psf':
                x, y, f = self.psf.psf_x.value, self.psf.psf_y.value, self.psf.psf(point)
                ax[idx, jdx].set(xticklabels=[], yticklabels=[])
            elif image_type.lower() == 'prf':
                x, y, f = self.psf.prf(point)
                if idx < (n - 1):
                    ax[idx, jdx].set(xticklabels=[])
                if jdx >= 1:
                    ax[idx, jdx].set(yticklabels=[])
    #             if jdx >= 1:
    #                 ax[idx, jdx].set(yticklabels=[])
            else: 
                raise ValueError("No such image type. Choose from `'PSF'`, or `'PRF'.`")
            ax[idx, jdx].pcolormesh(x, y, f.T, vmin=0, vmax=[0.1 if image_type.lower() == 'prf' else 0.005][0])
        ax[n//2, 0].set(ylabel='Y Pixel')
        ax[n - 1, n//2].set(xlabel='X Pixel')    
        ax[0, n//2].set(title=image_type.upper())
        return fig

    def wcs(self, target_ra, target_dec):
        # This is where we'd build or use a WCS.
        # Here we're assuming no distortions, no rotations.
        hdu = fits.PrimaryHDU()
        hdu.header['CTYPE1'] = 'RA---TAN'
        hdu.header['CTYPE2'] = 'DEC--TAN'
        hdu.header['CRVAL1'] = target_ra
        hdu.header['CRVAL2'] = target_dec
        hdu.header['CRPIX1'] = self.naxis1.value/2# + 0.5
        hdu.header['CRPIX2'] = self.naxis2.value/2# - 0.5
        hdu.header['NAXIS1'] = self.naxis1.value
        hdu.header['NAXIS2'] = self.naxis2.value
        hdu.header['CDELT1'] = -self.pixel_scale.to(u.deg/u.pixel).value
        hdu.header['CDELT2'] = self.pixel_scale.to(u.deg/u.pixel).value
        ## We're not doing any rotation and scaling right now... but those go in PC1_1, PC1_2, PC1_2, PC2_2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wcs = WCS(hdu.header)
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
            for idx, col in enumerate(np.arange(0, 1, 1/res)):
                for jdx, row in enumerate(np.arange(0, 1, 1/res)):
                    x1, y1, ar = self.psf.prf(prf_point, location=(col, row))
                    k = np.asarray(np.meshgrid(np.in1d(ys, y1), np.in1d(xs, x1))).all(axis=0)
                    grid[idx, jdx, k] = ar.ravel()
            return grid

        # Grid of Pixel Locations
        xs = np.arange(int(np.floor(self.psf.psf_x.min().value)) - 1, int(np.ceil(self.psf.psf_x.max().value)) + 2, 1)
        ys = np.arange(int(np.floor(self.psf.psf_y.min().value)) - 1, int(np.ceil(self.psf.psf_y.max().value)) + 2, 1)

        # Positions on detector
        x = np.linspace(int(np.ceil(self.psf.x1d.min().value)), int(np.floor(self.psf.x1d.max().value)), res)
        y = np.linspace(int(np.ceil(self.psf.y1d.min().value)), int(np.floor(self.psf.y1d.max().value)), res)
        
        # Sub Pixels
        xp, yp = np.arange(0, 1, 1/sub_res), np.arange(0, 1, 1/sub_res)
        
        # Build a grid of evaluated PRFs at each location
        grid = np.zeros((sub_res, sub_res, res, res, xs.shape[0], ys.shape[0]))
        for idx, col in enumerate(x):
            for jdx, row in enumerate(y):
                prf_point = (col, row, wavelength, temperature)
                grid[:, :, idx, jdx, :] = get_grid(prf_point, sub_res)

        grid = grid.transpose([4, 5, 2, 3, 0, 1])
        # Function to get PRF at any given location
        # Will interpolate across the detector, but will return the closest match in subpixel space
        def fastPRF(xloc:float, yloc:float):
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
            return (xs + (xloc - (xloc % 1)),
                ys + (yloc - (yloc % 1)),
                interpfunc(yloc, y, interpfunc(xloc, x, grid[:, :, :, :, np.argmin(np.abs(xp - (xloc % 1))), np.argmin(np.abs(yp - (yloc % 1)))])))
        return fastPRF


    #     X

    # def get_sky_catalog(self, target_ra, target_dec, magnitude_range=(-3, 16)):
    #     """Gets the source catalog of an input target
        
    #     Parameters
    #     ----------
    #     target_name : str
    #         Target name to obtain catalog for.

    #     Returns
    #     -------
    #     sky_catalog: pd.DataFrame   
    #         Pandas dataframe of all the sources near the target
    #     """

    #     # This is fixed for visda, for now
    #     radius = 0.155 #degrees
    #     # catalog_data = Catalogs.query_object(target_name, radius=radius, catalog="TIC")
    #     # target_ra, target_dec = catalog_data[0][['ra', 'dec']].values()

    #     # Get location and magnitude data
    #     ra, dec, mag = np.asarray(get_sky_catalog(target_ra, target_dec, radius=radius, magnitude_range=magnitude_range)).T
    #     k = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
    #     ra, dec, mag = ra[k], dec[k], mag[k]
    #     pix_coords = self.wcs(target_ra, target_dec).all_world2pix(np.vstack([ra, dec]).T, 1)
        
    # # we're assuming that Gaia B mag is very close to the Pandora visible magnitude
    #     counts = np.zeros_like(mag)
    #     flux = np.zeros_like(mag)
    #     wav = np.arange(100, 1000) * u.nm
    #     s = np.trapz(self.sensitivity(wav), wav)
    #     for idx, m in enumerate(mag):
    #         f = self.flux_from_mag(m)
    #         flux[idx] = f.value
    #         counts[idx] = (f * s).to(u.electron/u.second).value
    #     source_catalog = pd.DataFrame(np.vstack([ra, dec, mag, *pix_coords.T, counts, flux]).T, columns=['ra', 'dec', 'mag', 'x', 'y', 'counts', 'flux'])
    #     return source_catalog

    
    # def get_sky_image(self, source_catalog, wavelength=0.54*u.micron, temperature=10*u.deg_C, nreads=1):
    #     science_image = np.zeros(
    #         (self.naxis1.value.astype(int), self.naxis2.value.astype(int))
    #     )
    #     for idx, s in source_catalog.iterrows():
    #         x, y = s.x - self.naxis1.value//2, s.y - self.naxis2.value//2
    #         if ((x < -650) | (x > 650) | (y < -650) | (y > 650)):
    #             continue
    #         x1, y1, f = self.psf.prf(
    #             (y * u.pix, x * u.pix, wavelength, temperature)
    #         )
    #         X, Y = np.asarray(np.meshgrid(x1 + self.naxis1.value//2, y1 + self.naxis2.value//2)).astype(int)
    #         science_image[Y, X] += f.T * s.counts

        
    #     science_image *= u.electron/u.second
    #     # # background light?
    #     science_image += self.get_background_light_estimate(source_catalog.loc[0, 'ra'], source_catalog.loc[0, 'dec'])

    #     # time integrate
    #     science_image *= self.integration_time * nreads

    #     # fieldstop
    #     f = np.hypot(*(np.mgrid[:self.naxis1.value.astype(int), :self.naxis2.value.astype(int)] - np.hstack([self.naxis1.value.astype(int), self.naxis2.value.astype(int)])[:, None, None]/2))
    #     f = (f < ((0.15*u.deg)/self.pixel_scale).to(u.pix).value).astype(float)
    #     science_image *= f

    #     # noise
    #     noise = np.random.normal(loc=self.bias.value, scale=self.read_noise.value * np.sqrt(nreads),
    #                              size=(self.naxis1.value.astype(int), self.naxis2.value.astype(int))) * u.electron
    #     noise += np.random.poisson(lam=(self.dark * self.integration_time * nreads).value,
    #                                size=(self.naxis1.value.astype(int), self.naxis2.value.astype(int))) * u.electron

    #     science_image += noise
    #     return science_image

    def get_background_light_estimate(self, ra, dec):
        """Placeholder, will estimate the background light at different locations?
        Background in one integration...!
        """

        bkg = u.Quantity(np.zeros(self.shape, int), unit='electron', dtype='int')
        bkg[self.fieldstop] = u.Quantity(np.random.poisson(lam=2,
                                   size=self.fieldstop.sum()).astype(int), unit='electron', dtype='int')
        return bkg



    #   def diagnose_psf(self, wavelength=0.54 * u.micron, temperature=-10 * u.deg_C):
        # """Returns some diagnostic plots that show the PSF and PRF to check that the rotations look how we expect."""
        # with plt.style.context("seaborn-white"):
        #     """Returns a few of diagnostic plots to show whether all the
        #     rotations we are doing are in line with expectations..."""
        #     fig1, ax = plt.subplots(3, 3, figsize=(12, 10))
        #     ax[0, 0].imshow(
        #         self.psf.psf((-600 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[0, 0].set(xticks=[], yticks=[], title=[-600, -600])
        #     ax[0, 1].imshow(
        #         self.psf.psf((-600 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[0, 1].set(
        #         xticks=[],
        #         yticks=[],
        #         title=f"PSF at {wavelength.to_string(format='latex')},"
        #         + f" {temperature.to_string(format='latex')} (Model)\n{[0, 600]}",
        #     )
        #     ax[0, 2].imshow(
        #         self.psf.psf((-600 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[0, 2].set(xticks=[], yticks=[], title=[600, 600])
        #     ax[1, 0].imshow(
        #         self.psf.psf((0 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[1, 0].set(xticks=[], yticks=[], title=[-600, 0])
        #     ax[1, 1].imshow(
        #         self.psf.psf((0 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[1, 1].set(xticks=[], yticks=[], title=[0, 0])
        #     ax[1, 2].imshow(
        #         self.psf.psf((0 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[1, 2].set(xticks=[], yticks=[], title=[600, 0])
        #     ax[2, 0].imshow(
        #         self.psf.psf((600 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[2, 0].set(xticks=[], yticks=[], title=[-600, -600])
        #     ax[2, 1].imshow(
        #         self.psf.psf((600 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[2, 1].set(xticks=[], yticks=[], title=[-600, -600])
        #     im = ax[2, 2].imshow(
        #         self.psf.psf((600 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.01,
        #         origin="lower",
        #     )
        #     ax[2, 2].set(xticks=[], yticks=[], title=[-600, -600])
        #     cbar = plt.colorbar(mappable=im, ax=ax)
        #     cbar.set_label("Normalized Brightness")

        #     fig2, ax = plt.subplots(3, 3, figsize=(12, 10))
        #     ax[0, 0].pcolormesh(
        #         *self.psf.prf((600 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[0, 1].pcolormesh(
        #         *self.psf.prf((600 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[0, 2].pcolormesh(
        #         *self.psf.prf((600 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[1, 0].pcolormesh(
        #         *self.psf.prf((0 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[1, 1].pcolormesh(
        #         *self.psf.prf((0 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[1, 2].pcolormesh(
        #         *self.psf.prf((0 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[2, 0].pcolormesh(
        #         *self.psf.prf((-600 * u.pix, -600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[2, 1].pcolormesh(
        #         *self.psf.prf((-600 * u.pix, 0 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     im = ax[2, 2].pcolormesh(
        #         *self.psf.prf((-600 * u.pix, 600 * u.pix, wavelength, temperature)),
        #         vmin=0,
        #         vmax=0.1,
        #     )
        #     ax[0, 1].set(
        #         title=f"PRF at {wavelength.to_string(format='latex')}"
        #         + f"{temperature.to_string(format='latex')} \n(i.e. PSF On Pixel Scale)"
        #     )
        #     cbar = plt.colorbar(mappable=im, ax=ax)
        #     cbar.set_label("Normalized Brightness")

        #     ar = np.zeros(
        #         (self.naxis1.value.astype(int), self.naxis2.value.astype(int))
        #     )
        #     for y in np.linspace(-600, 600, 3):
        #         for x in np.linspace(-600, 600, 3):
        #             y1, x1, f = self.psf.prf(
        #                 (x * u.pix, y * u.pix, wavelength, temperature)
        #             )
        #             l = np.asarray(
        #                 [i.ravel() for i in np.meshgrid(x1, y1)]
        #             ) + np.asarray([self.naxis1.value / 2, self.naxis2.value / 2])[
        #                 :, None
        #             ].astype(
        #                 int
        #             )
        #             k = (
        #                 (l[0] > 0)
        #                 & (l[0] < self.naxis1.value.astype(int))
        #                 & (l[1] > 0)
        #                 & (l[1] < self.naxis2.value.astype(int))
        #             )
        #             ar[l[1][k], l[0][k]] += f.ravel()[k]
        #     fig3, ax = plt.subplots(figsize=(21, 20))
        #     im = ax.imshow(ar, vmin=0, vmax=0.01, origin="lower")
        #     ax.set(
        #         xlabel="X Pixel",
        #         ylabel="Y Pixel",
        #         title=f"PRF On Detector at {wavelength.to_string(format='latex')}"
        #         + f", {temperature.to_string(format='latex')}",
        #     )
        #     cbar = plt.colorbar(mappable=im, ax=ax)
        #     cbar.set_label("Normalized Brightness")
        # return fig1, fig2, fig3
