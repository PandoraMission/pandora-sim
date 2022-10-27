"""Holds metadata and methods on Pandora VISDA"""
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import votable

from . import PACKAGEDIR
from .detector import Detector


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
        return 2 * u.electron / u.second

    @property
    def read_noise(self):
        return 2.1 * u.electron

    @property
    def bias(self):
        return 100 * u.electron

    @property
    def integration_time(self):
        return 0.2 * u.second

    def throughput(self, wavelength):
        return wavelength.value**0 * 0.816

    def diagnose_psf(self, wavelength=0.54 * u.micron, temperature=-10 * u.deg_C):
        """Returns some diagnostic plots that show the PSF and PRF to check that the rotations look how we expect."""
        with plt.style.context("seaborn-white"):
            """Returns a few of diagnostic plots to show whether all the
            rotations we are doing are in line with expectations..."""
            fig1, ax = plt.subplots(3, 3, figsize=(12, 10))
            ax[0, 0].imshow(
                self.psf.psf((-600 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[0, 0].set(xticks=[], yticks=[], title=[-600, 600])
            ax[0, 1].imshow(
                self.psf.psf((0 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[0, 1].set(
                xticks=[],
                yticks=[],
                title=f"PSF at {wavelength.to_string(format='latex')},"
                + " {temperature.to_string(format='latex')} (Model)\n{[0, 600]}",
            )
            ax[0, 2].imshow(
                self.psf.psf((600 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[0, 2].set(xticks=[], yticks=[], title=[600, 600])
            ax[1, 0].imshow(
                self.psf.psf((-600 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[1, 0].set(xticks=[], yticks=[], title=[-600, 0])
            ax[1, 1].imshow(
                self.psf.psf((0 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[1, 1].set(xticks=[], yticks=[], title=[0, 0])
            ax[1, 2].imshow(
                self.psf.psf((600 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[1, 2].set(xticks=[], yticks=[], title=[600, 0])
            ax[2, 0].imshow(
                self.psf.psf((-600 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[2, 0].set(xticks=[], yticks=[], title=[-600, -600])
            ax[2, 1].imshow(
                self.psf.psf((0 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[2, 1].set(xticks=[], yticks=[], title=[-600, -600])
            im = ax[2, 2].imshow(
                self.psf.psf((600 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.1,
                origin="lower",
            )
            ax[2, 2].set(xticks=[], yticks=[], title=[-600, -600])
            cbar = plt.colorbar(mappable=im, ax=ax)
            cbar.set_label("Normalized Brightness")

            fig2, ax = plt.subplots(3, 3, figsize=(12, 10))
            ax[0, 0].pcolormesh(
                *self.psf.prf((-600 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[0, 1].pcolormesh(
                *self.psf.prf((0 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[0, 2].pcolormesh(
                *self.psf.prf((600 * u.pix, 600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[1, 0].pcolormesh(
                *self.psf.prf((-600 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[1, 1].pcolormesh(
                *self.psf.prf((0 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[1, 2].pcolormesh(
                *self.psf.prf((600 * u.pix, 0 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[2, 0].pcolormesh(
                *self.psf.prf((-600 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[2, 1].pcolormesh(
                *self.psf.prf((0 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            im = ax[2, 2].pcolormesh(
                *self.psf.prf((600 * u.pix, -600 * u.pix, wavelength, temperature)),
                vmin=0,
                vmax=0.01,
            )
            ax[0, 1].set(
                title=f"PRF at {wavelength.to_string(format='latex')}"
                + f"{temperature.to_string(format='latex')} \n(i.e. PSF On Pixel Scale)"
            )
            cbar = plt.colorbar(mappable=im, ax=ax)
            cbar.set_label("Normalized Brightness")

            ar = np.zeros(
                (self.naxis1.value.astype(int), self.naxis2.value.astype(int))
            )
            for y in np.linspace(-600, 600, 3):
                for x in np.linspace(-600, 600, 3):
                    x1, y1, f = self.psf.prf(
                        (x * u.pix, y * u.pix, wavelength, temperature)
                    )
                    l = np.asarray(
                        [i.ravel() for i in np.meshgrid(x1, y1)]
                    ) + np.asarray([self.naxis1.value / 2, self.naxis2.value / 2])[
                        :, None
                    ].astype(
                        int
                    )
                    k = (
                        (l[0] > 0)
                        & (l[0] < self.naxis1.value.astype(int))
                        & (l[1] > 0)
                        & (l[1] < self.naxis2.value.astype(int))
                    )
                    ar[l[1][k], l[0][k]] += f.ravel()[k]
            fig3, ax = plt.subplots(figsize=(11, 10))
            im = ax.imshow(ar, vmin=0, vmax=0.01, origin="lower")
            ax.set(
                xlabel="X Pixel",
                ylabel="Y Pixel",
                title=f"PRF On Detector at {wavelength.to_string(format='latex')}"
                + ", {temperature.to_string(format='latex')}",
            )
            cbar = plt.colorbar(mappable=im, ax=ax)
            cbar.set_label("Normalized Brightness")
        return fig1, fig2, fig3
