"""Work with instrument filters"""

from dataclasses import dataclass
from glob import glob

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c, h
from astropy.io import votable
from astropy.modeling import models
from astropy.utils.data import download_file

from . import PACKAGEDIR

# Filters bundled with package
filter_fnames = np.sort(glob(f"{PACKAGEDIR}/data/*.xml"))
filter_dict = {
    "_".join([filter.split("/")[-1].split(".")[idx] for idx in [0, 2]]): filter
    for filter in filter_fnames
}
zeropoints = {
    "GAIA_G": 2.50007e-9 * u.erg / u.cm**2 / u.s / u.angstrom,
    "GAIA_Grp": 1.25e-9 * u.erg / u.cm**2 / u.s / u.angstrom,
    "GAIA_Gbp": 4.09861e-9 * u.erg / u.cm**2 / u.s / u.angstrom,
    "HST_G102": 6.23e-10 * u.erg / u.cm**2 / u.s / u.angstrom,
    "HST_G141": 2.11e-10 * u.erg / u.cm**2 / u.s / u.angstrom,
    "2MASS_J": 3.13e-10 * u.erg / u.cm**2 / u.s / u.angstrom,
    "2MASS_H": 1.13e-10 * u.erg / u.cm**2 / u.s / u.angstrom,
    "2MASS_Ks": 4.28e-11 * u.erg / u.cm**2 / u.s / u.angstrom,
    "Pandora_Visible": np.nan,
}


@dataclass
class Filter(object):
    name: str

    def __post_init__(self):
        if not (self.name in filter_dict.keys()):
            raise ValueError(
                f"No such filter as {self.name}. Choose from {', '.join(filter_dict.keys())}"
            )
        df = (
            votable.parse(filter_dict[self.name])
            .get_first_table()
            .to_table()
            .to_pandas()
        )
        self.wavelength, self._transmission = np.asarray(
            df.Wavelength
        ) * u.angstrom, np.asarray(df.Transmission)
        self.flux_zero_point = (
            zeropoints[self.name]
            if np.isfinite(zeropoints[self.name])
            else self._estimate_zeropoint()
        )

    def __repr__(self):
        return f"filter ({self.name})"

    @property
    def midpoint(self):
        return np.average(self.wavelength, weights=self._transmission)

    def _estimate_zeropoint(self):
        wavelength, spectrum = load_vega()
        transmission = np.interp(
            wavelength.value,
            self.wavelength.value,
            self._transmission,
        )
        zeropoint = np.trapz(
            wavelength * spectrum * transmission, wavelength
        ) / np.trapz(wavelength * transmission, wavelength)
        return zeropoint

    def mag_from_flux(self, flux):
        return -2.5 * np.log10(flux / self.flux_zero_point)

    def flux_from_mag(self, mag):
        return self.flux_zero_point * 10 ** (-mag / 2.5)

    def transmission(self, wavelength=None):
        if wavelength is None:
            return self._transmission
        if not hasattr(wavelength, "unit"):
            raise ValueError("Please pass a wavelength with an astropy unit")
        new_transmission = np.interp(
            wavelength, self.wavelength.to(wavelength.unit), self._transmission
        )
        return new_transmission

    def photons_per_second_per_area(self, wavelength, spectrum):
        flux_lambda = spectrum * self.transmission(wavelength)
        photonflux = np.trapz(flux_lambda * wavelength / (h * c), wavelength)
        return (photonflux).to(1 / u.second / u.m**2) * u.photon

    def get_rgb_color(self):
        k = (self.wavelength.to(u.nm).value > 380) & (
            self.wavelength.to(u.nm).value < 750
        )
        if k.any():
            return wavelength_to_rgb(
                np.average(self.wavelength[k], weights=self._transmission[k])
                .to(u.nm)
                .value
            )
        else:
            return np.asarray([0, 0, 0])


def wavelength_to_rgb(wavelength, gamma=0.8):

    """This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return np.asarray((int(R), int(G), int(B))) / 256


@dataclass
class Target(object):
    name: str
    radius: int = 3

    def __post_init__(self):
        url = f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={self.name.replace(' ', '%20')}&-c.rs={self.radius}"
        df = (
            votable.parse(download_file(url)).get_first_table().to_table()
        )  # .to_pandas()
        wavelength = (c / (np.asarray(df["sed_freq"]) * u.GHz)).to(u.angstrom)
        wavelength = (c / (np.asarray(df["sed_freq"]) * u.GHz)).to(u.angstrom)
        sed_flux = np.asarray(df["sed_flux"]) * u.jansky
        sed_flux = sed_flux.to(
            u.erg / u.cm**2 / u.s / u.angstrom,
            equivalencies=u.spectral_density(wavelength),
        )
        sed_flux_err = np.asarray(df["sed_eflux"]) * u.jansky
        sed_flux_err = sed_flux_err.to(
            u.erg / u.cm**2 / u.s / u.angstrom,
            equivalencies=u.spectral_density(wavelength),
        )
        k = np.isfinite(sed_flux_err)
        k &= sed_flux_err != 0
        k &= (df["sed_eflux"].data / df["sed_flux"]) < 0.1
        s = np.argsort(wavelength[k])
        self.wavelength, self._spectrum, self._spectrum_err = (
            wavelength[k][s],
            sed_flux[k][s],
            sed_flux_err[k][s],
        )
        self.ra, self.dec = df["_RAJ2000"].data.mean(), df["_DEJ2000"].data.mean()
        self.fit()

    def __repr__(self):
        return f"Target {self.name}"

    def plot_spectrum(self):
        wave_high = (
            np.linspace(
                self.wavelength.min().value * 0.9,
                self.wavelength.max().value * 1.5,
                1000,
            )
            * u.angstrom
        )
        bb_model = self.spectrum(wave_high)
        with plt.style.context("seaborn-white"):
            fig = plt.figure()
            plt.errorbar(
                self.wavelength,
                self._spectrum,
                self._spectrum_err,
                ls="",
                marker=".",
                c="k",
            )
            plt.xlabel(f"Wavelength {self.wavelength.unit._repr_latex_()}")
            plt.ylabel(f"Spectrum {self._spectrum.unit._repr_latex_()}")
            plt.yscale("log")
            plt.xscale("log")
            plt.title(f"{self.name} Spectrum")

            plt.plot(wave_high, bb_model, c="r")
            plt.yscale("log")
            plt.xscale("log")
        return fig

    def fit(self):
        bb_data = (
            self._spectrum.to(
                u.erg / (u.Hz * u.s * u.cm**2),
                equivalencies=u.spectral_density(self.wavelength),
            )
            / u.sr
        )
        # bb_error = (
        #     self._spectrum_err.to(
        #         u.erg / (u.Hz * u.s * u.cm**2),
        #         equivalencies=u.spectral_density(self.wavelength),
        #     )
        #     / u.sr
        # )

        temperatures = np.linspace(1000, 10000, 100)
        chi2 = np.zeros_like(temperatures)
        for count in range(4):
            for idx, temperature in enumerate(temperatures):
                bb = models.BlackBody(temperature=temperature * u.K)
                bb_model = bb(self.wavelength)
                bb_model = bb_model * np.mean(bb_data / bb_model)
                chi2[idx] = np.nansum((bb_data - bb_model).value ** 2)
            t = temperatures[np.argmin(chi2)]
            temperatures = (temperatures - t) * 0.5 + t

        self.bestfit_temperature = temperatures[np.argmin(chi2)] * u.K
        self._blackbody = models.BlackBody(temperature=self.bestfit_temperature)
        bb_model = (self._blackbody(self.wavelength) * u.sr).to(
            self._spectrum.unit, equivalencies=u.spectral_density(self.wavelength)
        )
        self._corr = np.nanmean(self._spectrum / bb_model)

    def spectrum(self, wavelength):
        return (self._blackbody(wavelength) * u.sr).to(
            self._spectrum.unit, equivalencies=u.spectral_density(wavelength)
        ) * self._corr


def load_vega():
    wavelength, spectrum = np.loadtxt(f"{PACKAGEDIR}/data/vega.dat").T
    wavelength *= u.angstrom
    spectrum *= u.erg / u.cm**2 / u.s / u.angstrom
    return wavelength, spectrum
