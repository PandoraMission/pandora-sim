"""Deal with Pandora targets"""

# Standard library
import os
import warnings
from dataclasses import dataclass

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.io import votable
from astropy.modeling import models
from astropy.utils.data import download_file
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astroquery.vizier import Vizier

from . import PACKAGEDIR

phoenixpath = f"{PACKAGEDIR}/data/phoenix"
os.environ["PYSYN_CDBS"] = phoenixpath

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Extinction files not found in ")
    # Third-party
    import pysynphot


@dataclass
class Target(object):
    name: str

    def __post_init__(self):
        key_dict = {
            "Tmag": "tmag",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Teff": "teff",
            "logg": "logg",
            "Jmag": "jmag",
        }
        viz_dat = np.asarray(
            Vizier(columns=list(key_dict.keys()))
            .query_object(
                self.name, catalog="IV/39/tic82", radius=0.1 * u.arcsecond
            )[0][list(key_dict.keys())]
            .to_pandas()
            .iloc[0]
        )
        [
            setattr(self, key_dict[k], viz_dat[idx])
            for idx, k in enumerate(key_dict)
        ]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            planets_tab = NasaExoplanetArchive.query_region(
                table="pscomppars",
                coordinates=SkyCoord(ra=self.ra * u.deg, dec=self.dec * u.deg),
                radius=5 * u.arcsecond,
            )
            if len(planets_tab) != 0:
                attrs = ["pl_orbper", "pl_tranmid", "pl_trandur", "pl_trandep"]
                self.planets = {
                    letter: {
                        attr: planets_tab[planets_tab["pl_letter"] == letter][
                            attr
                        ][0].unmasked
                        for attr in attrs
                    }
                    for letter in planets_tab["pl_letter"]
                }
                # There's an error in the NASA exoplanet archive units that makes duration "days" instead of "hours"
                for planet in self.planets:
                    self.planets[planet]["pl_trandur"] = (
                        self.planets[planet]["pl_trandur"].value * u.hour
                    )

    def box_transit(self, time):
        lc = np.zeros_like(time)
        for planet in self.planets:
            phase = (
                time - self.planets[planet]["pl_tranmid"].to(u.day).value
            ) % self.planets[planet]["pl_orbper"].to(u.day).value
            mask = (
                phase < self.planets[planet]["pl_trandur"].to(u.day).value / 2
            ) | (
                phase
                > (
                    self.planets[planet]["pl_orbper"].to(u.day).value
                    - self.planets[planet]["pl_trandur"].to(u.day).value / 2
                )
            )
            lc += -np.nan_to_num(
                mask.astype(float)
                * (self.planets[planet]["pl_trandep"].value / 100)
            )
        return lc + 1

    def from_vizier(self, radius=2):
        vizier_url = f"https://vizier.cds.unistra.fr/viz-bin/sed?-c={self.name.replace(' ', '%20')}&-c.rs={radius}"
        df = (
            votable.parse(download_file(vizier_url))
            .get_first_table()
            .to_table()
        )  # .to_pandas()
        df = df[df["sed_flux"] / df["sed_eflux"] > 5]
        if len(df) == 0:
            raise ValueError("No valid photometry")
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
        self.wavelength, self.spectrum, self.spectrum_err = (
            wavelength[k][s],
            sed_flux[k][s],
            sed_flux_err[k][s],
        )
        self.ra, self.dec = (
            df["_RAJ2000"].data.mean(),
            df["_DEJ2000"].data.mean(),
        )
        self.fit()
        self.model_type = "sed_fit"
        return self

    def from_phoenix(self, teff=None, logg=None, jmag=None):

        if teff is not None:
            self.teff = teff
        if logg is not None:
            self.logg = logg
        if jmag is not None:
            self.jmag = jmag

        star = pysynphot.Icat("phoenix", self.teff, 0.0, self.logg)
        star_norm = star.renorm(
            self.jmag, "vegamag", pysynphot.ObsBandpass("johnson,j")
        )
        star_norm.convert("Micron")
        star_norm.convert("flam")

        mask = (star_norm.wave >= 0.1) * (star_norm.wave <= 3)
        wavelength = star_norm.wave[mask] * u.micron
        wavelength = wavelength.to(u.angstrom)

        sed = star_norm.flux[mask] * u.erg / u.s / u.cm**2 / u.angstrom
        self.wavelength, self.spectrum = wavelength, sed
        self.model_type = "phoenix"
        return self

    def __repr__(self):
        return f"Target {self.name}"

    def plot_spectrum(self, fig=None):
        wave_high = (
            np.linspace(
                self.wavelength.min().value * 0.9,
                self.wavelength.max().value * 1.5,
                1000,
            )
            * u.angstrom
        )
        bb_model = self.model_spectrum(wave_high)
        with plt.style.context("seaborn-white"):
            if fig is None:
                fig = plt.figure()
            if hasattr(self, "spectrum_err"):
                plt.errorbar(
                    self.wavelength,
                    self.spectrum,
                    self.spectrum_err,
                    ls="",
                    marker=".",
                    c="k",
                )
            else:
                plt.plot(
                    self.wavelength,
                    self.spectrum,
                    ls="-",
                    c="k",
                )
            plt.xlabel(f"Wavelength {self.wavelength.unit._repr_latex_()}")
            plt.ylabel(f"Spectrum {self.spectrum.unit._repr_latex_()}")
            plt.yscale("log")
            plt.xscale("log")
            plt.title(f"{self.name} Spectrum")

            plt.plot(wave_high, bb_model, c="r")
            plt.yscale("log")
            plt.xscale("log")
        return fig

    def fit(self):
        bb_data = (
            self.spectrum.to(
                u.erg / (u.Hz * u.s * u.cm**2),
                equivalencies=u.spectral_density(self.wavelength),
            )
            / u.sr
        )
        # bb_error = (
        #     self.spectrum_err.to(
        #         u.erg / (u.Hz * u.s * u.cm**2),
        #         equivalencies=u.spectral_density(self.wavelength),
        #     )
        #     / u.sr
        # )
        if np.isfinite(self.teff):
            temperatures = np.linspace(self.teff - 300, self.teff + 300, 100)
        else:
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
        self._blackbody = models.BlackBody(
            temperature=self.bestfit_temperature
        )
        bb_model = (self._blackbody(self.wavelength) * u.sr).to(
            self.spectrum.unit,
            equivalencies=u.spectral_density(self.wavelength),
        )
        self._corr = np.nanmean(self.spectrum / bb_model)

    def model_spectrum(self, wavelength):
        if self.model_type == "phoenix":
            return np.interp(
                wavelength,
                self.wavelength,
                self.spectrum,
                left=np.nan,
                right=np.nan,
            )
        else:
            return (self._blackbody(wavelength) * u.sr).to(
                self.spectrum.unit,
                equivalencies=u.spectral_density(wavelength),
            ) * self._corr
