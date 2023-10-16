"""Deal with Pandora targets"""

# Standard library
from dataclasses import dataclass
from typing import Union
from urllib.request import URLError

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.coordinates import SkyCoord
from astropy.io import votable
from astropy.modeling import models
from astropy.time import Time
from astropy.utils.data import download_file
from astroquery.vizier import Vizier

from . import PANDORASTYLE, logger
from .utils import get_phoenix_model, get_planets, get_sky_catalog


@dataclass
class Target(object):
    name: str
    ra: u.Quantity
    dec: u.Quantity
    logg: u.Quantity
    teff: u.Quantity
    bmag: u.Quantity
    jmag: u.Quantity
    coord: SkyCoord = None

    def __post_init__(self):
        self.ra, self.dec = u.Quantity(self.ra, u.deg), u.Quantity(
            self.dec, u.deg
        )
        self.teff = u.Quantity(self.teff, u.K)
        self.logg = u.Quantity(self.logg)
        self._wavelength, self._spectrum = get_phoenix_model(
            teff=self.teff, logg=self.logg, jmag=self.jmag
        )
        try:
            self._get_SED()
        except URLError:
            logger.warning("Can not access internet to get SED")

        try:
            self.planets = get_planets(
                self.coord.apply_space_motion(Time(2000, format="jyear"))
            )
        except URLError:
            logger.warning("Can not access internet to get planet ephemerides")

    @staticmethod
    def from_gaia(coord: Union[str, SkyCoord]):
        name = None
        if isinstance(coord, str):
            name = coord
            coord = SkyCoord.from_name(coord)
        elif not isinstance(coord, SkyCoord):
            raise ValueError("`coord` must be a `SkyCoord` or a name string.")
        cat = get_sky_catalog(
            coord.ra, coord.dec, radius=5 * u.arcsecond, limit=1
        )
        if name is None:
            name = cat["source_id"][0]
        return Target(
            name=name,
            ra=cat["coords"][0].ra,
            dec=cat["coords"][0].dec,
            coord=cat["coords"][0],
            logg=cat["logg"][0],
            teff=cat["teff"][0],
            bmag=cat["bmag"][0],
            jmag=cat["jmag"][0],
        )

    @staticmethod
    def from_TIC(coord: Union[str, SkyCoord]):
        _KEY_DICT = {
            "Tmag": "bmag",
            "RAJ2000": "ra",
            "DEJ2000": "dec",
            "Teff": "teff",
            "logg": "logg",
            "Jmag": "jmag",
        }
        name = None
        if isinstance(coord, str):
            name = coord
            viz_dat = np.asarray(
                Vizier(
                    columns=[
                        *list(_KEY_DICT.keys()),
                        "Dist",
                        "pmRA",
                        "pmDE",
                        "TIC",
                    ]
                )
                .query_object(
                    coord, catalog="IV/39/tic82", radius=0.1 * u.arcsecond
                )[0][[*list(_KEY_DICT.keys()), "Dist", "pmRA", "pmDE", "TIC"]]
                .to_pandas()
                .iloc[0]
            )
        elif isinstance(coord, SkyCoord):
            viz_dat = np.asarray(
                Vizier(
                    columns=[
                        *list(_KEY_DICT.keys()),
                        "Dist",
                        "pmRA",
                        "pmDE",
                        "TIC",
                    ]
                )
                .query_region(
                    coord, catalog="IV/39/tic82", radius=1 * u.arcsecond
                )[0][[*list(_KEY_DICT.keys()), "Dist", "pmRA", "pmDE", "TIC"]]
                .to_pandas()
                .iloc[0]
            )
            name = f"TIC {int(viz_dat[-1])}"
        else:
            raise ValueError("`coord` must be a `SkyCoord` or a name string.")

        kwargs = {
            _KEY_DICT[k]: viz_dat[idx] for idx, k in enumerate(_KEY_DICT)
        }
        kwargs["coord"] = SkyCoord(
            kwargs["ra"] * u.deg,
            kwargs["dec"] * u.deg,
            distance=viz_dat[-4] * u.pc if np.isfinite(viz_dat[-4]) else 0,
            pm_ra_cosdec=viz_dat[-3] * u.mas / u.year
            if np.isfinite(viz_dat[-3])
            else 0,
            pm_dec=viz_dat[-2] * u.mas / u.year
            if np.isfinite(viz_dat[-2])
            else 0,
            obstime=Time(2000, format="jyear"),
        ).apply_space_motion(Time.now())
        # overwrite to current RA, Dec
        kwargs["ra"] = kwargs["coord"].ra.deg
        kwargs["dec"] = kwargs["coord"].dec.deg
        return Target(name=name, **kwargs)

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

    def _get_SED(self, radius=2):
        """Get the SED data for the target from Vizier

        Parameters
        ----------
        radius: float
            Radius to query in arcseconds
        """
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
        self.SED = {
            "wavelength": wavelength[k][s],
            "sed_flux": sed_flux[k][s],
            "sed_flux_err": sed_flux_err[k][s],
        }
        self.SED["model"] = self._fit_SED(wavelength[k][s], sed_flux[k][s])
        return self

    def __repr__(self):
        return f"{self.name} [{self.ra}, {self.dec}]"

    def _repr_html_(self):
        return f"{self.name} ({self.ra._repr_latex_()},  {self.dec._repr_latex_()})"

    def plot_spectrum(self, fig=None):
        wave_high = (
            np.linspace(
                self._wavelength.min().value * 0.9,
                self._wavelength.max().value * 1.5,
                1000,
            )
            * u.angstrom
        )
        with plt.style.context(PANDORASTYLE):
            if fig is None:
                fig = plt.figure()
            if hasattr(self, "SED"):
                plt.errorbar(
                    self.SED["wavelength"],
                    self.SED["sed_flux"],
                    self.SED["sed_flux_err"],
                    ls="",
                    marker=".",
                    c="r",
                    label="SED",
                )
            plt.plot(
                wave_high,
                self.spectrum(wave_high),
                ls="-",
                c="k",
                label="Phoenix Spectrum",
            )
            plt.xlabel(f"Wavelength {self._wavelength.unit._repr_latex_()}")
            plt.ylabel(f"Spectrum {self._spectrum.unit._repr_latex_()}")
            plt.yscale("log")
            plt.xscale("log")
            plt.title(f"{self.name} Spectrum")
            plt.legend()
        return fig

    def _fit_SED(self, wavelength, sed_flux):
        bb_data = (
            sed_flux.to(
                u.erg / (u.Hz * u.s * u.cm**2),
                equivalencies=u.spectral_density(wavelength),
            )
            / u.sr
        )
        if np.isfinite(self.teff.value):
            temperatures = np.linspace(
                self.teff.value - 300, self.teff.value + 300, 100
            )
        else:
            temperatures = np.linspace(1000, 10000, 100)
        chi2 = np.zeros_like(temperatures)
        for count in range(4):
            for idx, temperature in enumerate(temperatures):
                bb = models.BlackBody(temperature=temperature * u.K)
                bb_model = bb(wavelength)
                bb_model = bb_model * np.mean(bb_data / bb_model)
                chi2[idx] = np.nansum((bb_data - bb_model).value ** 2)
            t = temperatures[np.argmin(chi2)]
            temperatures = (temperatures - t) * 0.5 + t

        sed_bestfit_temperature = temperatures[np.argmin(chi2)] * u.K
        _blackbody = models.BlackBody(temperature=sed_bestfit_temperature)
        bb_model = (_blackbody(wavelength) * u.sr).to(
            sed_flux.unit,
            equivalencies=u.spectral_density(wavelength),
        )
        _corr = np.nanmean(sed_flux / bb_model)

        def func(wavelength):
            return (_blackbody(wavelength) * u.sr).to(
                sed_flux.unit,
                equivalencies=u.spectral_density(wavelength),
            ) * _corr

        return func

    def spectrum(self, wavelength: u.Quantity):
        """Returns the interpolated spectrum of the target based on Phoenix models."""
        return np.interp(
            wavelength,
            self._wavelength,
            self._spectrum,
            left=np.nan,
            right=np.nan,
        )
