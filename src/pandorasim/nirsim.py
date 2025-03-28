"""Simulator for Visible Detector"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandorapsf as pp
from astropy.io import fits
from astropy.time import Time
from pandorasat import PANDORASTYLE, NIRDetector, get_logger
from pandorasat.phoenix import get_phoenix_model

from . import __version__
from .docstrings import add_docstring
from .sim import Sim
from .utils import FRAME_BIT_DICT, get_integrations, get_jitter

__all__ = ["NIRSim"]

logger = get_logger("pandora-sim")


class NIRSim(Sim):
    def __init__(
        self,
    ):
        """
        NIR Simulator for Pandora.
        """
        super().__init__(detector=NIRDetector())
        self.psf = self.psf.freeze_dimension(row=0 * u.pixel, column=0 * u.pixel)
        self.subarray_size = self.detector.subarray_size
        self.dark = self.detector.dark
        self.read_noise = self.detector.read_noise
        self.bias = self.detector.bias
        self.bias_uncertainty = self.detector.bias_uncertainty

        self.dark = 0.1 * u.electron / u.second
        self.read_noise = 8 * u.electron
        self.bias = 5000 * u.electron
        self.bias_uncertainty = 500 * u.electron

    @add_docstring("ra", "dec", "theta")
    def point(self, ra, dec, roll):
        """
        Point the simulation in a direction.
        """
        super().point(ra=ra, dec=dec, roll=roll)
        self._build_scene()

    @add_docstring("source_catalog")
    def from_source_catalog(self, source_catalog):
        """
        Use a source catalog to "point"
        """
        super().from_source_catalog(source_catalog=source_catalog)
        self._get_spectra(source_catalog)
        self._build_scene()

    def _build_scene(self):
        logger.start_spinner("Building trace scene object...")
        self.tracescene = pp.TraceScene(
            self.locations,
            psf=self.psf,
            shape=self.subarray_size,
            corner=(0, 0),
            wav_bin=1,
        )
        logger.stop_spinner()

    def _get_source_catalog(self):
        source_catalog = super()._get_source_catalog(gbpmagnitude_range=(-6, 21))
        source_catalog = self._get_spectra(source_catalog)
        return source_catalog

    def _get_spectra(self, source_catalog):
        logger.start_spinner("Interpolating PHOENIX spectra...")
        spectra = np.zeros((len(source_catalog), self.psf.trace_wavelength.shape[0]))
        for idx, teff, logg, j in zip(
            range(len(source_catalog)),
            np.nan_to_num(np.asarray(source_catalog.teff), 5777),
            np.nan_to_num(np.asarray(source_catalog.logg), 4.5),
            source_catalog.jmag,
        ):
            if teff < 2000:
                logger.warn(
                    f"{teff}K source in catalog is not modelable. Setting to 2000K."
                )
                teff = 2000
            if teff > 10000:
                logger.warn(
                    f"{teff}K source in catalog is not modelable. Setting to 10000K."
                )
                teff = 10000
            wav, spec = get_phoenix_model(teff=teff, logg=logg, jmag=j)
            spectra[idx, :] = self.psf.integrate_spectrum(wav, spec)
        logger.stop_spinner()

        # Units of electrons/s
        self.spectra = spectra * u.electron / u.s
        return source_catalog

    @property
    def background_rate(self):
        return 4 * u.electron / u.second

    @add_docstring(
        "SC_Resets1",
        "SC_Resets2",
        "SC_DropFrames1",
        "SC_DropFrames2",
        "SC_DropFrames3",
        "SC_ReadFrames",
        "SC_Groups",
        "SC_Integrations",
        "start_time",
        "noise",
        "jitter",
        "output_type",
    )
    def observe(
        self,
        SC_Resets1=1,
        SC_Resets2=1,
        SC_DropFrames1=0,
        SC_DropFrames2=16,
        SC_DropFrames3=0,
        SC_ReadFrames=4,
        SC_Groups=2,
        SC_Integrations=10,
        start_time=Time("2000-01-01T12:00:00", scale="utc"),
        target_spectrum_function=None,
        noise=True,
        jitter=True,
        output_type="fits",
    ):
        """
        Returns an observation as though taken from the Pandora Observatory.

        Parameters:
        -----------

        Returns:
        --------
        result : np.ndarray or astropy.io.fits.HDUList
            Result of the simulation. Either a numpy array with shape (nROIs, nframes, nrows, ncolumns), or a fits format.

        """

        integration_time = self.detector.frame_time()
        integration_info = get_integrations(
            SC_Resets1=SC_Resets1,
            SC_Resets2=SC_Resets2,
            SC_DropFrames1=SC_DropFrames1,
            SC_DropFrames2=SC_DropFrames2,
            SC_DropFrames3=SC_DropFrames3,
            SC_ReadFrames=SC_ReadFrames,
            SC_Groups=SC_Groups,
            SC_Integrations=SC_Integrations,
        )
        cadences = np.sum(
            [
                np.sum(
                    [(i == FRAME_BIT_DICT["read"]).all() for i in inte if len(i) > 0]
                )
                for inte in integration_info
            ]
        )
        integration_arrays = [np.hstack(idx) for idx in integration_info]

        nreads = np.sum([len(i) for i in integration_arrays])
        # one where a frame is being reset.
        resets = np.hstack(integration_arrays) != 1

        dt = np.arange(nreads) * integration_time.to(u.second).value
        time = start_time.jd + (dt * u.second).to(u.day).value

        source_flux = (
            self.spectra.T[:, :, None]
            * np.ones(nreads)[None, None, :]
            * integration_time
            * resets.astype(float)
        )

        if jitter:
            _, row, column, _ = get_jitter(
                rowstd=((1 * u.arcsecond) / self.detector.pixel_scale)
                .to(u.pixel)
                .value,
                colstd=((1 * u.arcsecond) / self.detector.pixel_scale)
                .to(u.pixel)
                .value,
                nframes=nreads,
                frame_time=integration_time.to(u.second).value,
            )
            delta_pos = np.asarray([row, column])
        else:
            delta_pos = None
        if target_spectrum_function is not None:
            source_flux[0, :, :] *= target_spectrum_function(time)

        # Data has shape (ntargets, nwavelength, ntimes), units of electrons
        data = self.tracescene.model(source_flux, delta_pos)
        data[data < 0] = 0

        if noise:
            # Apply poisson (shot) noise, data now has shape  (ntargets, nwavelength, ntimes), units of electrons
            data = np.random.poisson(data)

            # Apply background to every read, units of electrons
            data += np.random.poisson(
                (self.background_rate * integration_time.to(u.second)).value,
                size=data.shape,
            ).astype(int) * resets[:, None, None].astype(int)

            # Read Noise
            data += np.random.normal(
                loc=0,
                scale=self.read_noise.value,
                size=data.shape,
            ).astype(int)

            # Add poisson noise for the dark current to every frame, units of electrons
            data += np.random.poisson(
                lam=(self.dark * integration_time.to(u.second)).value,
                size=data.shape,
            ).astype(int)

        # Data in units of DN
        #    data = self.detector.apply_gain(u.Quantity(data.ravel(), unit='electron')).value.reshape(data.shape)

        # Crap gain for now because gain calculations are wicked broken
        data = (data.astype(float) * 0.5).astype(int)
        bias = self.bias.value * 0.5
        bias_std = self.bias_uncertainty.value * 0.5
        # Any pixels greater than uint16 are maxed out (saturated)
        data[(data + bias) > 2**16] = 2**16

        # Splits data into arrays representing each integration
        data_by_integration = np.array_split(
            data, np.cumsum(np.asarray([i.shape for i in integration_arrays])[:, 0])
        )[:-1]

        def get_group_masks(integration_info):
            """Function to create a boolean mask which is true for all the frames in each group."""
            masks = []
            for idx, i in enumerate(integration_info):
                if ((i == 2).all()) & (len(i) > 0):
                    masks.append(
                        (
                            np.hstack(
                                [
                                    np.zeros(len(i), bool)
                                    if idx != jdx
                                    else np.ones(len(i), bool)
                                    for jdx, i in enumerate(integration_info)
                                ]
                            )
                        )
                    )
            return np.asarray(masks)

        result = []
        for cdx, d, info in zip(range(cadences), data_by_integration, integration_info):
            # Cumulative sum for reading up the ramp
            d = np.cumsum(d, axis=0).astype(np.uint16)
            if noise:
                d += np.random.normal(loc=bias, scale=bias_std, size=d.shape).astype(
                    np.uint16
                )

            # Masks for each group
            group_masks = get_group_masks(info)
            # Output is the average in each group across the time dimension, after being cast in uint32 arrays.
            result.append(
                np.asarray(
                    [d[mask].astype(np.uint32).mean(axis=0) for mask in group_masks],
                    dtype=np.uint32,
                )
            )

        result = np.asarray(result, dtype=np.uint32)
        if output_type == "array":
            return result
        if output_type == "fits":
            return self._array_to_fits(
                result,
                start_time=start_time,
                SC_Resets1=SC_Resets1,
                SC_Resets2=SC_Resets2,
                SC_DropFrames1=SC_DropFrames1,
                SC_DropFrames2=SC_DropFrames2,
                SC_DropFrames3=SC_DropFrames3,
                SC_ReadFrames=SC_ReadFrames,
                SC_Groups=SC_Groups,
                SC_Integrations=SC_Integrations,
            )
        else:
            raise ValueError(f"Can not parse output type `{output_type}`")

    def show_subarray(self, **kwargs):
        """Plot an example subarray observation."""
        data = self.observe(
            SC_Integrations=1, jitter=False, noise=True, output_type="array"
        )[0, :, :, :]
        d = data.astype(int)[-1] - data.astype(int)[0]
        vmin, vmax = (
            kwargs.pop("vmin", np.percentile(d, 1)),
            kwargs.pop("vmax", -np.percentile(d, 1)),
        )
        cmap = kwargs.pop("cmap", "Greys")
        _, ax = plt.subplots(figsize=(3, 6))
        with plt.style.context(PANDORASTYLE):
            im = ax.imshow(
                d,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="lower",
                cmap=cmap,
            )
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Counts In (Last Group - First Group) [DN]")
        ax.set(
            title=f"{self.detector.name}\n(RA:{self.ra:.02f}, Dec:{self.dec:.02f}, Roll:{self.roll:.02f})",
            xlabel="Column [pixel]",
            ylabel="Row [pixel]",
        )
        return ax

    @property
    def primary_kwds(self):
        return {
            "EXTNAME": ("PRIMARY", "name of extension"),
            "NEXTEND": (1, "number of standard extensions"),
            "SIMDATA": (True, "simulated data"),
            "SCIDATA": (False, "science data"),
            "TELESCOP": ("NASA Pandora", "telescope"),
            "INSTRMNT": ("NIRDA", "instrument"),
            "CREATOR": ("Pandora DPC", "creator of this product"),
            "CRSOFTV": ("v" + str(__version__), "creator software version"),
            "TARG_RA": (self.ra.value, "target right ascension [deg]"),
            "TARG_DEC": (self.dec.value, "target declination [deg]"),
            "EXPDELAY": (-1, "exposure time delay [ms]"),
            "RICEX": (-1, "bit noise parameter for Rice compression"),
            "RICEY": (-1, "bit noise parameter for Rice compression"),
        }

    def _array_to_fits(
        self,
        data,
        start_time,
        SC_Resets1,
        SC_Resets2,
        SC_DropFrames1,
        SC_DropFrames2,
        SC_DropFrames3,
        SC_ReadFrames,
        SC_Groups,
        SC_Integrations,
    ):
        integration_time = self.detector.frame_time()

        primary_kwds = self.primary_kwds
        primary_kwds["SC_Resets1"] = (
            SC_Resets1,
            "Number of reset frames at the start of the first integration of exposure",
        )
        primary_kwds["SC_Resets2"] = (
            SC_Resets2,
            "Number of resent frames at the start of 1 through n integrations of exposure",
        )
        primary_kwds["SC_DropFrames1"] = (
            SC_DropFrames1,
            "Number of dropped frames after reset of any integration of exposure",
        )
        primary_kwds["SC_DropFrames2"] = (
            SC_DropFrames2,
            "Number of dropped frames in every group of integrations of exposure except the last group",
        )
        primary_kwds["SC_DropFrames3"] = (
            SC_DropFrames3,
            "Number of dropped frames in the last group of each integration of exposure",
        )
        primary_kwds["SC_ReadFrames"] = (
            SC_ReadFrames,
            "Number of frames read during each group of integration of exposure",
        )
        primary_kwds["SC_Groups"] = (
            SC_Groups,
            "Number of groups per integration of exposure",
        )
        primary_kwds["SC_Integrations"] = (
            SC_Integrations,
            "Number of integrations per exposure",
        )

        corstime = int(
            np.floor((start_time - Time("2000-01-01T12:00:00", scale="utc")).sec)
        )
        finetime = int(corstime % 1 * 10**9 // 1)

        primary_kwds["FRAMTIME"] = (
            self.detector.frame_time().to(u.second).value,
            "Frame time of each read",
        )
        primary_kwds["GRPTIME"] = (
            SC_ReadFrames * integration_time.to(u.second).value,
            "Exposure time of each group",
        )
        primary_kwds["CORSTIME"] = (
            corstime,
            "seconds since the TAI Epoch (12PM Jan 1, 2000)",
        )
        primary_kwds["FINETIME"] = (finetime, "nanoseconds added to CORSTIME seconds")
        primary_hdu = fits.PrimaryHDU()
        for key, value in primary_kwds.items():
            primary_hdu.header[key] = value

        image_hdu = fits.ImageHDU(data)
        image_kwds = {
            "NAXIS1": (
                data.shape[3],
                "pixel columns",
            ),  # might need to change these
            "NAXIS2": (data.shape[2], "pixel rows"),
            "NAXIS3": (data.shape[1], "number of groups"),
            "NAXIS4": (data.shape[0], "number of integrations"),
            "EXTNAME": ("SCIENCE", "extension name"),
            "TTYPE1": ("COUNTS", "data title: raw pixel counts"),
            "TFORM1": ("J", "data format: images of unsigned 32-bit integers"),
            "TUNIT1": ("count", "data units: count"),
        }

        for key, value in image_kwds.items():
            image_hdu.header[key] = value
        for card in self.wcs.to_header(relax=True).cards:
            image_hdu.header.append(card)

        hdulist = fits.HDUList([primary_hdu, image_hdu])

        return hdulist
