"""Simulator for Visible Detector"""

from copy import deepcopy

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandorapsf as pp
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from pandorasat import PANDORASTYLE, VisibleDetector, get_logger
import pandas as pd

from . import __version__
from .docstrings import add_docstring
from .sim import Sim
from .utils import get_jitter

__all__ = ["VisibleSim"]

logger = get_logger("pandora-sim")


class VisibleSim(Sim):
    @add_docstring("ROI_size", "nROIs")
    def __init__(self, ROI_size=(50, 50), nROIs=9, ROI_corners=None):
        """
        Visible Simulator for Pandora.
        """
        super().__init__(detector=VisibleDetector())
        self.ROI_size = ROI_size
        self.nROIs = nROIs

    @add_docstring("nROIs")
    def select_ROI_corners(self, nROIs, magnitude_limit=14):
        """Selects the corners of ROIs.

        This is currently a placeholder. SOC will provide direction on how ROIs will be selected.

        Parameters:
        -----------
        magnitude_limit : float
            Visual magnitude limit down to which ROI targets will be considered.
        """
        source_mag = deepcopy(np.asarray(self.source_catalog.mag))
        locations = deepcopy(self.locations)

        # This downweights sources far from the middle
        r = 1 - np.hypot(
            *(self.locations - np.asarray(self.detector.shape) / 2).T
        ) / np.hypot(*np.asarray(self.detector.shape))
        source_mag += -2.5 * np.log10(r)

        k = (source_mag < magnitude_limit) & (self.source_catalog.ruwe <= 1.2)
        locations, source_mag = locations[k], source_mag[k]
        size = np.asarray(self.ROI_size)
        crpix = self.wcs.wcs.crpix
        corners = [(-size[0] // 2 + crpix[0], -size[1] // 2 + crpix[1])]
        while len(corners) < nROIs:
            if len(locations) == 0:
                return corners
            idx = np.argmin(source_mag)
            corner = np.round(locations[idx]).astype(int) - size // 2
            if ~np.any(
                [
                    (np.abs(c[0] - corner[0]) < size[0] // 2)
                    & (np.abs(c[1] - corner[1]) < size[1] // 2)
                    for c in corners
                ]
            ):
                corners.append(tuple(c for c in corner))

            k = ~np.in1d(np.arange(len(locations)), idx)
            locations = locations[k]
            source_mag = source_mag[k]
        return corners

    @add_docstring("ra", "dec", "theta")
    def point(self, ra, dec, roll):
        """
        Point the simulation in a direction.
        """
        super().point(ra=ra, dec=dec, roll=roll)
        self.ROI_corners = None
        self._build_scene()

    @add_docstring("source_catalog")
    def from_source_catalog(self, source_catalog: pd.DataFrame):
        """
        Create a simulation based on a catalog
        """
        super().from_source_catalog(
            source_catalog=self._calculate_counts(source_catalog)
        )
        r, c = self.source_catalog.row.values, self.source_catalog.column.values
        s = np.argsort(np.hypot(r - 1024, c - 1024))
        corners = np.vstack(
            [
                r[s][: self.nROIs] - self.ROI_size[0] // 2,
                c[s][: self.nROIs] - self.ROI_size[1] // 2,
            ]
        ).T.astype(int)
        self.ROI_corners = [tuple(corner) for corner in corners]
        self._build_scene()

    def _build_scene(self):
        # logger.start_spinner("Building scene object...")
        self.scene = pp.Scene(
            self.locations - np.asarray(self.detector.shape) / 2,
            self.psf,
            self.detector.shape,
            corner=(-self.detector.shape[0] // 2, -self.detector.shape[1] // 2),
        )
        # logger.stop_spinner()
        if self.ROI_corners is None:
            self.ROI_corners = self.select_ROI_corners(self.nROIs)
            self.nROIs = len(self.ROI_corners)

        # logger.start_spinner("Building ROI scene object...")

        k = np.any(
            [
                (self.source_catalog.row > c[0])
                & (self.source_catalog.row < (c[0] + self.ROI_size[0]))
                & (self.source_catalog.column > c[1])
                & (self.source_catalog.column < (c[1] + self.ROI_size[1]))
                for c in self.ROI_corners
            ],
            axis=0,
        )

        self.roiscene = pp.ROIScene(
            self.locations[k] - np.asarray(self.detector.shape) / 2,
            self.psf,
            self.detector.shape,
            corner=(
                -np.asarray(self.detector.shape[0]) // 2,
                -np.asarray(self.detector.shape[1]) // 2,
            ),
            ROI_size=self.ROI_size,
            ROI_corners=[
                (
                    int(r[0] - self.detector.shape[0] // 2),
                    int(r[1] - self.detector.shape[1] // 2),
                )
                for r in self.ROI_corners
            ],
            nROIs=self.nROIs,
        )
        # logger.stop_spinner()

    def _get_source_catalog(self):
        source_catalog = super()._get_source_catalog(gbpmagnitude_range=(-6, 18))
        return self._calculate_counts(source_catalog)

    def _calculate_counts(self, source_catalog):
        # we're assuming that Gaia B mag is very close to the Pandora visible magnitude
        vis_counts = np.zeros(len(source_catalog))
        vis_flux = np.zeros(len(source_catalog))
        # wav = np.arange(100, 1000) * u.nm
        # s = np.trapz(self.detector.sensitivity(wav), wav)
        for idx, m in enumerate(np.asarray(source_catalog.mag)):
            f = self.detector.mag_to_flux(m)
            vis_flux[idx] = f.value
            # counts in electron/u.second
            vis_counts[idx] = (f).to(u.electron / u.second).value

        source_catalog["counts"] = vis_counts
        source_catalog["flux"] = vis_flux
        return source_catalog

    @property
    def background_rate(self):
        return 4 * u.electron / u.second

    @add_docstring("nreads", "noise")
    def get_FFI(
        self,
        nreads: int = 50,
        noise=True,
    ):
        """Get a single frame of data as an FFI

        Returns:
        --------
        data : np.ndarray
            Returns a single FFI as a numpy array with dtype uint32.
        """
        int_time = self.detector.integration_time * nreads
        source_flux = (
            (np.asarray(self.source_catalog.counts) * u.electron / u.second) * int_time
        ).value.astype(int)

        # FFI has shape (nrows, ncolumns), in units of electrons.
        ffi = self.scene.model(source_flux)

        # Apply poisson (shot) noise, ffi now has shape  (nrows, ncolumns), units of electrons
        ffi = np.random.poisson(self.scene.model(source_flux)[0])
        if hasattr(self.detector, "fieldstop"):
            ffi *= self.detector.fieldstop.astype(int)

        if noise:
            # Apply background to every read, units of electrons
            bkg = np.random.poisson(
                (self.background_rate * int_time).value, size=ffi.shape
            ).astype(int)
            if hasattr(self.detector, "fieldstop"):
                bkg *= self.detector.fieldstop.astype(int)
            ffi += bkg

            # # Apply a bias to every read which is a Gaussian with mean = bias * nreads value and std = (nreads * (read noise)**2)**0.5
            # We actually do this as a sum because otherwise the integer math doesn't work out...!?

            test_distribution = (
                np.random.normal(
                    loc=self.detector.bias.value,
                    scale=self.detector.read_noise.value,
                    size=(nreads, 10000),
                )
                .astype(int)
                .sum(axis=0)
            )
            ffi += np.random.normal(
                loc=test_distribution.mean(),
                scale=self.detector.read_noise.value * np.sqrt(nreads),
                size=(ffi.shape),
            ).astype(int)

            # Add poisson noise for the dark current to every frame, units of electrons
            ffi += np.random.poisson(
                lam=(self.detector.dark_rate * int_time).value,
                size=ffi.shape,
            ).astype(int)

        # Apply gain
        #        ffi = self.detector.apply_gain(u.Quantity(ffi.ravel(), unit='electron')).value.reshape(ffi.shape)
        # Crap gain for now because gain calculations are wicked broken
        ffi *= 2

        # This is a bit hacky, but for FFIs we'll be ok. We do this because working with individual reads for FFI data is slow.
        ffi[ffi > (nreads * 2**16)] = nreads * 2**16
        return ffi

    @add_docstring(
        "nreads",
        "nframes",
        "start_time",
        "target_flux_function",
        "noise",
        "jitter",
        "output_type",
        "bin_frames",
    )
    def observe(
        self,
        nreads=50,
        nframes=100,
        start_time=Time("2000-01-01T12:00:00", scale="utc"),
        target_flux_function=None,
        noise=True,
        jitter=True,
        output_type="fits",
        bin_frames=10,
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
        if not nreads / bin_frames == nreads // bin_frames:
            raise ValueError(
                "`bin_frames` must be a factor of `nreads` (e.g. `nreads=50` can be binned by a factor of 2, 5, 10... etc)"
            )

        nr = nreads // bin_frames
        integration_time = self.detector.integration_time * bin_frames
        dt = np.arange(nr * nframes) * integration_time.to(u.second).value
        time = start_time.jd + (dt * u.second).to(u.day).value

        k = np.any(
            [
                (self.source_catalog.row > c[0])
                & (self.source_catalog.row < (c[0] + self.ROI_size[0]))
                & (self.source_catalog.column > c[1])
                & (self.source_catalog.column < (c[1] + self.ROI_size[1]))
                for c in self.ROI_corners
            ],
            axis=0,
        )

        source_flux = (
            (np.asarray(self.source_catalog.counts[k]) * u.electron / u.second)
            * integration_time.to(u.second)
        ).value.astype(int)[:, None] * np.ones(nr * nframes)[None, :]

        if jitter:
            _, row, column, _ = get_jitter(
                rowstd=((1 * u.arcsecond) / self.detector.pixel_scale)
                .to(u.pixel)
                .value,
                colstd=((1 * u.arcsecond) / self.detector.pixel_scale)
                .to(u.pixel)
                .value,
                nframes=nframes * nr,
                frame_time=integration_time.to(u.second).value,
            )
            delta_pos = np.asarray([row, column])
        else:
            delta_pos = None
        if target_flux_function is not None:
            source_flux[0, :] *= target_flux_function(time)

        # Data has shape (nROIs, ntimes, nrows, ncolumns), units of electrons
        data = self.roiscene.model(source_flux, delta_pos)
        data[data < 0] = 0

        if noise:
            # Apply poisson (shot) noise, data now has shape  (nROIs, nreads, nrows, ncolumns), units of electrons
            data = np.random.poisson(data)

            # Apply background to every read, units of electrons
            data += np.random.poisson(
                (self.background_rate * integration_time.to(u.second)).value,
                size=data.shape,
            ).astype(int)

            # # Apply a bias to every read which is a Gaussian with mean = bias value and std = read noise
            # We have to estimate the mean bias when summed across reads because the integer math messes with the mean
            test_distribution = (
                np.random.normal(
                    loc=self.detector.bias.value,
                    scale=self.detector.read_noise.value,
                    size=(nreads, 10000),
                )
                .astype(int)
                .sum(axis=0)
            )

            data += np.random.normal(
                loc=test_distribution.mean() / nr,
                scale=self.detector.read_noise.value * np.sqrt(bin_frames),
                size=data.shape,
            ).astype(int)

            # Add poisson noise for the dark current to every frame, units of electrons
            data += np.random.poisson(
                lam=(self.detector.dark_rate * integration_time.to(u.second)).value,
                size=data.shape,
            ).astype(int)

        # Data in units of DN
        #    data = self.detector.apply_gain(u.Quantity(data.ravel(), unit='electron')).value.reshape(data.shape)

        # Crap gain for now because gain calculations are wicked broken
        data *= 2
        # Any pixels greater than uint16 are maxed out (saturated)
        data[data > 2**16] = 2**16

        # bin down the data across the read dimension
        data = data.reshape((self.nROIs, nframes, nr, *self.ROI_size)).sum(axis=2)

        if output_type == "array":
            return data
        elif output_type == "fits":
            return self._roi_to_fits(
                data,
                nreads=nreads,
                start_time=start_time,
            )

    def show_ROI(self):
        """Plot an example of an ROI."""
        d = self.observe(
            nreads=50, nframes=1, output_type="array", jitter=False, noise=True
        )[:, 0, :, :]
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 99)
        # Find the next largest perfect square from the number of subarrays given
        next_square = int(np.ceil(np.sqrt(self.nROIs)) ** 2)
        n = int(np.sqrt(next_square))

        _, ax = plt.subplots(n, n, figsize=(7, 6), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.03, wspace=0.05)
        for idx in range(self.nROIs):
            l = ((idx - (idx % n)) // n, idx % n)
            with plt.style.context(PANDORASTYLE):
                im = ax[l[0], l[1]].imshow(
                    d[idx],
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    origin="lower",
                    cmap="Greys",
                )
                if l[1] == 0:
                    ax[l[0], l[1]].set(ylabel="Row\n[subarray pixel]")
                if l[0] == (n - 1):
                    ax[l[0], l[1]].set(xlabel="Column\n[subarray pixel]")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(
            f"Counts after {(self.detector.integration_time * 50).to_string(format='latex')} [DN]"
        )
        plt.suptitle(
            f"{self.detector.name}\n(RA:{self.ra:.02f}, Dec:{self.dec:.02f}, Roll:{self.roll:.02f})"
        )
        return ax

    def show_FFI(self, ax=None):
        """Plot an example of an FFI."""
        ffi = self.get_FFI(nreads=50)
        d = self.observe(
            nreads=50, nframes=1, output_type="array", jitter=False, noise=True
        )[:, 0, :, :]
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 99)

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        with plt.style.context(PANDORASTYLE):
            im = ax.imshow(
                ffi,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                origin="lower",
                cmap="Greys",
            )
            for corner in self.ROI_corners:
                r, c = (
                    np.asarray([0, 0, self.ROI_size[0], self.ROI_size[0], 0])
                    + corner[0],
                    np.asarray([0, self.ROI_size[1], self.ROI_size[1], 0, 0])
                    + corner[1],
                )
                ax.plot(c, r, c="r")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(
                f"Counts after {(self.detector.integration_time * 50).to_string(format='latex')} [DN]"
            )
            ax.set(
                xlabel="Column [pixel]",
                ylabel="Row [pixel]",
                title=f"{self.detector.name}\n(RA:{self.ra:.02f}, Dec:{self.dec:.02f}, Roll:{self.roll:.02f})",
            )
        return ax

    @property
    def primary_kwds(self):
        return {
            "EXTNAME": ("PRIMARY", "name of extension"),
            "NEXTEND": (2, "number of standard extensions"),
            "SIMDATA": (True, "simulated data"),
            "SCIDATA": (False, "science data"),
            "TELESCOP": ("NASA Pandora", "telescope"),
            "INSTRMNT": ("VISDA", "instrument"),
            "CREATOR": ("Pandora DPC", "creator of this product"),
            "CRSOFTV": ("v" + str(__version__), "creator software version"),
            "TARG_RA": (self.ra.value, "target right ascension [deg]"),
            "TARG_DEC": (self.dec.value, "target declination [deg]"),
            "EXPDELAY": (-1, "exposure time delay [ms]"),
            "RICEX": (-1, "bit noise parameter for Rice compression"),
            "RICEY": (-1, "bit noise parameter for Rice compression"),
        }

    @property
    def roi_kwds(self):
        return {
            "NAXIS": (2, "number of array dimensions"),
            "NAXIS1": (
                len(self.ROI_corners[0]),
                "length of dimension 1",
            ),
            "NAXIS2": (len(self.ROI_corners), "length of dimension 2"),
            "PCOUNT": (0, "number of group parameters"),
            "GCOUNT": (1, "number of groups"),
            "TFIELDS": (2, "number of table fields"),
            "TTYPE1": ("Column", "table field 1 type"),
            "TFORM1": ("I21", "table field 1 format"),
            "TUNIT1": ("pix", "table field 1 unit"),
            "TBCOL1": (1, ""),
            "TTYPE2": ("Row", "table field 2 type"),
            "TFORM2": ("I21", "table field 2 format"),
            "TUNIT2": ("pix", "table field 2 unit"),
            "TBCOL2": (22, ""),
            "EXTNAME": ("ROITABLE", "name of extension"),
            "NROI": (
                len(self.ROI_corners),
                "number of regions of interest",
            ),
            "ROISTRTX": (
                -1,
                "region of interest origin position in column",
            ),
            "ROISTRTY": (-1, "region of interest origin position in row"),
            "ROISIZEX": (-1, "region of interest size in column"),
            "ROISIZEY": (-1, "region of interest size in row"),
        }

    def _roi_to_fits(self, data, start_time, nreads=50, integration_time=None):
        if integration_time is None:
            integration_time = self.detector.integration_time
        n_arrs, nframes, nrows, ncols = data.shape

        corstime = int(
            np.floor((start_time - Time("2000-01-01T12:00:00", scale="utc")).sec)
        )
        finetime = int(corstime % 1 * 10**9 // 1)

        primary_kwds = self.primary_kwds
        primary_kwds["FRMSREQD"] = (nframes, "number of frames requested")
        primary_kwds["FRMSCLCT"] = (nframes, "number of frames collected")
        primary_kwds["NUMCOAD"] = (nreads, "number of frames coadded")
        primary_kwds["FRMTIME"] = (
            (nreads * integration_time).to(u.second).value,
            "time in each frame [s]",
        )
        primary_kwds["CORSTIME"] = (
            corstime,
            "seconds since the TAI Epoch (12PM Jan 1, 2000)",
        )
        primary_kwds["FINETIME"] = (finetime, "nanoseconds added to CORSTIME seconds")
        primary_hdu = fits.PrimaryHDU()
        for key, value in primary_kwds.items():
            primary_hdu.header[key] = value

        # Find the next largest perfect square from the number of subarrays given
        next_square = int(np.ceil(np.sqrt(n_arrs)) ** 2)
        sq_sides = int(np.sqrt(next_square))

        padding = np.zeros((next_square - n_arrs, nframes, nrows, ncols), dtype=int)
        subarrays = np.append(data, padding, axis=0)
        subarrays = subarrays.reshape((sq_sides, sq_sides, *data.shape[1:])).transpose(
            [2, 0, 1, 3, 4]
        )
        image_data = np.block(
            [[subarrays[:, i, j] for j in range(sq_sides)] for i in range(sq_sides)]
        ).astype(data.dtype)
        image_kwds = {
            "NAXIS": (3, "number of array dimensions"),
            "NAXIS1": (
                image_data.shape[1],
                "first axis size",
            ),  # might need to change these
            "NAXIS2": (image_data.shape[2], "second axis size"),
            "NAXIS3": (image_data.shape[0], "third axis size"),
            "EXTNAME": ("SCIENCE", "extension name"),
            "TTYPE1": ("COUNTS", "data title: raw pixel counts"),
            "TFORM1": ("J", "data format: images of unsigned 32-bit integers"),
            "TUNIT1": ("count", "data units: count"),
        }
        image_hdu = fits.ImageHDU(image_data)
        for key, value in image_kwds.items():
            image_hdu.header[key] = value

        for card in self.wcs.to_header(relax=True).cards:
            image_hdu.header.append(card)

        table_hdu = fits.table_to_hdu(Table(self.ROI_corners))
        for key, value in self.roi_kwds.items():
            table_hdu.header[key] = value

        hdulist = fits.HDUList([primary_hdu, image_hdu, table_hdu])
        return hdulist
