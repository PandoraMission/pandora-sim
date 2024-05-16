import pandorasat as ps
import pandorapsf as pp
import matplotlib.pyplot as plt
from pandorasat import PANDORASTYLE

from astropy.table import Table
from astropy.time import Time
from astropy.io import fits

# from pandorasim import PandoraSim
import astropy.units as u
import pandas as pd
from pandorasat import VisibleDetector
import numpy as np
from copy import deepcopy

from .utils import get_jitter
from . import __version__


class VisibleSim(object):
    """Holds methods for simulating data from the Visible Detector on Pandora."""

    def __init__(
        self,
        ra: u.Quantity,
        dec: u.Quantity,
        roll: u.Quantity,
        ROI_size=(50, 50),
        nROIs=9,
    ):
        """
        Visible Simulator for Pandora.

        Properties
        ----------
        ra: float
            Right Ascension of the pointing
        dec: float
            Declination of the pointing
        theta: float
            Roll angle of the pointing
        ROI_size: tuple
            Size in pixels of each region of interest
        nROIs: int
            Number of regions of interest.
        """
        self.ra, self.dec, self.roll, self.ROI_size, self.nROIs = (
            ra,
            dec,
            roll,
            ROI_size,
            nROIs,
        )
        self.detector = VisibleDetector()
        self.psf = pp.PSF.from_name("VISDA")
        self.wcs = self.detector.get_wcs(self.ra, self.dec)
        self.source_catalog = self._get_source_catalog()
        self.locations = np.asarray(
            [self.source_catalog.vis_row, self.source_catalog.vis_column]
        ).T
        self.ROI_corners = self.select_ROI_corners(self.nROIs)
        self.scene = None
        self.roiscene = None

    def world_to_pixel(self, ra, dec, distortion=True):
        """Helper function. This function ensures we keep the row-major convention in pandora-sim.

        Parameters
        ----------
        ra : float
            Right Ascension to be converted to pixel position.
        dec : float
            Declination to be converted to pixel position.
        distortion : bool
            Flag whether to account for the distortion in the WCS when converting from RA/Dec
            to pixel position. Default is True.

        Returns
        -------
        np.ndarray
            Row and column positions of each provided RA and Dec.
        """
        coords = np.vstack(
            [
                ra.to(u.deg).value if isinstance(ra, u.Quantity) else ra,
                dec.to(u.deg).value if isinstance(dec, u.Quantity) else dec,
            ]
        ).T
        if distortion:
            column, row = self.wcs.all_world2pix(coords, 0).T
        else:
            column, row = self.wcs.wcs_world2pix(coords, 0).T
        return np.vstack([row, column])

    def pixel_to_world(self, row, column, distortion=True):
        """Helper function. This function ensures we keep the row-major convention in pandora-sim.

        Parameters
        ----------
        row : float
            Pixel row position to be converted to sky coordinates.
        column : float
            Pixel column position to be converted to sky coordinates.
        distortion : bool
            Flag whether to account for the distortion in the WCS when converting from pixel position
            to sky coordinates. Default is True.

        Returns
        -------
        np.ndarray
            RA and Dec of input pixel positions.
        """
        coords = np.vstack(
            [
                column.to(u.pixel).value if isinstance(column, u.Quantity) else column,
                row.to(u.pixel).value if isinstance(row, u.Quantity) else row,
            ]
        ).T
        if distortion:
            return self.wcs.all_pix2world(coords, 0).T * u.deg
        else:
            return self.wcs.wcs_pix2world(coords, 0).T * u.deg

    def __repr__(self):
        return f"VisibleSim [({self.ra:.3f}, {self.dec:.3f})]"

    def _get_source_catalog(self, distortion: bool = True, **kwargs) -> pd.DataFrame:
        """Gets the source catalog of an input target

        Parameters
        ----------
        target_name : str
            Target name to obtain catalog for.
        distortion : bool
            Whether to apply a distortion to the WCS when mapping RA and Dec of catalog targets
            to detector pixels. Default is True.
        **kwargs
            Additional arguments passed to the ps.utils.get_sky_catalog function.

        Returns
        -------
        sky_catalog: pd.DataFrame
            Pandas dataframe of all the sources near the target
        """

        # This is fixed for visda, for now
        fieldstop_radius = (
            (self.detector.fieldstop_radius / self.detector.pixel_size)
            * self.detector.pixel_scale
        ).to(u.deg)
        radius = np.min(
            [
                (2 * fieldstop_radius.value**2) ** 0.5,
                (
                    2
                    * ((2048 * u.pix * self.detector.pixel_scale).to(u.deg).value / 2)
                    ** 2
                )
                ** 0.5,
            ]
        )

        # Get location and magnitude data
        cat = ps.utils.get_sky_catalog(
            self.ra, self.dec, radius=radius * u.deg, **kwargs
        )

        ra, dec, mag = cat["coords"].ra.deg, cat["coords"].dec.deg, cat["bmag"]
        vis_pix_coords = self.world_to_pixel(ra, dec, distortion=distortion)

        k = (
            np.abs(vis_pix_coords[0] - self.detector.shape[0] / 2)
            < self.detector.shape[0] / 2
        ) & (
            np.abs(vis_pix_coords[1] - self.detector.shape[1] / 2)
            < self.detector.shape[1] / 2
        )
        new_cat = deepcopy(cat)
        for key, item in new_cat.items():
            new_cat[key] = item[k]
        vis_pix_coords, ra, dec, mag = (
            vis_pix_coords[:, k],
            ra[k],
            dec[k],
            mag[k],
        )

        # we're assuming that Gaia B mag is very close to the Pandora visible magnitude
        vis_counts = np.zeros_like(mag)
        vis_flux = np.zeros_like(mag)
        wav = np.arange(100, 1000) * u.nm
        s = np.trapz(self.detector.sensitivity(wav), wav)
        for idx, m in enumerate(mag):
            f = self.detector.flux_from_mag(m)
            vis_flux[idx] = f.value
            # counts in electron/u.second
            vis_counts[idx] = (f * s).to(u.electron / u.second).value
        source_catalog = (
            pd.DataFrame(
                np.vstack(
                    [
                        ra,
                        dec,
                        mag,
                        *vis_pix_coords,
                        vis_counts,
                        vis_flux,
                        new_cat["jmag"],
                        new_cat["teff"].value,
                        new_cat["logg"],
                        new_cat["RUWE"],
                        new_cat["ang_sep"].value,
                    ]
                ).T,
                columns=[
                    "ra",
                    "dec",
                    "mag",
                    "vis_row",
                    "vis_column",
                    "vis_counts",
                    "vis_flux",
                    "jmag",
                    "teff",
                    "logg",
                    "ruwe",
                    "ang_sep",
                ],
            )
            .astype({"vis_counts": int})
            .drop_duplicates(["ra", "dec", "mag"])
            .reset_index(drop=True)
        )
        return source_catalog

    def select_ROI_corners(self, nROIs, magnitude_limit=14):
        """Selects the corners of ROIs.

        This is currently a placeholder. SOC will provide direction on how ROIs will be selected.
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
                raise ValueError(f"Can not select {nROIs} ROIs")
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

    @property
    def background_rate(self):
        return 4 * u.electron / u.second

    def get_FFI(
        self,
        nreads: int = 50,
    ):
        """Get a single frame of data as an FFI

        Parameters
        ----------
        nreads: int
            Number of reads to co-add together to make each frame. Default for Pandora VISDA is 50.
        """
        # This takes a fair amount of time and memory so let's only make it if we need it
        if self.scene is None:
            self.scene = pp.Scene(
                self.locations - 1024,
                self.psf,
                self.detector.shape,
                corner=(-1024, -1024),
            )

        shape = self.detector.shape
        int_time = self.detector.integration_time * nreads
        source_flux = (
            (np.asarray(self.source_catalog.vis_counts) * u.electron / u.second)
            * int_time
        ).value.astype(int)

        # FFI has shape (nrows, ncolumns), in units of electrons.
        ffi = self.scene.model(source_flux)

        # Apply poisson (shot) noise, ffi now has shape  (nrows, ncolumns), units of electrons
        ffi = np.random.poisson(self.scene.model(source_flux)[0])

        # Apply background to every read, units of electrons
        ffi += np.random.poisson(
            (self.background_rate * int_time).value, size=ffi.shape
        ).astype(int)

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
            lam=(self.detector.dark * int_time).value,
            size=ffi.shape,
        ).astype(int)

        # Apply gain
        #        ffi = self.detector.apply_gain(u.Quantity(ffi.ravel(), unit='electron')).value.reshape(ffi.shape)
        # Crap gain for now because gain calculations are wicked broken
        ffi *= 2

        # This is a bit hacky, but for FFIs we'll be ok. We do this because working with individual reads for FFI data is slow.
        ffi[ffi > (nreads * 2**16)] = nreads * 2**16
        return ffi

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
        """Returns an observation as though taken from the Pandora Observatory.

        Parameters
        ----------
        nreads: int
            Number of reads to co-add together to make each frame. Default for Pandora VISDA is 50.
        nframes : int
            Number of frames to return. Default is 100 frames.
        start_time: astropy.time.Time
            Time to start the observation. This is only used if returning a fits file, or supplying a flux function.
        target_flux_function : function or None.
            Function governing the generation of the time series from the subarray.
            Default is None.
        noise : bool
            Flag determining whether noise is simulated and included. Default is True.
        jitter : bool
            Flag determining whether jitter is simulated and included. Default is True.
        output_type: str
            String flag to determine the result output. Valid strings are "fits" or "array".
        bin_frames: int
            If `nreads` is high, many reads must be calculated per frame stored. To reduce this, set `bin_frames`.
            If `bin_frames=10`, only `nreads/10` reads will be calculated, each with an exposure time of
            `self.detector.integration_time * bin_frames`.

        Returns
        -------
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
                (self.source_catalog.vis_row > c[0])
                & (self.source_catalog.vis_row < (c[0] + self.ROI_size[0]))
                & (self.source_catalog.vis_column > c[1])
                & (self.source_catalog.vis_column < (c[1] + self.ROI_size[1]))
                for c in self.ROI_corners
            ],
            axis=0,
        )
        if self.roiscene is None:
            self.roiscene = pp.ROIScene(
                self.locations[k] - 1024,
                self.psf,
                self.detector.shape,
                corner=(-1024, -1024),
                ROI_size=self.ROI_size,
                ROI_corners=self.ROI_corners,
                nROIs=self.nROIs,
            )

        source_flux = (
            (np.asarray(self.source_catalog.vis_counts[k]) * u.electron / u.second)
            * integration_time.to(u.second)
        ).value.astype(int)[:, None] * np.ones(nr * nframes)[None, :]

        if jitter:
            _, row, column, theta = get_jitter(
                nframes=nframes * nr, frame_time=integration_time.to(u.second).value
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
                lam=(self.detector.dark * integration_time.to(u.second)).value,
                size=data.shape,
            ).astype(int)

        # Data in units of DN
        #    data = self.detector.apply_gain(u.Quantity(data.ravel(), unit='electron')).value.reshape(data.shape)

        # Crap gain for now because gain calculations are wicked broken
        data *= 2
        # Any pixels greater than uint16 are maxed out (saturated)
        data[data > 2**16] = 2**16

        # bin down the data across the read dimension
        data = data.reshape((self.nROIs, nframes, nr, 50, 50)).sum(axis=2)

        if output_type == "array":
            return data
        elif output_type == "fits":
            return self._roi_to_fits(
                data,
                nreads=nreads,
                start_time=start_time,
                integration_time=integration_time,
            )

    def show_ROI(self):
        """Plot an example of an ROI."""
        d = self.observe(
            nreads=50, nframes=1, output_type="array", jitter=False, noise=True
        )[:, 0, :, :]
        vmin, vmax = np.percentile(d, 1), np.percentile(d, 90)
        # Find the next largest perfect square from the number of subarrays given
        next_square = int(np.ceil(np.sqrt(self.nROIs)) ** 2)
        n = int(np.sqrt(next_square))

        _, ax = plt.subplots(n, n, figsize=(7, 6), sharex=True, sharey=True)
        plt.subplots_adjust(hspace=0.03, wspace=0.05)
        for idx in range(self.nROIs):
            l = np.where(np.arange(self.nROIs).reshape((n, n)) == idx)
            with plt.style.context(PANDORASTYLE):
                im = ax[l[0][0], l[1][0]].imshow(
                    d[idx],
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                    origin="lower",
                    cmap="Greys",
                )
                if l[1][0] == 0:
                    ax[l[0][0], l[1][0]].set(ylabel="Row\n[subarray pixel]")
                if l[0][0] == (n - 1):
                    ax[l[0][0], l[1][0]].set(xlabel="Column\n[subarray pixel]")
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

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 6))

        with plt.style.context(PANDORASTYLE):
            im = ax.imshow(
                ffi,
                vmin=np.percentile(ffi, 10),
                vmax=np.percentile(ffi, 99),
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
            nreads * integration_time.to(u.second).value,
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

        for card in self.wcs.to_header().cards:
            image_hdu.header.append(card)

        table_hdu = fits.table_to_hdu(Table(self.ROI_corners))
        for key, value in self.roi_kwds.items():
            table_hdu.header[key] = value

        hdulist = fits.HDUList([primary_hdu, image_hdu, table_hdu])
        return hdulist
