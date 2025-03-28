# Standard library
import warnings
from copy import copy
from typing import Dict, Optional

# Third-party
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery import log as asqlog
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

asqlog.setLevel("ERROR")
# Third-party

FRAME_BIT_DICT = {"reset": 1, "read": 2, "drop": 4}


def get_simple_cosmic_ray_image(
    cosmic_ray_expectation=0.4,
    average_cosmic_ray_flux: u.Quantity = u.Quantity(1e3, unit="DN"),
    cosmic_ray_distance: u.Quantity = u.Quantity(0.01, unit=u.pixel / u.DN),
    image_shape=(2048, 2048),
):
    """Function to get a simple cosmic ray image

    This function has no basis in physics at all.
    The true rate of cosmic rays, the energy deposited, sampling distributions.
    All this function can do is put down fairly reasonable "tracks" that mimic the impact of cosmic rays, with some tuneable parameters to change the properties.

    """
    ncosmics = np.random.poisson(cosmic_ray_expectation)
    im = np.zeros(image_shape, dtype=int)

    for ray in np.arange(ncosmics):
        # Random flux drawn from some exponential...
        cosmic_ray_counts = (
            np.random.exponential(average_cosmic_ray_flux.value)
            * average_cosmic_ray_flux.unit
        )

        # Random location
        xloc = np.random.uniform(0, image_shape[0])
        yloc = np.random.uniform(0, image_shape[1])
        # Random angle into the detector
        theta = np.random.uniform(
            -0.5 * np.pi, 0.5 * np.pi
        )  # radians from the top of the sensor?
        # Random angle around
        phi = np.random.uniform(0, 2 * np.pi)

        r = np.sin(theta) * (cosmic_ray_distance * cosmic_ray_counts).value

        x1, x2, y1, y2 = (
            xloc,
            xloc + (r * np.cos(phi)),
            yloc,
            yloc + (r * np.sin(phi)),
        )
        m = (y2 - y1) / (x2 - x1)
        c = y1 - (m * x1)

        xs, ys = np.sort([x1, x2]).astype(int), np.sort([y1, y2]).astype(int)
        xs, ys = (
            [xs[0], xs[1] if np.diff(xs) > 0 else xs[1] + 1],
            [
                ys[0],
                ys[1] if np.diff(ys) > 0 else ys[1] + 1,
            ],
        )

        coords = np.vstack(
            [
                np.round(np.arange(*xs, 0.005)).astype(int),
                np.round(m * np.arange(*xs, 0.005) + c).astype(int),
            ]
        ).T
        coords = coords[(coords[:, 1] >= ys[0]) & (coords[:, 1] <= ys[1])]
        if len(coords) == 0:
            continue
        fper_element = cosmic_ray_counts / len(coords)
        coords = coords[
            (
                (coords[:, 0] >= 0)
                & (coords[:, 0] < image_shape[0])
                & (coords[:, 1] >= 0)
                & (coords[:, 1] < image_shape[1])
            )
        ]
        coords, wcoords = np.unique(coords, return_counts=True, axis=0)
        im[coords[:, 0], coords[:, 1]] = np.random.poisson(
            (wcoords * fper_element).value
        )
    return u.Quantity(im, dtype=int, unit=u.DN)


def get_planets(coord: SkyCoord, radius: u.Quantity = 20 * u.arcsecond) -> dict:
    """Largish default radius for high proper motion targets this breaks
    Returns a dictionary of dictionaries with planet parameters.
    """
    try:
        coord2000 = coord.apply_space_motion(Time(2016, format="jyear"))
    except ValueError:
        coord2000 = coord
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        planets_tab = NasaExoplanetArchive.query_region(
            table="pscomppars", coordinates=coord2000, radius=radius
        )
        if len(planets_tab) != 0:
            attrs = ["pl_orbper", "pl_tranmid", "pl_trandur", "pl_trandep"]
            planets = {
                letter: {
                    attr: planets_tab[planets_tab["pl_letter"] == letter][attr][
                        0
                    ].unmasked
                    for attr in attrs
                }
                for letter in planets_tab["pl_letter"]
            }
            # There's an error in the NASA exoplanet archive units that makes duration "days" instead of "hours"
            for planet in planets:
                planets[planet]["pl_trandur"] = (
                    planets[planet]["pl_trandur"].value * u.hour
                )
        else:
            planets = {}
    return planets


def get_jitter(
    rowstd: float = 1,
    colstd: float = 0.3,
    thetastd: float = 0.0005,
    correlation_time: float = 1 * u.second,
    nframes: int = 50,
    frame_time: float = 0.2 * u.second,
    seed=None,
):
    """Returns some random, time correlated jitter time-series.

    Parameters:
    ----------
    rowstd: float
        Standard deviation of jitter in pixels in row/y axis
    colstd: float
        Standard deviation of jitter in pixels in col/x axis
    thetastd: float
        Standard deviation of jitter in degrees in y axis
    correlation_time: float
        The timescale over which data is correlated in seconds.
        Increase this value for smoother time-series
    nframes: int
        Number of frames to generate
    frame_time: float
        The time spacing for each frame
    seed: Optional, int
        Optional seed for random walk

    Returns:
    --------
    time : np.ndarray
        Time array in seconds
    row: np.ndarray
        Jitter in the row/y axis in pixels
    col: np.ndarray
        Jitter in the column/x axis in pixels
    theta: np.ndarray
        Jitter in angle in degrees
    """
    time = np.arange(nframes) * frame_time  # noqa:F811
    tstd = (correlation_time / frame_time).value

    def jitter_func(std):
        f = np.random.normal(0, std, size=nframes)
        return convolve(f, Gaussian1DKernel(tstd)) * tstd**0.5

    jitter = []
    for idx, std, unit in zip(
        [0, 1, 2], [rowstd, colstd, thetastd], [u.pixel, u.pixel, u.deg]
    ):
        if seed is not None:
            np.random.seed(seed + idx)
        jitter.append(jitter_func(std) * unit)

    return time, *jitter


def get_integrations(
    SC_Resets1,
    SC_Resets2,
    SC_DropFrames1,
    SC_DropFrames2,
    SC_DropFrames3,
    SC_ReadFrames,
    SC_Groups,
    SC_Integrations,
):
    """
    Function to generate the integration frames as a list of arrays containing flags denoting
    the status of each frame. The frames are grouped by status and ordered chronologically.

    Parameters
    ----------
    SC_Resets1 : int
        Number of reset frames at the start of the first integration of exposure
    SC_Resets2 : int
        Number of resent frames at the start of 1 through n integrations of exposure
    SC_DropFrames1 : int
        Number of dropped frames after reset of any integration of exposure
    SC_DropFrames2 : int
        Number of dropped frames in every group of integrations of exposure except the last group
    SC_DropFrames3 : int
        Number of dropped frames in the last group of each integration of exposure
    SC_ReadFrames : int
        Number of frames read during each group of integration of exposure
    SC_Groups : int
        Number of groups per integration of exposure
    SC_Integrations : int
        Number of integrations per exposure

    Returns
    -------
    integrations : list
        List of arrays containing each frame in the integrations grouped in chronological order
        with each integer element representing that frame's status.
    """
    cintn = [
        np.zeros(SC_Resets2, int) + FRAME_BIT_DICT["reset"],
        *[
            np.zeros(SC_ReadFrames, int) + FRAME_BIT_DICT["read"],
            np.zeros(SC_DropFrames2, int) + FRAME_BIT_DICT["drop"],
        ]
        * (SC_Groups - 1),
        np.zeros(SC_ReadFrames, int) + FRAME_BIT_DICT["read"],
    ]
    cint1 = copy(cintn)
    cint1.pop(0)

    cint1.insert(0, np.zeros(SC_Resets1, int) + FRAME_BIT_DICT["reset"])
    cint1.insert(1, np.zeros(SC_DropFrames1, int) + FRAME_BIT_DICT["drop"])

    integrations = [cint1]
    for idx in np.arange(SC_Integrations - 1):
        integrations.append(copy(cintn))

    integrations[-1].append(np.zeros(SC_DropFrames3, int) + FRAME_BIT_DICT["drop"])
    return integrations


def get_plot_vectors(inte):
    """
    Helper function to provide plotting information for each frame to the `plot_integrations`
    function.

    Parameters
    ----------
    inte : array
        The array of integer flags representing each frame in a single integration of exposure

    Returns
    -------
    vectors : list of arrays
        List of arrays specifying the plotting parameters for each individual frame
    """
    s = np.where(inte | FRAME_BIT_DICT["reset"] == FRAME_BIT_DICT["reset"])[0]
    if len(s) > 0:
        s = s[-1]
        y = np.hstack(
            [
                np.zeros(s + 1, dtype=int),
                np.arange(s + 1, len(inte), dtype=int) - s + 1,
            ]
        )
    else:
        y = np.arange(0, len(inte), dtype=int)

    cols = np.vstack(
        [
            inte | FRAME_BIT_DICT["reset"] == FRAME_BIT_DICT["reset"],
            inte | FRAME_BIT_DICT["read"] == FRAME_BIT_DICT["read"],
            inte | FRAME_BIT_DICT["drop"] == FRAME_BIT_DICT["drop"],
        ]
    ).T
    return np.hstack([y[:, None], cols])


def plot_nirda_integrations(
    SC_Resets1,
    SC_Resets2,
    SC_DropFrames1,
    SC_DropFrames2,
    SC_DropFrames3,
    SC_ReadFrames,
    SC_Groups,
    SC_Integrations,
):
    """
    Function to plot the integrations of the NIRDA visually given values describing
    the integration strategy.

    Parameters
    ----------
    SC_Resets1 : int
        Number of reset frames at the start of the first integration of exposure
    SC_Resets2 : int
        Number of resent frames at the start of 1 through n integrations of exposure
    SC_DropFrames1 : int
        Number of dropped frames after reset of any integration of exposure
    SC_DropFrames2 : int
        Number of dropped frames in every group of integrations of exposure except the last group
    SC_DropFrames3 : int
        Number of dropped frames in the last group of each integration of exposure
    SC_ReadFrames : int
        Number of frames read during each group of integration of exposure
    SC_Groups : int
        Number of groups per integration of exposure
    SC_Integrations : int
        Number of integrations per exposure
    """
    integrations = get_integrations(
        SC_Resets1,
        SC_Resets2,
        SC_DropFrames1,
        SC_DropFrames2,
        SC_DropFrames3,
        SC_ReadFrames,
        SC_Groups,
        SC_Integrations,
    )
    cadences = [
        np.sum([(i == FRAME_BIT_DICT["read"]).all() for i in inte if len(i) > 0])
        for inte in integrations
    ]
    integrations = [np.hstack(idx) for idx in integrations]
    dat = np.vstack([get_plot_vectors(inte) for inte in integrations])
    with plt.style.context("seaborn-v0_8-white"):
        fig, ax = plt.subplots(figsize=(np.max([5, len(dat) // 25]), 2))
        ax.plot(dat[:, 0], ls="--", c="grey")
        ax.scatter(
            np.arange(len(dat))[dat[:, 1].astype(bool)],
            dat[:, 0][dat[:, 1].astype(bool)],
            c="k",
            label="Reset",
        )
        ax.scatter(
            np.arange(len(dat))[dat[:, 2].astype(bool)],
            dat[:, 0][dat[:, 2].astype(bool)],
            c="lightblue",
            label="Frames",
        )
        ax.scatter(
            np.arange(len(dat))[dat[:, 3].astype(bool)],
            dat[:, 0][dat[:, 3].astype(bool)],
            c="r",
            label="Dropped Frames",
            marker="x",
        )
        ax.set(
            xlabel="Time",
            ylabel="Counts",
            xticks=[],
            yticks=[],
            title=(
                f"{SC_Integrations} Integrations, "
                + f"{SC_Groups} Groups\n{SC_ReadFrames} Read/group, "
                + f"{SC_DropFrames2} Drop/group\n{len(dat)} total reads, "
                + f"{np.sum(cadences)} total cadences"
            ),
        )
        ax.legend(frameon=True, bbox_to_anchor=(1.1, 1.05))
    return fig


def save_to_FITS(
    data: np.ndarray,
    filename: str,
    primary_kwds: Dict,
    image_kwds: Dict,
    roitable: bool = False,
    roitable_kwds: Optional[Dict] = None,
    roi_data: Optional[np.ndarray] = None,
    overwrite: bool = True,
):
    primary_hdu = fits.PrimaryHDU()

    for key, value in primary_kwds.items():
        primary_hdu.header[key] = value

    image_hdu = fits.ImageHDU(data)

    for key, value in image_kwds.items():
        image_hdu.header[key] = value

    hdu_list = [primary_hdu, image_hdu]

    if roitable:
        # table_hdu = fits.TableHDU(roi_data)
        table_hdu = fits.table_to_hdu(roi_data)

        for key, value in roitable_kwds.items():
            table_hdu.header[key] = value

        hdu_list.append(table_hdu)

    hdul = fits.HDUList(hdu_list)

    hdul.writeto(filename, overwrite=overwrite)
