# Standard library
import json
import sys
from urllib.parse import quote as urlencode

# Third-party
import astropy.units as u
import numpy as np
import pandas as pd
import requests
from astropy.constants import c, h
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.time import Time
from astropy.io import fits

from . import PACKAGEDIR
from . import __version__


def mast_query(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data"""

    # Base API url
    request_url = "https://mast.stsci.edu/api/v0/invoke"

    # Grab Python Version
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent": "python-requests/" + version,
    }

    # Encoding the request as a json string
    req_string = json.dumps(request)
    req_string = urlencode(req_string)

    # Perform the HTTP request
    resp = requests.post(
        request_url, data="request=" + req_string, headers=headers
    )

    # Pull out the headers and response content
    head = resp.headers
    content = resp.content.decode("utf-8")

    return head, content


def get_sky_catalog(
    ra=210.8023,
    dec=54.349,
    radius=0.155,
    magnitude_range=(-3, 16),
    columns="ra, dec, gaiabp",
):
    """We use this instead of astroquery so we can query based on magnitude filters, and reduce the columns

    See documentation at:
    https://mast.stsci.edu/api/v0/_services.html
    https://mast.stsci.edu/api/v0/pyex.html#MastCatalogsFilteredTicPy
    https://mast.stsci.edu/api/v0/_t_i_cfields.html
    """
    request = {
        "service": "Mast.Catalogs.Filtered.Tic.Position.Rows",
        "format": "json",
        "params": {
            "columns": columns,
            "filters": [
                {
                    "paramName": "gaiabp",
                    "values": [
                        {"min": magnitude_range[0], "max": magnitude_range[1]}
                    ],
                }
            ],
            "ra": ra,
            "dec": dec,
            "radius": radius,
        },
    }

    headers, out_string = mast_query(request)
    out_data = json.loads(out_string)

    df = pd.DataFrame.from_dict(out_data["data"])
    s = np.argsort(np.hypot(np.asarray(df.ra) - ra, np.asarray(df.dec) - dec))
    return df.loc[s].reset_index(drop=True)


def photon_energy(wavelength):
    return ((h * c) / wavelength) * 1 / u.photon


def load_vega():
    wavelength, spectrum = np.loadtxt(f"{PACKAGEDIR}/data/vega.dat").T
    wavelength *= u.angstrom
    spectrum *= u.erg / u.cm**2 / u.s / u.angstrom
    return wavelength, spectrum


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


def get_jitter(
    rowstd: float = 1,
    colstd: float = 0.3,
    thetastd: float = 0.0005,
    correlation_time=1 * u.second,
    nframes=200,
    frame_time=0.2 * u.second,
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


def get_flatfield(stddev=0.005, seed=777):
    np.random.seed(seed)
    """ This generates and writes a dummy flatfield file. """
    for detector in ["VISDA", "NIRDA"]:
        hdr = fits.Header()
        hdr["AUTHOR"] = "Christina Hedges"
        hdr["VERSION"] = __version__
        hdr["DATE"] = Time.now().strftime("%d-%m-%Y")
        hdr["STDDEV"] = stddev
        hdu0 = fits.PrimaryHDU(header=hdr)
        hdulist = fits.HDUList(
            [
                hdu0,
                fits.CompImageHDU(
                    data=np.random.normal(1, stddev, (2048, 2048)), name="FLAT"
                ),
            ]
        )
        hdulist.writeto(
            f"{PACKAGEDIR}/data/flatfield_{detector}_{Time.now().strftime('%Y-%m-%d')}.fits",
            overwrite=True,
            checksum=True,
        )
    return


def get_simple_cosmic_ray_image(
    cosmic_ray_expectation=0.4,
    average_cosmic_ray_flux: u.Quantity = u.Quantity(1e3, unit="DN"),
    cosmic_ray_distance: u.Quantity = u.Quantity(0.01, unit=u.pixel / u.DN),
    image_shape=(2048, 2048),
    gain_function=lambda x:x.value*u.electron,
):
    """Function to get a simple cosmic ray image

    This function has no basis in physics at all. 
    The rate of cosmic rays, the energy deposited, sampling distributions, all of it is completely without a basis in physics. 
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
        xs, ys = [xs[0], xs[1] if np.diff(xs) > 0 else xs[1] + 1], [
            ys[0],
            ys[1] if np.diff(ys) > 0 else ys[1] + 1,
        ]

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
            gain_function(wcoords * fper_element).value
        )
    return u.Quantity(im, dtype=int, unit=u.electron)
