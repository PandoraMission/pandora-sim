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

from . import PACKAGEDIR


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
    xstd: float = 1,
    ystd: float = 0.3,
    correlation_time=1 * u.second,
    nframes=200,
    frame_time=0.2 * u.second,
    seed=None,
):
    """Returns some random, time correlated jitter time-series.

    Parameters:
    ----------
    xstd: float
        Standard deviation of jitter in pixels in x axis
    ystd: float
        Standard deviation of jitter in pixels in y axis
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
    x: np.ndarray
        Jitter in the x axis
    y: np.ndarray
        Jitter in the y axis
    """
    time = np.arange(nframes) * frame_time  # noqa:F811
    tstd = (correlation_time / frame_time).value

    def jitter_func(std):
        f = np.random.normal(0, std, size=nframes)
        return convolve(f, Gaussian1DKernel(tstd)) * tstd**0.5

    jitter = []
    for idx, std in enumerate([xstd, ystd]):
        if seed is not None:
            np.random.seed(seed + idx)
        jitter.append(jitter_func(std))

    return time, *jitter * u.pixel
