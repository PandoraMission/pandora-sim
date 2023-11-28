# Standard library
import os
import warnings
from functools import lru_cache

# Third-party
import astropy.units as u
import numpy as np
from astropy.constants import c, h
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.coordinates import Distance, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery import log as asqlog
from astroquery.gaia import Gaia
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from copy import copy
import matplotlib.pyplot as plt

from . import PACKAGEDIR, __version__

phoenixpath = f"{PACKAGEDIR}/data/phoenix"
os.environ["PYSYN_CDBS"] = phoenixpath


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Extinction files not found in ")
    # Third-party
    import pysynphot


asqlog.setLevel("ERROR")
# Third-party

frame_dict = {"reset": 1, "read": 2, "drop": 4}


def get_phoenix_model(teff, logg, jmag):
    logg1 = logg.value if isinstance(logg, u.Quantity) else logg
    star = pysynphot.Icat(
        "phoenix",
        teff.value if isinstance(teff, u.Quantity) else teff,
        0,
        logg1 if np.isfinite(logg1) else 5,
    )
    star_norm = star.renorm(
        jmag, "vegamag", pysynphot.ObsBandpass("johnson,j")
    )
    star_norm.convert("Micron")
    star_norm.convert("flam")

    mask = (star_norm.wave >= 0.1) * (star_norm.wave <= 3)
    wavelength = star_norm.wave[mask] * u.micron
    wavelength = wavelength.to(u.angstrom)

    sed = star_norm.flux[mask] * u.erg / u.s / u.cm**2 / u.angstrom
    return wavelength, sed


def get_planets(
    coord: SkyCoord, radius: u.Quantity = 20 * u.arcsecond
) -> dict:
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
                    attr: planets_tab[planets_tab["pl_letter"] == letter][
                        attr
                    ][0].unmasked
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


# def mast_query(request):
#     """Perform a MAST query.

#     Parameters
#     ----------
#     request (dictionary): The MAST request json object

#     Returns head,content where head is the response HTTP headers, and content is the returned data"""

#     # Base API url
#     request_url = "https://mast.stsci.edu/api/v0/invoke"

#     # Grab Python Version
#     version = ".".join(map(str, sys.version_info[:3]))

#     # Create Http Header Variables
#     headers = {
#         "Content-type": "application/x-www-form-urlencoded",
#         "Accept": "text/plain",
#         "User-agent": "python-requests/" + version,
#     }

#     # Encoding the request as a json string
#     req_string = json.dumps(request)
#     req_string = urlencode(req_string)

#     # Perform the HTTP request
#     resp = requests.post(
#         request_url, data="request=" + req_string, headers=headers
#     )

#     # Pull out the headers and response content
#     head = resp.headers
#     content = resp.content.decode("utf-8")

#     return head, content


# def get_sky_catalog(
#     ra=210.8023,
#     dec=54.349,
#     radius=0.155,
#     tessmagnitude_range=(-3, 16),
#     jmagnitude_range=(-3, 20),
#     columns="ra, dec, gaiabp",
# ):
#     """We use this instead of astroquery so we can query based on magnitude filters, and reduce the columns

#     See documentation at:
#     https://mast.stsci.edu/api/v0/_services.html
#     https://mast.stsci.edu/api/v0/pyex.html#MastCatalogsFilteredTicPy
#     https://mast.stsci.edu/api/v0/_t_i_cfields.html
#     """
#     request = {
#         "service": "Mast.Catalogs.Filtered.Tic.Position.Rows",
#         "format": "json",
#         "params": {
#             "columns": columns,
#             "filters": [
#                 {
#                     "paramName": "gaiabp",
#                     "values": [
#                         {
#                             "min": tessmagnitude_range[0],
#                             "max": tessmagnitude_range[1],
#                         }
#                     ],
#                     "paramName": "jmag",
#                     "values": [
#                         {
#                             "min": jmagnitude_range[0],
#                             "max": jmagnitude_range[1],
#                         }
#                     ],
#                 }
#             ],
#             "ra": ra.to(u.deg).value if isinstance(ra, u.Quantity) else ra,
#             "dec": dec.to(u.deg).value if isinstance(dec, u.Quantity) else dec,
#             "radius": radius.to(u.deg).value if isinstance(radius, u.Quantity) else radius,
#         },
#     }

#     headers, out_string = mast_query(request)
#     out_data = json.loads(out_string)
#     df = pd.DataFrame.from_dict(out_data["data"])
#     if len(df) > 0:
#         s = np.argsort(
#             np.hypot(
#                 np.asarray(df.ra) - [ra.to(u.deg).value
#                 if isinstance(ra, u.Quantity)
#                 else ra],
#                 np.asarray(df.dec) - [dec.to(u.deg).value
#                 if isinstance(dec, u.Quantity)
#                 else dec],
#             )
#         )
#         df = df.loc[s].reset_index(drop=True)
#     else:
#         df = pd.DataFrame(columns=columns.split(", "))
#     return df

@lru_cache
def get_sky_catalog(
    ra: float,
    dec: float,
    radius: float = 0.155,
    gbpmagnitude_range: tuple = (-3, 20),
    limit=None,
    gaia_keys: list = [],
    time: Time = Time.now()
) -> dict :
    """
    Gets a catalog of coordinates on the sky based on an input RA, Dec, and radius as well as
    a magnitude range for Gaia. The user can also specify additional keywords to be grabbed
    from Gaia catalog.

    Parameters
    ----------
    ra : float
        Right Ascension of the center of the query radius in degrees.
    dec : float
        Declination of the center of the query radius in degrees.
    radius : float
        Radius centered on ra and dec that will be queried in degrees.
    gbpmagnitude_range : tuple
        Magnitude limits for the query. Targets outside of this range will not be included in
        the final output dictionary.
    limit : int
        Maximum number of targets from query that will be included in output dictionary. If a
        limit is specified, targets will be included based on proximity to specified ra and dec.
    gaia_keys : list
        List of additional Gaia archive columns to include in the final output dictionary.
    time : astropy.Time object
        Time at which to evaluate the positions of the targets in the output dictionary.

    Returns
    -------
    cat : dict
        Dictionary of values from the Gaia archive for each keyword.
    """

    base_keys = ["source_id",
                 "ra",
                 "dec",
                 "parallax",
                 "pmra",
                 "pmdec",
                 "radial_velocity",
                 "ruwe",
                 "phot_bp_mean_mag",
                 "teff_gspphot",
                 "logg_gspphot",
                 "phot_g_mean_flux",
                 "phot_g_mean_mag",]

    all_keys = base_keys + gaia_keys

    query_str = f"""
    SELECT {f'TOP {limit} ' if limit is not None else ''}* FROM (
        SELECT gaia.{', gaia.'.join(all_keys)}, dr2.teff_val AS dr2_teff_val,
        dr2.rv_template_logg AS dr2_logg, tmass.j_m, tmass.j_msigcom, tmass.ph_qual, DISTANCE(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        POINT(gaia.ra, gaia.dec)) AS ang_sep,
        EPOCH_PROP_POS(gaia.ra, gaia.dec, gaia.parallax, gaia.pmra, gaia.pmdec,
        gaia.radial_velocity, gaia.ref_epoch, 2000) AS propagated_position_vector
        FROM gaiadr3.gaia_source AS gaia
        JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS xmatch USING (source_id)
        JOIN gaiadr3.dr2_neighbourhood AS xmatch2 ON gaia.source_id = xmatch2.dr3_source_id
        JOIN gaiadr2.gaia_source AS dr2 ON xmatch2.dr2_source_id = dr2.source_id
        JOIN gaiadr3.tmass_psc_xsc_join AS xjoin USING (clean_tmass_psc_xsc_oid)
        JOIN gaiadr1.tmass_original_valid AS tmass ON
        xjoin.original_psc_source_id = tmass.designation
        WHERE 1 = CONTAINS(
        POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
        CIRCLE(gaia.ra, gaia.dec, {(u.Quantity(radius, u.deg) + 50*u.arcsecond).value}))
        AND gaia.parallax IS NOT NULL
        AND gaia.phot_bp_mean_mag > {gbpmagnitude_range[0]}
        AND gaia.phot_bp_mean_mag < {gbpmagnitude_range[1]}) AS subquery
    WHERE 1 = CONTAINS(
    POINT({u.Quantity(ra, u.deg).value}, {u.Quantity(dec, u.deg).value}),
    CIRCLE(COORD1(subquery.propagated_position_vector), COORD2(subquery.propagated_position_vector), {u.Quantity(radius, u.deg).value}))
    ORDER BY ang_sep ASC
    """
    job = Gaia.launch_job_async(query_str, verbose=False)
    tbl = job.get_results()
    if len(tbl) == 0:
        raise ValueError("Could not find matches.")
    plx = tbl["parallax"].value.filled(fill_value=0)
    plx[plx < 0] = 0
    cat = {
        "jmag": tbl["j_m"].data.filled(np.nan),
        "bmag": tbl["phot_bp_mean_mag"].data.filled(np.nan),
        "gmag": tbl["phot_g_mean_mag"].data.filled(np.nan),
        "gflux": tbl["phot_g_mean_flux"].data.filled(np.nan),
        "ang_sep": tbl["ang_sep"].data.filled(np.nan) * u.deg,
    }
    cat["teff"] = (
        tbl["teff_gspphot"].data.filled(
            tbl["dr2_teff_val"].data.filled(np.nan)
        )
        * u.K
    )
    cat["logg"] = tbl["logg_gspphot"].data.filled(
        tbl["dr2_logg"].data.filled(np.nan)
    )
    cat["RUWE"] = tbl["ruwe"].data.filled(99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat["coords"] = SkyCoord(
            ra=tbl["ra"].value.data * u.deg,
            dec=tbl["dec"].value.data * u.deg,
            pm_ra_cosdec=tbl["pmra"].value.filled(fill_value=0)
            * u.mas
            / u.year,
            pm_dec=tbl["pmdec"].value.filled(fill_value=0) * u.mas / u.year,
            obstime=Time.strptime("2016", "%Y"),
            distance=Distance(parallax=plx * u.mas, allow_negative=True),
            radial_velocity=tbl["radial_velocity"].value.filled(fill_value=0)
            * u.km
            / u.s,
        ).apply_space_motion(time)
    cat["source_id"] = np.asarray(
        [f"Gaia DR3 {i}" for i in tbl["source_id"].value.data]
    )
    for key in gaia_keys:
        cat[key] = tbl[key].data.filled(np.nan)
    return cat


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
    correlation_time: float = 1 * u.second,
    nframes: int = 200,
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
    gain_function=lambda x: x.value * u.electron,
):
    """
    Function to get a simple cosmic ray image

    This function has no basis in physics at all. The rate of cosmic rays, the energy deposited,
    sampling distributions, all of it is completely without a basis in physics. All this function
    can do is put down fairly reasonable "tracks" that mimic the impact of cosmic rays, with some
    tuneable parameters to change the properties.

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
        np.zeros(SC_Resets2, int) + frame_dict["reset"],
        *[
            np.zeros(SC_ReadFrames, int) + frame_dict["read"],
            np.zeros(SC_DropFrames2, int) + frame_dict["drop"],
        ]
        * (SC_Groups - 1),
        np.zeros(SC_ReadFrames, int) + frame_dict["read"],
    ]
    cint1 = copy(cintn)
    cint1.pop(0)

    cint1.insert(0, np.zeros(SC_Resets1, int) + frame_dict["reset"])
    cint1.insert(1, np.zeros(SC_DropFrames1, int) + frame_dict["drop"])

    integrations = [cint1]
    for idx in np.arange(SC_Integrations - 1):
        integrations.append(copy(cintn))

    integrations[-1].append(np.zeros(SC_DropFrames3, int) + frame_dict["drop"])
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
    s = np.where(inte | frame_dict["reset"] == frame_dict["reset"])[0]
    if len(s) > 0:
        s = s[-1]
        y = np.hstack(
            [np.zeros(s + 1, dtype=int), np.arange(s + 1, len(inte), dtype=int) - s + 1]
        )
    else:
        y = np.arange(0, len(inte), dtype=int)

    cols = np.vstack(
        [
            inte | frame_dict["reset"] == frame_dict["reset"],
            inte | frame_dict["read"] == frame_dict["read"],
            inte | frame_dict["drop"] == frame_dict["drop"],
        ]
    ).T
    return np.hstack([y[:, None], cols])


def plot_integrations(
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
        np.sum([(i == frame_dict["read"]).all() for i in inte if len(i) > 0])
        for inte in integrations
    ]
    integrations = [np.hstack(idx) for idx in integrations]
    dat = np.vstack([get_plot_vectors(inte) for inte in integrations])
    with plt.style.context("seaborn-white"):
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
            title=(f"{SC_Integrations} Integrations, " +
                   f"{SC_Groups} Groups\n{SC_ReadFrames} Read/group, " +
                   f"{SC_DropFrames2} Drop/group\n{len(dat)} total reads, " +
                   f"{np.sum(cadences)} total cadences"),
        )
        ax.legend(frameon=True, bbox_to_anchor=(1.1, 1.05))
    return fig
