"""Functions pertaining to ROI selection"""

from importlib import resources as impresources

import numpy as np
import astropy.units as u
import pandas as pd
from pandorapsf import data
from pandorasat import VisibleDetector

from .docstrings import add_docstring
from .utils import calc_intensity_differences


@add_docstring("nROIs")
def select_ROI_corners(
    ra: float,
    dec: float,
    nROIs: int,
    source_cat: pd.DataFrame,
    locations: pd.DataFrame,
    theta: u.Quantity = 0 * u.deg,
    ROI_size: tuple = (50, 50),
    magnitude_limit: float = 14.0,
    contam_rad: float = 25.0,
    contam_threshold: float = 10.0,
):
    """Selects the corners of ROIs.

    This is currently a placeholder. SOC will provide direction on how ROIs will be selected.

    Parameters:
    -----------
    magnitude_limit : float
        Visual magnitude limit down to which ROI targets will be considered. Default is 14.
    contam_rad : float
        Radius in pixels within which to consider contamination from crowding for a given
        ROI. Default is 25.
    contam_threshold : float
        Value dictating above which the ratio of star flux to all nearby flux is acceptable.
        Stars with contamination ratios below this value will be discarded, e.g. stars whose
        flux is less than 10 times the sum of flux from nearby stars will be discarded.
        Default is 10.
    """
    detector = VisibleDetector()

    center_coords = [tuple((source_cat.ra[0], source_cat.dec[0]))]

    # Define a custom normalization function for use in calculating weights
    def normalize(values, maximum=None, minimum=None, low_offset=0):
        """Normalizes an arbitrary set of values between 1 and some arbitrary value <1.

        Parameters
        ----------
        values : np.ndarray
            Values to normalize.
        maximum : float or None
            Maximum value to set as 1 when normalized. If None, the maximum value in the specified
            `values` will be used.
        minimum : float or None
            Maximum value to set as 0 or the `low_offset` when normalized. If None, the minimum value
            in the specified `values` will be used.
        low_offset : float
            The lower bound of the normalized set, i.e. the `values` will be normalized between
            1 and this value. Default is 0.

        Returns
        -------
        norm_vals : np.ndarray
            Normalized values.
        """
        if maximum is None:
            maximum = np.amax(values)
        if minimum is None:
            minimum = np.amin(values)
        norm_arr = (values - minimum) / (maximum - minimum)
        norm_vals = low_offset + (norm_arr * (1 - low_offset))
        return norm_vals

    def gaussian(r, A, sigma):
        """Function defining an axisymmetric Gaussian centered on the origin"""
        val = A * np.exp(-(r**2) / (2 * sigma**2))
        return val

    # Get functional form of PRF maxima Gaussian fit
    # _, A_fit, sigma_fit = pp.PSF.from_name("visda").calc_prf_maxima()
    gauss_params = impresources.files(data) / "prf_gauss_params.csv"
    A_fit, sigma_fit = np.loadtxt(gauss_params, unpack=True)

    # Weight stars based on PRF maximum value
    pixel_sep = np.hypot(*(locations - np.asarray(detector.shape) / 2).T)
    prf_vals = gaussian(pixel_sep, A_fit, sigma_fit)
    prf_weights = normalize(
        prf_vals, maximum=gaussian(0, A_fit, sigma_fit), low_offset=0.25
    )

    # Weight stars by brightness
    mag_weights = normalize(source_cat.flux)

    # Calculate the contamination of each star by nearby stars within a given radius
    intensity_ratios = calc_intensity_differences(
        locations, source_cat.flux, contam_rad
    )
    intensity_weights = normalize(intensity_ratios)
    # source_cat["intensity_ratio"] = intensity_weights

    # Combine all target weights
    target_weights = prf_weights * mag_weights * intensity_weights

    # Remove stars that are too close to target or edge of fieldstop, stars
    # that are either too dim or might saturate the detector, and stars that
    # are not free of contamination above some threshold
    k = (
        (pixel_sep < 1998)
        & (pixel_sep > 50)
        & (source_cat.mag > 8)
        & (source_cat.mag < magnitude_limit)
        & (intensity_ratios > contam_threshold)
    )
    locations, target_weights, source_cat = (
        locations[k],
        target_weights[k],
        source_cat[k].reset_index(drop=True),
    )

    # Finding the corner of the ROIs for the ranked stars
    size = np.asarray(ROI_size)
    crpix = detector.get_wcs(ra, dec, theta=theta).wcs.crpix
    corners = [(-size[0] // 2 + crpix[0], -size[1] // 2 + crpix[1])]
    while len(corners) < nROIs:
        if len(locations) == 0:
            raise ValueError(f"Can not select {nROIs} ROIs")
        idx = np.argmax(target_weights)
        corner = np.round(locations[idx]).astype(int) - size // 2
        if ~np.any(
            [
                (np.abs(c[0] - corner[0]) < size[0] // 2)
                & (np.abs(c[1] - corner[1]) < size[1] // 2)
                for c in corners
            ]
        ):
            corners.append(tuple(c for c in corner))
            center_coords.append(tuple((source_cat.ra[idx], source_cat.dec[idx])))

        k = ~np.in1d(np.arange(len(locations)), idx)
        locations = locations[k]
        target_weights = target_weights[k]
    return corners, center_coords
