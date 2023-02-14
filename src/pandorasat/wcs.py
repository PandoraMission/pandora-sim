"""Tools to build and work with Pandora's WCS"""
# Third-party
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS, Sip
import matplotlib.pyplot as plt


from .detector import Detector


def get_wcs(
    detector: Detector,
    target_ra: u.Quantity,
    target_dec: u.Quantity,
    crpix1: int = None,
    crpix2: int = None,
    theta: u.Quantity = 0 * u.deg,
    distortion_file: str = None,
    order: int = 3,
) -> WCS.wcs:
    """Get the World Coordinate System for a detector

    Parameters:
    -----------
    detector : pandorasat.Detector
        The detector to build the WCS for
    target_ra: astropy.units.Quantity
        The target RA in degrees
    target_dec: astropy.units.Quantity
        The target Dec in degrees
    theta: astropy.units.Quantity
        The observatory angle in degrees
    distortion_file: str
        Optional file path to a distortion CSV file. See `read_distortion_file`
    """
    xreflect = True
    yreflect = False
    hdu = fits.PrimaryHDU()
    hdu.header["CTYPE1"] = "RA---TAN"
    hdu.header["CTYPE2"] = "DEC--TAN"
    matrix = np.asarray(
        [
            [np.cos(theta).value, -np.sin(theta).value],
            [np.sin(theta).value, np.cos(theta).value],
        ]
    )
    hdu.header["CRVAL1"] = target_ra.value
    hdu.header["CRVAL2"] = target_dec.value
    for idx in range(2):
        for jdx in range(2):
            hdu.header[f"PC{idx+1}_{jdx+1}"] = matrix[idx, jdx]
    hdu.header["CRPIX1"] = (
        detector.naxis1.value // 2 if crpix1 is None else crpix1
    )
    hdu.header["CRPIX2"] = (
        detector.naxis2.value // 2 if crpix2 is None else crpix2
    )
    hdu.header["NAXIS1"] = detector.naxis1.value
    hdu.header["NAXIS2"] = detector.naxis2.value
    hdu.header["CDELT1"] = detector.pixel_scale.to(u.deg / u.pixel).value * (
        -1
    ) ** (int(xreflect))
    hdu.header["CDELT2"] = detector.pixel_scale.to(u.deg / u.pixel).value * (
        -1
    ) ** (int(yreflect))
    if distortion_file is not None:
        wcs = _get_distorted_wcs(
            detector, hdu.header, distortion_file, order=order
        )
    else:
        wcs = WCS(hdu.header)
    return wcs


def read_distortion_file(detector: Detector, distortion_file: str):
    """Helper function to read a distortion file.

    This file must be a CSV file that contains a completely "square" grid of pixels
    "Parax X" and "Parax Y", and a corresponding set of distorted pixel positions
    "Real X" and "Real Y". These should be centered CRPIX1 and CRPIX2.

    Parameters:
    -----------
    distortion_file: str
        File path to a distortion CSV file.

    Returns:
    --------
    X : np.ndarray
        X pixel positions in undistorted frame, centered around CRPIX1
    Y : np.ndarray
        Y pixel positions in undistorted frame, centered around CRPIX2
    Xp : np.ndarray
        X pixel positions in distorted frame, centered around CRPIX1
    Yp : np.ndarray
        Y pixel positions in distorted frame, centered around CRPIX2
    """
    df = pd.read_csv(distortion_file)
    # Square grid of pixels (TRUTH)
    X = (
        (u.Quantity(np.asarray(df["Parax X"]), "mm") / detector.pixel_size)
        .to(u.pix)
        .value
    )
    Y = (
        (u.Quantity(np.asarray(df["Parax Y"]), "mm") / detector.pixel_size)
        .to(u.pix)
        .value
    )
    # Distorted pixel positions (DISTORTED)
    Xp = (
        (u.Quantity(np.asarray(df["Real X"]), "mm") / detector.pixel_size)
        .to(u.pix)
        .value
    )
    Yp = (
        (u.Quantity(np.asarray(df["Real Y"]), "mm") / detector.pixel_size)
        .to(u.pix)
        .value
    )
    return X, Y, Xp, Yp


def _get_distorted_wcs(
    detector: Detector, hdr: fits.Header, distortion_file: str, order: int = 3
):
    """Helper function to get the distorted WCS coefficients out of a distortion file.

    See https://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf for more information
    """
    # Original, undistorted WCS
    wcs1 = WCS(hdr)
    X, Y, Xp, Yp = read_distortion_file(detector, distortion_file)
    # Rotation matrix from file
    matrix = np.asarray(
        [[hdr[f"PC{idx+1}_{jdx+1}"] for jdx in range(2)] for idx in range(2)]
    )

    # True pixel positions
    pix = np.vstack([X.ravel(), Y.ravel()]).T + np.asarray(
        [hdr["CRPIX1"], hdr["CRPIX2"]]
    )

    # # Square grid of RA and Dec, propagating rotation and scaling
    # RA, Dec = (
    #     matrix.dot(np.asarray([X.ravel().copy(), Y.ravel().copy()]))
    #     * np.asarray([hdr["CDELT1"], hdr["CDELT2"]])[:, None]
    #     + np.asarray([hdr["CRVAL1"], hdr["CRVAL2"]])[:, None]
    # )

    # # Test that producing coordinates by using rotation and scaling matches the WCS output.
    # # If it doesn't, there's something extra about your WCS that pandora-sat isn't expecting
    # diff = np.hypot(*(np.vstack([RA, Dec]) - wcs1.all_pix2world(pix, 1).T))
    # if not (diff < (0.1 * u.arcsecond).to(u.deg).value).all():
    #     raise ValueError("Original WCS does not produce expected RA/Dec")

    # # Distorted grid of RA and Dec, propagating rotation and scaling
    # RAp, Decp = (
    #     matrix.dot(np.asarray([Xp.ravel().copy(), Yp.ravel().copy()]))
    #     * np.asarray([hdr["CDELT1"], hdr["CDELT2"]])[:, None]
    #     + np.asarray([hdr["CRVAL1"], hdr["CRVAL2"]])[:, None]
    # )

    # Below we find the optimum distortion coefficients using linear algebra
    M = np.vstack(
        [
            X.ravel() ** idx * Y.ravel() ** jdx
            for idx in range(order + 1)
            for jdx in range(order + 1)
        ]
    ).T

    coeffs = [
        np.linalg.solve(M.T.dot(M), M.T.dot(-(X - Xp).ravel())).reshape(
            (order + 1, order + 1)
        ),
        np.linalg.solve(M.T.dot(M), M.T.dot(-(Y - Yp).ravel())).reshape(
            (order + 1, order + 1)
        ),
    ]
    coeffsP = [
        np.linalg.solve(M.T.dot(M), M.T.dot((X - Xp).ravel())).reshape(
            (order + 1, order + 1)
        ),
        np.linalg.solve(M.T.dot(M), M.T.dot((Y - Yp).ravel())).reshape(
            (order + 1, order + 1)
        ),
    ]

    # Build a SIP object
    sip = Sip(
        coeffs[0],
        coeffs[1],
        coeffsP[0],
        coeffsP[1],
        (hdr["crpix1"], hdr["crpix2"]),
    )

    # fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12, 3))
    # ax[0].scatter(*pix.T, s=3)
    # ax[1].scatter(*sip.pix2foc(pix, 1).T, s=3)
    # diff = np.hypot(*(pix.T - sip.foc2pix(sip.pix2foc(pix, 1), 1).T))
    # im = ax[2].scatter(*sip.foc2pix(sip.pix2foc(pix, 1), 1).T, s=3, c=diff, cmap='viridis', vmin=0, vmax=np.percentile(diff, 99))
    # cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("|Input - Output| [pixels]")
    # ax[0].set_title("Input Pixels")
    # ax[1].set_title("Pixel -> Focal Plane")
    # ax[2].set_title("Pixel -> Focal Plane -> Pixel")


    # Check that the new correction distorts correctly
    if not np.all(
        np.hypot(
            *(np.vstack([Xp.ravel(), Yp.ravel()]) - sip.pix2foc(pix, 1).T)
        )
        < 0.1
    ):
        raise ValueError("WCS SIP does not produce expected pixel distortion")
    # Check that the new correction goes back to original coordinates correctly
    if not np.all(
        np.hypot(*(pix.T - sip.foc2pix(sip.pix2foc(pix, 1), 1).T)) < 0.1
    ):
        raise ValueError("WCS SIP does not invert precisely")

    # Build a new WCS object
    hdr2 = hdr.copy()
    for idx, L in enumerate("AB"):
        hdr2[f"{L}_ORDER"] = order
        hdr2[f"{L}P_ORDER"] = order
    hdr2["CTYPE1"] = "RA---TAN-SIP"
    hdr2["CTYPE2"] = "DEC--TAN-SIP"

    wcs2 = WCS(hdr2)
    wcs2.sip = sip

    # # Check the full rotation, scaling and distortion produces the expected result
    # fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 4))
    # ax[0].scatter(RA, Dec, label='Input RA/Dec Grid')
    # ax[0].scatter(*wcs1.all_pix2world(pix, 1).T, s=3, label='WCS Processed')
    # ax[0].set_title("Undistorted")
    # ax[0].legend(frameon=True)

    # ax[1].scatter(RAp, Decp, label='Input Distorted RA/Dec')
    # ax[1].scatter(*wcs2.all_pix2world(pix, 1).T, s=3, label='WCS Processed')
    # ax[1].set_title("Distorted")
    # ax[1].legend(frameon=True)
    # print('WCS1\n')
    # print(wcs1.to_header(False).__repr__())
    # print('\n\nWCS2\n')
    # print(wcs2.to_header(False).__repr__())

    # print(wcs2.all_pix2world(pix, 1)[:, 1].max())

    # diffp = np.hypot(*(np.vstack([RAp, Decp]) - wcs2.all_pix2world(pix, 1).T))

    # # Everything should be good to a pixel
    # if not (diffp < (detector.pixel_scale * 1*u.pixel).to(u.deg).value).all():
    #     raise ValueError("New WCS does not produce expected RA/Dec")
    return wcs2
