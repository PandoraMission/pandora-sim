# Third-party
import astropy.units as u
import numpy as np
import pytest

# First-party/Local
from pandorasim import PACKAGEDIR, PSF, PandoraSim, Target, __version__


def test_version():
    assert __version__ == "0.5.0"


def test_pandorasim():
    p = PandoraSim(ra=180 * u.deg, dec=0 * u.deg, theta=10 * u.deg)
    nirda = p.NIRDA
    visda = p.VISDA
    wavelength = np.linspace(0.1, 2, 1000) * u.micron
    nirda.sensitivity(wavelength)
    visda.sensitivity(wavelength)
    return


@pytest.mark.remote_data
def test_trace():
    p = PandoraSim(ra=180 * u.deg, dec=0 * u.deg, theta=10 * u.deg)
    t = Target.from_gaia("GJ 436")
    nirda = p.NIRDA
    wavelength = np.linspace(0.1, 2, 6000) * u.micron
    spectrum = t.spectrum(wavelength)
    nirda.get_trace(
        wavelength,
        spectrum.value**0 * spectrum.unit,
        target_center=[40, 250],
    )
    return


def test_psf():
    """Test the PSF class"""
    vPSF = PSF.from_file(f"{PACKAGEDIR}/data/pandora_vis_20220506.fits")
    assert vPSF.ndims == 4
    vPSF.prf(vPSF.midpoint)
    vPSF.prf((600, -600, 0.6, 0))
    #    vPSF.prf(vPSF.midpoint, freeze_dimensions=[0, 1, 2, 3])
    #    vPSF.prf(vPSF.midpoint, freeze_dimensions=["column", "row"])
    x, y, prf = vPSF.prf(vPSF.midpoint)
    np.isclose(np.trapz(np.trapz(prf, x, axis=0), y, axis=0), 1, atol=1e-5)

    nirPSF = PSF.from_file(f"{PACKAGEDIR}/data/pandora_nir_20220506.fits")
    assert nirPSF.ndims == 2
    nirPSF.prf(nirPSF.midpoint, location=(1 * u.pixel, 3 * u.pixel))
    nirPSF.prf((1, 10), location=(1, 3))
    x, y, prf = nirPSF.prf(nirPSF.midpoint, location=(0, 0))
    np.isclose(np.trapz(np.trapz(prf, x, axis=0), y, axis=0), 1, atol=1e-5)


# Dead now
# @pytest.mark.remote_data
# def test_visiblesim():
#     theta = 10 * u.deg
#     # Set the target name to any from the target list.
#     targetname = "TRAPPIST-1"
#     c = SkyCoord.from_name(targetname)
#     # Set up the "observatory"
#     p = PandoraSim(c.ra, c.dec, theta)
#     # Get a list of sources nearby that will be on the detector
#     source_catalog = p.SkyCatalog
#     assert isinstance(source_catalog, pd.DataFrame)
#     # Set up jitter properties
#     rowjitter_1sigma = 2 * u.pix
#     coljitter_1sigma = 2 * u.pix
#     jitter_timescale = 1 * u.second
#     # 40 frames
#     nt = 40
#     prf_func = p.VISDA.get_fastPRF(
#         wavelength=0.54 * u.micron, temperature=10 * u.deg_C
#     )
#     # Build the images
#     time, rowcenter, colcenter, thetacenter, science_images = p.get_sky_images(
#         target_ra=c.ra,
#         target_dec=c.dec,
#         theta=theta,
#         nreads=1,
#         nt=nt,
#         rowjitter_1sigma=rowjitter_1sigma,
#         coljitter_1sigma=coljitter_1sigma,
#         jitter_timescale=jitter_timescale,
#         prf_func=prf_func,
#     )
#     assert isinstance(science_images, u.Quantity)
#     assert isinstance(time, u.Quantity)
#     assert isinstance(colcenter, u.Quantity)
#     assert isinstance(rowcenter, u.Quantity)
#     assert isinstance(thetacenter, u.Quantity)
#     assert science_images.shape == (40, 2048, 2048)
#     assert time.shape == (40,)
#     assert colcenter.shape == (40,)
#     assert rowcenter.shape == (40,)
#     assert np.abs(colcenter).max().value < 10
#     assert np.abs(rowcenter).max().value < 10
