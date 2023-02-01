# Third-party
import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pytest

# First-party/Local
from pandorasat import PACKAGEDIR, PSF, PandoraSat, Target, __version__


def test_version():
    assert __version__ == "0.1.2"


def test_pandorasat():
    PandoraSat()
    nirda = PandoraSat.NIRDA
    visda = PandoraSat.VISDA
    wavelength = np.linspace(0.1, 2, 1000) * u.micron
    nirda.sensitivity(wavelength)
    visda.sensitivity(wavelength)
    assert np.isclose(nirda.midpoint.value, 1.29750, atol=1e-3)
    assert np.isclose(visda.midpoint.value, 0.55399, atol=1e-3)
    return


@pytest.mark.remote_data
def test_trace():
    t = Target("GJ 436").from_vizier()
    t = Target("GJ 436").from_phoenix()
    nirda = PandoraSat.NIRDA
    wavelength = np.linspace(0.1, 2, 6000) * u.micron
    spectrum = t.model_spectrum(wavelength)
    # nirda.get_trace(t, 2, target_center=(20, 200))

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
    vPSF.prf(vPSF.midpoint, freeze_dimensions=[0, 1, 2, 3])
    vPSF.prf(vPSF.midpoint, freeze_dimensions=["x", "y"])
    x, y, prf = vPSF.prf(vPSF.midpoint)
    np.isclose(np.trapz(np.trapz(prf, x, axis=0), y, axis=0), 1, atol=1e-5)

    nirPSF = PSF.from_file(f"{PACKAGEDIR}/data/pandora_nir_20220506.fits")
    assert nirPSF.ndims == 2
    nirPSF.prf(nirPSF.midpoint, location=(1 * u.pixel, 3 * u.pixel))
    nirPSF.prf((1, 10), location=(1, 3))
    x, y, prf = nirPSF.prf(nirPSF.midpoint, location=(0, 0))
    np.isclose(np.trapz(np.trapz(prf, x, axis=0), y, axis=0), 1, atol=1e-5)


@pytest.mark.remote_data
def test_visiblesim():
    # Set up the "observatory"
    p = PandoraSat()
    # Set the target name to any from the target list.
    targetname = "TRAPPIST-1"
    c = SkyCoord.from_name(targetname)
    # Get a list of sources nearby that will be on the detector
    source_catalog = p.get_sky_catalog(c.ra.deg, c.dec.deg)
    # Set up jitter properties
    xjitter_3sigma = 2*u.pix
    yjitter_3sigma = 2*u.pix
    jitter_timescale = 1*u.second
    # 40 frames
    nt = 40
    prf_func = p.VISDA.get_fastPRF(wavelength=0.54*u.micron, temperature=10*u.deg_C)
    # Build the images
    time, xcenter, ycenter, science_images = p.get_sky_images(source_catalog,
                                                            nreads=1,
                                                            nt=nt, 
                                                            xjitter_3sigma=xjitter_3sigma,
                                                            yjitter_3sigma=yjitter_3sigma, 
                                                            jitter_timescale=jitter_timescale,
                                                            prf_func=prf_func)
    assert isinstance(science_images, u.Quantity)
    assert isinstance(time, u.Quantity)
    assert isinstance(xcenter, u.Quantity)
    assert isinstance(ycenter, u.Quantity)
    assert science_images.shape == (40, 2048, 2048)
    assert time.shape == (40,)
    assert xcenter.shape == (40,)
    assert ycenter.shape == (40,)
    assert np.abs(xcenter).max().value < 10
    assert np.abs(ycenter).max().value < 10
