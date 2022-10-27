import astropy.units as u
import numpy as np

from pandorasat import PandoraSat, Target, PSF, __version__
from pandorasat import PACKAGEDIR


def test_version():
    assert __version__ == "0.1.1"


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


def test_trace():
    t = Target("GJ 436").from_vizier()
    t = Target("GJ 436").from_phoenix()
    nirda = PandoraSat.NIRDA
    nirda.get_trace(t, 2, target_center=(20, 200))
    return

def test_psf():
    """Test the PSF class"""
    vPSF = PSF(f"{PACKAGEDIR}/data/pandora_vis_20220506.fits")
    assert vPSF.ndims == 4
    vPSF.prf(vPSF.midpoint)
    vPSF.prf((600, -600, 0.6, 0))
    vPSF.prf(vPSF.midpoint, freeze_dimensions=[0, 1, 2, 3])
    vPSF.prf(vPSF.midpoint, freeze_dimensions=['x', 'y'])
    nirPSF = PSF(f"{PACKAGEDIR}/data/pandora_nir_20220506.fits")
    assert nirPSF.ndims == 1
    nirPSF.prf(nirPSF.midpoint, location=(1*u.pixel, 3*u.pixel))
    nirPSF.prf(1, location=(1, 3))
