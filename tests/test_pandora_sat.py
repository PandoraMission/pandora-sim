import astropy.units as u
import numpy as np

from pandorasat import PandoraSat, Target, __version__


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


def test_psf():
    t = Target("GJ 436").from_vizier()
    t = Target("GJ 436").from_phoenix()
    nirda = PandoraSat.NIRDA
    nirda.get_trace(t, 2, target_center=(20, 200))
    return
