from pandorasat import __version__
from pandorasat import PandoraSat


def test_version():
    assert __version__ == "0.1.0"


def test_pandorasat():
    PandoraSat()
    return
