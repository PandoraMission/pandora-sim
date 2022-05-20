from pandorasat import PandoraSat, __version__


def test_version():
    assert __version__ == "0.1.0"


def test_pandorasat():
    PandoraSat()
    return
