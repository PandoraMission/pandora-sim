import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits

from pandorasim import TESTDIR, NIRSim


def test_visible():
    c = SkyCoord.from_name("Kepler-10")
    self = NIRSim()
    self.point(ra=c.ra, dec=c.dec, roll=-40 * u.deg)
    _ = self.show_subarray()
    plt.savefig(TESTDIR + "output/test_subarray.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    hdulist = self.observe()
    assert isinstance(hdulist, fits.HDUList)
