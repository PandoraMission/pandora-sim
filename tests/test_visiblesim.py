from astropy.io import fits
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from pandorasim import VisibleSim, TESTDIR

def test_visible():
    c = SkyCoord.from_name("Kepler-10")
    self = VisibleSim(ra=c.ra, dec=c.dec, roll=-40*u.deg, nROIs=9)
    _ = self.show_FFI()
    plt.savefig(TESTDIR + "output/test_FFI.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    _ = self.show_ROI()
    plt.savefig(TESTDIR + "output/test_ROI.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    def astro_func(time):
        """Input time in jd, output flux at that time-step.

        Parameters:
        -----------
        time: float
            Time step in seconds

        Returns:
        --------
        normalized_flux: float
            The normalized flux of the target at that point in time.
        """
        period = (3*u.hour).to(u.day).value
        return 1 + 0.1 * np.sin((2*np.pi/period) * time + 4.034)
        
    hdulist = self.observe(nframes=300, target_flux_function=astro_func)
    assert isinstance(hdulist, fits.HDUList)