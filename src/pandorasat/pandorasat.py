"""Holds metadata and methods on Pandora"""

from dataclasses import dataclass

import astropy.units as u
import pandas as pd

from . import PACKAGEDIR
from .irdetector import NIRDetector
from .optics import Optics
from .orbit import Orbit
from .visibledetector import VisibleDetector


@dataclass
class PandoraSat:
    """Holds information and methods for the full Pandora system.

    Args:
        NIRDA (IRDetector): Class of the NIRDA properties
        VISDA (IRDetector): Class of the VISDA properties
        Optics (IRDetector): Class of the Optics properties
        Orbit (IRDetector): Class of the Orbit properties
    """

    Orbit = Orbit()
    Optics = Optics()
    NIRDA = NIRDetector(
        "NIR", 1.19 * u.arcsec / u.pixel, 18.0 * u.um / u.pixel, 2.0 * u.electron / u.DN
    )
    VISDA = VisibleDetector(
        "Visible",
        0.78 * u.arcsec / u.pixel,
        6.5 * u.um / u.pixel,
        2.0 * u.electron / u.DN,
    )
    targetlist = pd.read_csv(f"{PACKAGEDIR}/data/targets.csv")

    def __repr__(self):
        return "Pandora Sat"
