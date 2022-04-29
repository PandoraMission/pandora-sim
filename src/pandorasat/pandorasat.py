"""Holds metadata and methods on Pandora"""

from dataclasses import dataclass
from .irdetector import IRDetector
from .visibledetector import VisibleDetector
from .orbit import Orbit
from .optics import Optics


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
    NIRDA = IRDetector()
    VISDA = VisibleDetector()

    def __repr__(self):
        return "Pandora Sat"
