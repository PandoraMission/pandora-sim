"""Holds metadata and methods on Pandora"""

from dataclasses import dataclass
import pandas as pd
from .irdetector import IRDetector
from .optics import Optics
from .orbit import Orbit
from .visibledetector import VisibleDetector
from . import PACKAGEDIR
import numpy as np


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
    targetlist = pd.read_csv(f"{PACKAGEDIR}/data/targets.csv")

    def __repr__(self):
        return "Pandora Sat"
