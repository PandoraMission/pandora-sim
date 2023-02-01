"""Holds basic metadata on the optics of Pandora"""

# Standard library
from dataclasses import dataclass

# Third-party
import astropy.units as u


@dataclass
class Optics:
    """Holds basic metadata on the optics of Pandora

    Args:
        mirror_diameter (float): Diameter of the Pandora mirror
    """

    mirror_diameter: float = 0.43 * u.m

    def __repr__(self):
        return "Pandora Optics"
