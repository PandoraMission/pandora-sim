"""Holds basic metadata on the optics of Pandora"""

from dataclasses import dataclass

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
