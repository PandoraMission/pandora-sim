"""Holds basic metadata on Pandora orbit"""

from dataclasses import dataclass
import astropy.units as u


@dataclass
class Orbit:
    """Holds basic metadata on the orbit of Pandora"""

    period: float = 90 * u.minute

    def __repr__(self):
        return "Pandora Orbit"
