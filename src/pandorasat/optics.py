"""Holds basic metadata on the optics of Pandora"""

from dataclasses import dataclass
from typing import Tuple

import astropy.units as u
from numpy import typing as npt


@dataclass
class Optics:
    """Holds basic metadata on the optics of Pandora

    Args:
        mirror_diameter (float): Diameter of the Pandora mirror
    """

    mirror_diameter: float = 0.43 * u.m

    def __repr__(self):
        return "Pandora Optics"

    def PSF(self) -> Tuple[npt.NDArray]:
        """Loads the PSF from a file,
        Should return npt.NDArrays of subpixel position and brightness"""
        raise NotImplementedError
