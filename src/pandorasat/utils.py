import astropy.units as u
from astropy.constants import c, h

import numpy as np


def photon_energy(wavelength):
    return ((h * c) / wavelength) * 1 / u.photon
