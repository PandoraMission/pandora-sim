__version__ = "0.1.0"
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .pandorasat import PandoraSat  # noqa
from .filters import *
from .targets import *
