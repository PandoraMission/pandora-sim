__version__ = "0.1.1"
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))

from .pandorasat import PandoraSat  # noqa
from .targets import Target  # noqa
