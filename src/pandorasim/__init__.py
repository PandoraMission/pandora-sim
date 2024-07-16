__version__ = "1.0.10"
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

# Standard library
import logging  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("pandorasim")


from .nirsim import NIRSim  # noqa
from .utils import plot_nirda_integrations  # noqa
from .visiblesim import VisibleSim  # noqa
