# Standard library
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa


def get_version():
    try:
        return version("pandorasim")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

# Standard library
import logging  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("pandorasim")


from .nirsim import NIRSim  # noqa
from .utils import plot_nirda_integrations  # noqa
from .visiblesim import VisibleSim  # noqa
