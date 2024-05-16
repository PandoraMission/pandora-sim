__version__ = "1.0.0"
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

# Standard library
import logging  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("pandorasim")


from .visiblesim import VisibleSim   # noqa
