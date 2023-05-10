__version__ = "0.4.1"
# Standard library
import os  # noqa

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
PANDORASTYLE = "{}/data/pandora.mplstyle".format(PACKAGEDIR)

# Standard library
import logging  # noqa: E402
import shutil  # noqa: E402
from glob import glob  # noqa: E402

# Third-party
from astropy.utils.data import download_file  # noqa: E402

from .utils import get_flatfield  # noqa: E402

logging.basicConfig()
logger = logging.getLogger("pandorasat")

if not os.path.isfile(f"{PACKAGEDIR}/data/pandora_vis_20220506.fits"):
    # Download vis PSF
    logger.warning("No PSF file found. Downloading 100MB VIS PSF file.")
    p = download_file(
        "https://zenodo.org/record/7596336/files/pandora_vis_20220506.fits?download=1",
        pkgname="pandora-sat",
    )
    shutil.move(p, f"{PACKAGEDIR}/data/pandora_vis_20220506.fits")
    logger.warning(
        f"VIS PSF downloaded to {PACKAGEDIR}/data/pandora_vis_20220506.fits."
    )

if not os.path.isfile(f"{PACKAGEDIR}/data/pandora_nir_20220506.fits"):
    # Download nir PSF
    logger.warning("No PSF file found. Downloading 10MB NIR PSF")
    p = download_file(
        "https://zenodo.org/record/7596336/files/pandora_nir_20220506.fits?download=1",
        pkgname="pandora-sat",
    )
    shutil.move(p, f"{PACKAGEDIR}/data/pandora_nir_20220506.fits")
    logger.warning(
        f"NIR PSF downloaded to {PACKAGEDIR}/data/pandora_nir_20220506.fits."
    )

flatnames = glob(f"{PACKAGEDIR}/data/flatfield_*.fits")
if len(flatnames) is None:
    # Make a bogus flatfield
    logger.warning("No flatfield file found. Generating a random one for you.")
    get_flatfield()
    logger.warning(
        f"Generated flatfield in {PACKAGEDIR}/data/pandora_nir_20220506.fits."
    )

from .pandorasat import PandoraSat  # noqa
from .psf import PSF  # noqa
from .targets import Target  # noqa
