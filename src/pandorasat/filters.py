"""Work with instrument filters"""

from glob import glob

from . import PACKAGEDIR

# Filters bundled with package
filter_fnames = glob(f"{PACKAGEDIR}/data/*.xml")
filters = {
    "_".join([filter.split("/")[-1].split(".")[idx] for idx in [0, 2]]): filter
    for filter in filter_fnames
}
