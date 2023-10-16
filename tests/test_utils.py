# Third-party
import astropy.units as u
import numpy as np
import pytest

# First-party/Local
from pandorasim.utils import get_sky_catalog


@pytest.mark.remote_data
def test_get_sky_catalog():
    # Works with no units
    cat = get_sky_catalog(ra=210.8023, dec=54.349, radius=0.05)
    assert isinstance(cat, dict)
    assert np.all(
        [
            k in cat.keys()
            for k in [
                "teff",
                "logg",
                "jmag",
                "bmag",
                "RUWE",
                "ang_sep",
                "coords",
                "source_id",
            ]
        ]
    )
    assert len(cat["coords"]) > 1

    # Works with units
    cat = get_sky_catalog(
        ra=210.8023 * u.deg, dec=54.349 * u.deg, radius=0.05 * u.deg
    )
    assert isinstance(cat, dict)
    assert np.all(
        [
            k in cat.keys()
            for k in [
                "teff",
                "logg",
                "jmag",
                "bmag",
                "RUWE",
                "ang_sep",
                "coords",
                "source_id",
            ]
        ]
    )
    assert len(cat["coords"]) > 1

    # Can return the top 1 hit
    cat = get_sky_catalog(
        ra=210.8023 * u.deg, dec=54.349 * u.deg, radius=0.02 * u.deg, limit=1
    )
    assert len(cat["coords"]) == 1
