# Third-party
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

# First-party/Local
from pandorasim.targets import Target


@pytest.mark.remote_data
def test_from_methods():
    for input in ["GJ 1214", SkyCoord.from_name("GJ 1214")]:
        for method in [Target.from_gaia, Target.from_TIC]:
            targ = method(input)
            assert isinstance(targ.coord, SkyCoord)
            assert np.all(
                [
                    hasattr(targ, attr)
                    for attr in [
                        "ra",
                        "dec",
                        "teff",
                        "logg",
                        "jmag",
                        "bmag",
                        "planets",
                        "SED",
                    ]
                ]
            )
            assert len(targ.planets) == 1
            assert len(targ.planets["b"]) == 4
