<a href="https://github.com/pandoramission/pandora-sim/actions/workflows/tests.yml"><img src="https://github.com/pandoramission/pandora-sim/workflows/pytest/badge.svg" alt="Test status"/></a> [![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://pandoramission.github.io/pandora-sim/)

# PandoraSim

This Python package contains **basic** functions to simulate data from Pandora including creating estimates of, e.g. count rates from targets on the detectors.

### Installation

Eventually you will be able to install this package via pip. For now, clone the github repository, enter the `pandora-sim` directory, and run the command `poetry install` to install the package. This will allow `pandora-sim` to be called like any other Python package.

Make sure to upgrade regularly to get the latest functionality and Pandora metadata.


### Example Usage

Below is an example usage of some of the functionality in this package. For more in-depth walkthroughs and examples of the `pandora-sim` functionality, see the example notebooks in the `docs/` directory.

```python
import pandorasim as ps
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
c = SkyCoord.from_name("GJ 436")
p = ps.PandoraSim(ra=c.ra, dec=c.dec, theta=10*u.deg)
fig = p.plot_footprint()
plt.show()
```

See our API documentation for full details on the functions available in this package.
