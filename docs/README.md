<a href="https://github.com/pandoramission/pandora-sat/actions/workflows/tests.yml"><img src="https://github.com/pandoramission/pandora-sat/workflows/pytest/badge.svg" alt="Test status"/></a> [![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://pandoramission.github.io/pandora-sat/)

# PandoraSat

This Python package contains metadata for Pandora, and **basic** functions to create estimates of, e.g. count rates from targets on the detectors.

### Installation

To install this package you can use

```
pip install pandora-sat --upgrade
```

Make sure to upgrade regularly to get the latest estimates of the Pandora meta data.


### Example Usage

Below is an example usage of some of the functionality in this package. In general, this package will allow you to get metadata from specific subsystems of Pandora.

```python
from pandorasat import PandoraSat
print(PandoraSat.NIRDA.gain)
print(PandoraSat.Optics.PSF)
print(PandoraSat.VisibleDetector.sensitivity(wavelength))
print(PandoraSat.Orbit.period)
```

See our API documentation for full details on the metadata available in this package.
