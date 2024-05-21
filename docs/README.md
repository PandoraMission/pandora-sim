<a href="https://github.com/pandoramission/pandora-sim/actions/workflows/tests.yml"><img src="https://github.com/pandoramission/pandora-sim/workflows/pytest/badge.svg" alt="Test status"/></a> [![Generic badge](https://img.shields.io/badge/documentation-live-blue.svg)](https://pandoramission.github.io/pandora-sim/)

# PandoraSim

This Python package contains classes to simulate data from Pandora, and provides estimates for level 1 products from the spacecraft.

## Installation

If you are installing the simulator and expect you will not wish to tweak any of the code internal to the simulator or any other aspects of DPC software, you can install `pandora-sim` with `pip`

```
pip install pandorasim --upgrade
```

However, if you are either:

1. Expect to tweak/update `pandorasim` or any of the Pandora software dependencies
2. Need to run `pandorasim` in a different environment to your native environment

You may want to install with `poetry`. You can do this with

```
pip install --upgrade poetry
git clone https://github.com/PandoraMission/pandora-sim
cd pandorasim
poetry install
```

To run `pandorasim` you can then work however you work in Python, be make sure you

1. Are working in the `pandorasim` directoy
2. Use the correct `poetry` environment by prepending all your commands with `poetry run`. E.g. `poetry run jupyterlab`, `poetry run python`, `poetry run pytest` etc.

### Dependencies

This package depends on two other packages from the Pandora software ecosystem.

- [`pandorasat`](https://github.com/PandoraMission/pandora-sat/)
- [`pandorapsf`](https://github.com/PandoraMission/pandora-psf/)

Each of these packages are **updated often** as we gain new insights into what to expect from Pandora. If you are working with the simulator, you should make sure to keep your versions of all packages updated.
