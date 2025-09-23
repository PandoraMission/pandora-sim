# 1.2.0
- Updated variable names for NIRsim.dark and NIRsim.readnoise to match pandorsat>0.12.2
- Updated variable names for VisibleSim.dark and VisibleSim.readnoise to match pandorsat>0.12.2
- Updated version number dependencies for pandorasat and pandorapsf in pyproject.toml
- Added dev dependencies for pandorasim
- Changed bias in detector class for NIRsim to use the mean bias from pandora_ref (i.e. not be multi-dimensional)
- Changed noise handling in VisibleSim to use the mean bias from pandora_ref (note: underlying bias is still multi-dimensional)

# 1.1.1

- Fixed bug identified by @yoavrotman in #21

# 1.1.0

- Fixed bugs down stream in pandora-psf and pandora-sat
- Updated with pandora-blank changes

# 1.0.0

- Switched PSF and trace functions to PandoraPSF
- New ROI, subarray, and FFI visualization functions
- Changed output to fits files

# 0.5.1

- Added in ability to save VISDA FFIs and ROIs as FITS files
- Updated the intro-to-pandorasim.ipynb tutorial
- Added FFIs and ROIs as properties of PandoraSim class

# 0.5.0

- Integration with new standalone pandora-sat package
- Addition of the ability to plot a NIRDA integration scheme
- Updated docstrings and cleaned unnecessary comments

# 0.4.1

- Renaming to PandoraSim, PandoraSat will be broken out

# 0.3

- Changed API to enforce row-major indexing everywhere
- Added `conventions.ipynb` documentation to state conventions

# 0.2.0

- Added ability to create a distorted WCS
- Added jitter in the observatory rotation angle as well as position
- Increased field stop radius
- Added variable gain
- Added a very simple way to "simulate" cosmic rays

# 0.1.2

- Added capability and documentation to make basic visual simulations
