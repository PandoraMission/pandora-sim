import pandorapsf as pp
import pandorasat as ps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.time import Time
from scipy.interpolate import RectBivariateSpline
import astropy.units as u
from tqdm import tqdm

visda = ps.VisibleDetector()
wcs = visda.get_wcs(0 * u.deg, 0 * u.deg, theta=0 * u.deg)
psf = pp.PSF.from_name("visda", scale=2)

shape = (256 + 1, 256 + 1)

Xpix, Ypix = np.meshgrid(np.linspace(0, 2048, shape[1]), np.linspace(0, 2048, shape[0]))
Xpix, Ypix = Xpix.ravel(), Ypix.ravel()
Yfoc, Xfoc = wcs.sip.pix2foc(np.vstack([Ypix, Xpix]).T, 0).T
c, r = psf.psf_column.value, psf.psf_row.value
C, R = np.meshgrid(c, r)
y0, x0 = np.zeros((2, np.prod(shape)))
for idx in tqdm(range(np.prod(shape))):
    ar = psf.psf(row=Xpix[idx], column=Ypix[idx])
    x0[idx], y0[idx] = np.average(C, weights=ar), np.average(R, weights=ar)
y0, x0 = y0.reshape(shape), x0.reshape(shape)
Ypix, Xpix = Ypix.reshape(shape), Xpix.reshape(shape)
Yfoc, Xfoc = Yfoc.reshape(shape), Xfoc.reshape(shape)

fig, ax = plt.subplots(2, 2, figsize=(10, 13), facecolor="white")
im = ax[0, 0].pcolormesh(Ypix, Xpix, x0, vmin=-3, vmax=3, cmap="coolwarm")
cbar = plt.colorbar(im, ax=ax[0, 0], orientation="horizontal")
cbar.set_label("Offset [pixels]")
ax[0, 0].set(
    xlabel="Pixel Plane Column Position [pixels]",
    ylabel="Pixel Plane Row Position [pixels]",
    title="PSF Center of Mass Column Position [pixels]",
)
im = ax[0, 1].pcolormesh(Ypix, Xpix, y0, vmin=-3, vmax=3, cmap="coolwarm")
cbar = plt.colorbar(im, ax=ax[0, 1], orientation="horizontal")
cbar.set_label("Offset [pixels]")
ax[0, 1].set(
    xlabel="Pixel Plane Column Position [pixels]",
    ylabel="Pixel Plane Row Position [pixels]",
    title="PSF Center of Mass Row Position [pixels]",
)
im = ax[1, 0].pcolormesh(Yfoc, Xfoc, x0, vmin=-3, vmax=3, cmap="coolwarm")
cbar = plt.colorbar(im, ax=ax[1, 0], orientation="horizontal")
cbar.set_label("Offset [pixels]")
ax[1, 0].set(
    xlabel="Focal Plane Column Position [pixels]",
    ylabel="Focal Plane Row Position [pixels]",
    title="PSF Center of Mass Column Position [pixels]",
)
im = ax[1, 1].pcolormesh(Yfoc, Xfoc, y0, vmin=-3, vmax=3, cmap="coolwarm")
cbar = plt.colorbar(im, ax=ax[1, 1], orientation="horizontal")
cbar.set_label("Offset [pixels]")
ax[1, 1].set(
    xlabel="Focal Plane Column Position [pixels]",
    ylabel="Focal Plane Column Position [pixels]",
    title="PSF Center of Mass Row Position [pixels]",
)
plt.subplots_adjust(hspace=0.3, wspace=0.4)
plt.savefig(
    "psfposition_lookuptables/psfposition_lookuptables.png",
    dpi=150,
    bbox_inches="tight",
)

df = pd.DataFrame(
    RectBivariateSpline(Xpix[0], Ypix[:, 0], x0)(
        np.arange(0, 2048), np.arange(0, 2048)
    ),
    columns=np.arange(0, 2048),
    index=np.arange(0, 2048),
)
df.to_csv(
    "psfposition_lookuptables/PandoraPSFPositionLookUp_PSF_Center_x.csv",
    index=False,
    header=False,
)
df = pd.DataFrame(
    RectBivariateSpline(Xpix[0], Ypix[:, 0], y0)(
        np.arange(0, 2048), np.arange(0, 2048)
    ),
    columns=np.arange(0, 2048),
    index=np.arange(0, 2048),
)
df.to_csv(
    "psfposition_lookuptables/PandoraPSFPositionLookUp_PSF_Center_y.csv",
    index=False,
    header=False,
)
df = pd.DataFrame(
    RectBivariateSpline(Xpix[0], Ypix[:, 0], Xfoc)(
        np.arange(0, 2048), np.arange(0, 2048)
    ),
    columns=np.arange(0, 2048),
    index=np.arange(0, 2048),
)
df.to_csv(
    "psfposition_lookuptables/PandoraPSFPositionLookUp_Focal_Plane_Position_x.csv",
    index=False,
    header=False,
)
df = pd.DataFrame(
    RectBivariateSpline(Xpix[0], Ypix[:, 0], Yfoc)(
        np.arange(0, 2048), np.arange(0, 2048)
    ),
    columns=np.arange(0, 2048),
    index=np.arange(0, 2048),
)
df.to_csv(
    "psfposition_lookuptables/PandoraPSFPositionLookUp_Focal_Plane_Position_y.csv",
    index=False,
    header=False,
)

readme = f"""Pandora PSF Position Look Up Tables

These files contain the current best estimates of PSF positions for Pandora as of {Time.now().isot[:10]}. These were created with

- pandorasat version {ps.__version__}
- pandorapsf version {pp.__version__}
    - VISDA PSF created on {pp.config["SETTINGS"]["vis_psf_creation_date"][:10]}
    - NIRDA PSF created on {pp.config["SETTINGS"]["nir_psf_creation_date"][:10]}

The files included are
    - PandoraPSFPositionLookUp_PSF_Center_x.csv
    - PandoraPSFPositionLookUp_PSF_Center_y.csv
    - PandoraPSFPositionLookUp_Focal_Plane_Position_x.csv
    - PandoraPSFPositionLookUp_Focal_Plane_Position_y.csv

Each file contains a CSV table with 2048 x 2048 entries.
The column positions in this table correspond to pixel columns, and the rows in this table correspond to pixel rows.
The units of each file are pixels.
"x" refers to columns and "y" refers to rows. 
All files have been evaluated using the PSF model on a (256 x 256) grid and then interpolated to (2048 x 2048).
The PSF models for Pandora are only available in a small central region. Values outside of this central region are assumed to be the last grid point available.
i.e. Outside of the bounds of the input PSF models from LLNL, this look-up table uses the last available PSF model.

PandoraPSFPositionLookUp_PSF_Center:
------------------------------------

The PandoraPSFPositionLookUp_PSF_Center files contain the Center of Mass for a PSF evaluated at that pixel position in the x (column) and y (row) dimension. 
A PSF located at pixel (0, 0) on the detector pixel plane will have a Center of Mass specified by the (0, 0) coordinate of the PandoraPSFPositionLookUp_PSF_Center file.

PandoraPSFPositionLookUp_Focal_Plane_Position:
----------------------------------------------

The PandoraPSFPositionLookUp_Focal_Plane_Position files contain the focal plane position.
For a PSF located at pixel (0, 0) the position in the undistorted focal plane is given by PandoraPSFPositionLookUp_Focal_Plane_Position.
This file can be used to understand the effect of position distortions. 
"""

with open("psfposition_lookuptables/readme.md", "w") as f:
    f.write(readme)
