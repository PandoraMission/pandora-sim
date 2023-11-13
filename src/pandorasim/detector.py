"""Generic Detector class"""

# Standard library
import abc
from dataclasses import dataclass

# Third-party
import astropy.units as u
import numpy as np
from pandorasat import PandoraSat
# from pandorasat.hardware import Hardware
# from pandorasat.utils import load_vega, photon_energy


@dataclass
class Detector(abc.ABC):
    """Holds information on a Detector

    Attributes
    ----------

    name: str
        Name of the detector. This will determine which files are loaded, choose
        from `"visda"` or `"nirda"`
    pixel_scale: float
        The pixel scale of the detector in arcseconds/pixel
    pixel_size: float
        The pixel size in microns/mm
    gain: float, optional
        The gain in electrons per data unit
    transpose_psf : bool
        Transpose the LLNL input PSF file, i.e. rotate 90 degrees
    """

    # Detector Properties
    name: str
    ra: u.Quantity
    dec: u.Quantity
    theta: u.Quantity
    # pixel_scale: float
    # pixel_size: float
    #    gain: float = 0.5 * u.electron / u.DN
    transpose_psf: bool = False

    def __post_init__(self):
        if self.name == 'nirda':
            self.det = PandoraSat().NIRDA
        elif self.name == 'visda':
            self.det = PandoraSat().VISDA
        self._setup()
        self.zeropoint = self.det.estimate_zeropoint()

#    @property
#    def naxis1(self):
#        """WCS's are COLUMN major, so naxis1 is the number of columns"""
#        return self.shape[1] * u.pixel

#    @property
#    def naxis2(self):
#        """WCS's are COLUMN major, so naxis2 is the number of rows"""
#        return self.shape[0] * u.pixel

    def world_to_pixel(self, ra, dec, distortion=True):
        """Helper function.

        This function ensures we keep the row-major convention in pandora-sim.
        """
        coords = np.vstack(
            [
                ra.to(u.deg).value if isinstance(ra, u.Quantity) else ra,
                dec.to(u.deg).value if isinstance(dec, u.Quantity) else dec,
            ]
        ).T
        if distortion:
            column, row = self.wcs.all_world2pix(coords, 0).T
        else:
            column, row = self.wcs.wcs_world2pix(coords, 0).T
        return np.vstack([row, column])

    def pixel_to_world(self, row, column, distortion=True):
        """Helper function.

        This function ensures we keep the row-major convention in pandora-sat.
        """
        coords = np.vstack(
            [
                column.to(u.pixel).value
                if isinstance(column, u.Quantity)
                else column,
                row.to(u.pixel).value if isinstance(row, u.Quantity) else row,
            ]
        ).T
        if distortion:
            return self.wcs.all_pix2world(coords, 0).T * u.deg
        else:
            return self.wcs.wcs_pix2world(coords, 0).T * u.deg

#    def __repr__(self):
#        return f"Pandora {self.name} Detector"

#    def qe(self, wavelength):
#        """
#        Calculate the quantum efficiency of the detector.

#        Parameters:
#            wavelength (npt.NDArray): Wavelength in microns as `astropy.unit`

#        Returns:
#            qe (npt.NDArray): Array of the quantum efficiency of the detector
#        """
#        pass

#    def throughput(self, wavelength):
#        pass

#    def sensitivity(self, wavelength):
#        sed = 1 * u.erg / u.s / u.cm**2 / u.angstrom
#        E = photon_energy(wavelength)
#        telescope_area = np.pi * (Hardware().mirror_diameter / 2) ** 2
#        photon_flux_density = (
#            telescope_area * sed * self.throughput(wavelength) / E
#        ).to(u.photon / u.second / u.angstrom) * self.qe(wavelength)
#        sensitivity = photon_flux_density / sed
#        return sensitivity

#    @property
#    def midpoint(self):
#        """Mid point of the sensitivity function"""
#        w = np.arange(0.1, 3, 0.005) * u.micron
#        return np.average(w, weights=self.sensitivity(w))

#    def _estimate_zeropoint(self):
#        """Use Vega SED to estimate the zeropoint of the detector"""
#        wavelength, spectrum = load_vega()
#        sens = self.sensitivity(wavelength)
#        zeropoint = np.trapz(spectrum * sens, wavelength) / np.trapz(
#            sens, wavelength
#        )
#        return zeropoint

#    def mag_from_flux(self, flux):
#        return -2.5 * np.log10(flux / self.zeropoint)

#    def flux_from_mag(self, mag):
#        return self.zeropoint * 10 ** (-mag / 2.5)

    def wavelength_to_pixel(self, wavelength):
        if not hasattr(self.det, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self.det._dispersion_df
        return np.interp(
            wavelength,
            np.asarray(df.Wavelength) * u.micron,
            np.asarray(df.Pixel) * u.pixel,
            left=np.nan,
            right=np.nan,
        )

    def pixel_to_wavelength(self, pixel):
        if not hasattr(self.det, "_dispersion_df"):
            raise ValueError("No wavelength dispersion information")
        df = self.det._dispersion_df
        return np.interp(
            pixel,
            np.asarray(df.Pixel) * u.pixel,
            np.asarray(df.Wavelength) * u.micron,
            left=np.nan,
            right=np.nan,
        )
