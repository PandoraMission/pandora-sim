"""Holds metadata and methods on Pandora VISDA"""

from dataclasses import dataclass


@dataclass
class VisibleDetector:
    """Holds information on the Visible Detector"""

    # Detector Properties
    npix_column: int = 2048 * u.pixel
    npix_row: int = 2048 * u.pixel
    pixel_scale: float = 0.78 * u.arcsec / u.pixel
    pixel_size: float = 6.5 u.um / u.pixel
    
    def __repr__(self):
        return "Pandora Visible Detector"
