"""Holds metadata and methods on Pandora VISDA"""

from dataclasses import dataclass


@dataclass
class VisibleDetector:
    """Holds information on the Visible Detector"""

    def __repr__(self):
        return "Pandora Visible Detector"
