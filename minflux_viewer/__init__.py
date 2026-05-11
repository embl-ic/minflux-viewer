"""MINFLUX Data Viewer — Python/Qt port of the MATLAB application."""

from pathlib import Path

__version__ = "0.2.1"
__author__  = "Ziqiang Huang"
__email__   = "ziqiang.huang@embl.de"
__license__ = "MIT"
__institution__ = "EMBL Imaging Centre — Advanced Light Microscopy Service group"


def resource_path(*parts: str) -> Path:
    """Return ``<repo>/resources/<parts>`` as an absolute :class:`Path`."""
    here = Path(__file__).resolve().parent.parent
    return here.joinpath("resources", *parts)
