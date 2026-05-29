"""MINFLUX Data Viewer — Python/Qt port of the MATLAB application."""

from pathlib import Path

__version__ = "0.2.1"
__author__  = "Ziqiang Huang"
__email__   = "ziqiang.huang@embl.de"
__license__ = "MIT"
__institution__ = "EMBL Imaging Centre — Advanced Light Microscopy Service group"


def resource_path(*parts: str) -> Path:
    """Return path to bundled resources; works in dev tree and frozen exe."""
    import sys
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent
    return base.joinpath("resources", *parts)
