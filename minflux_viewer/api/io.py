"""Public CPU file/batch API, also importable without a running viewer.

Use ``load``, ``save`` and ``batch`` directly in standalone Python, or the same
functions as ``mfv.io`` inside the viewer. No implicit GUI registration occurs.
"""

from ..core.batch_processing import run_batch as batch
from ..core.dataset_io import load_dataset as load
from ..core.dataset_io import save_dataset as save
from ._base import Namespace

__all__ = ["Io", "load", "save", "batch"]


class Io(Namespace):
    """Detached synchronous I/O; submit CPU work via mfv.run.background in Qt."""

    name = "io"
    load = staticmethod(load)
    save = staticmethod(save)
    batch = staticmethod(batch)
