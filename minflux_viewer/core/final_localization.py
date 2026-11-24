"""Which raw MINFLUX rows are final localizations -- the one rule every reader uses.

A raw m2410 table has one row per (localization x iteration) event. The
localization a viewer shows is the event of its **final** iteration.

When the file records its own status (``sta``), that is authoritative: Abberior
documents ``sta == 1`` as "valid final iteration" (MINFLUX Data Parameters
m2410), and it marks each row individually. That matters when a measurement
runs several threads: a two-thread acquisition whose threads end at different
iterations (one at its 4th, one at its 3rd) lost every final localization of
the shorter thread to the older rule, which kept only the rows at the single
highest iteration. On single-thread data the two rules select identical rows
(checked on every local m2410 dataset).

Including invalid rows, the final *events* are the rows Imspector flags
``fnl`` ("is a final event"), valid or not.

Without ``sta`` (m2205, older exports, generated data) the older rule stands:
the rows at the highest iteration that has a (valid) row.

A ``sta`` column is a status record when it carries Abberior's documented
codes. Then it is authoritative even when **no** row is final: such a table
has no final localizations, and falling back to "the highest iteration" would
present explicitly non-final events (``sta`` 2, "valid, not final") as
localizations. A column of zeros (no documented code) is not a status record
and leaves the older rule in place.

Pure NumPy; no other imports, so any reader can use it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["final_rows", "records_final_status", "STA_VALID_FINAL", "STA_CODES"]

#: ``sta`` value Abberior documents as "valid final iteration".
STA_VALID_FINAL = 1

#: Every ``sta`` code Abberior documents (MINFLUX Data Parameters m2410): 1 valid
#: final iteration, 2 valid not final, 4 derived iteration (aggregation),
#: 5 reserved, 6 failed CFR check, 8 no signal, 9 DAC out of range,
#: 11 background measurement.
STA_CODES = (1, 2, 4, 5, 6, 8, 9, 11)


def records_final_status(sta, n: int) -> bool:
    """Whether *sta* is a per-row status record of length *n*: numeric, 1-D,
    and holding at least one documented code (``STA_CODES``). It need not
    hold a final row -- see the module docstring."""
    if sta is None:
        return False
    arr = np.asarray(sta)
    if arr.ndim != 1 or arr.shape[0] != n or arr.dtype.kind not in "biuf":
        return False
    return bool(np.isin(arr, STA_CODES).any())


def final_rows(itr, vld=None, sta=None, fnl=None, *, vld_only: bool = True) -> np.ndarray:
    """Boolean mask of the final-localization rows.

    * ``sta`` is a status record: ``sta == 1`` (and ``vld``) -- possibly none
      -- with ``vld_only=False``, every row flagged ``fnl`` as well, when the
      file has ``fnl``.
    * otherwise: ``vld`` (if *vld_only*) at the highest iteration among them.
    """
    itr_arr = np.asarray(itr).ravel()
    n = itr_arr.shape[0]
    vld_arr = (np.ones(n, dtype=bool) if vld is None
               else np.asarray(vld, dtype=bool).ravel())
    if records_final_status(sta, n):
        final_valid = (np.asarray(sta).ravel() == STA_VALID_FINAL) & vld_arr
        if vld_only:
            return final_valid
        if fnl is not None and np.asarray(fnl).shape[0] == n:
            return final_valid | np.asarray(fnl, dtype=bool).ravel()
    base = vld_arr if vld_only else np.ones(n, dtype=bool)
    if not base.any():
        return np.zeros(n, dtype=bool)
    finite = base & np.isfinite(itr_arr) if itr_arr.dtype.kind == "f" else base
    if not finite.any():
        return np.zeros(n, dtype=bool)
    return base & (itr_arr == itr_arr[finite].max())
