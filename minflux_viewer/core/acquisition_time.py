"""
minflux_viewer.core.acquisition_time
====================================
The **instrument** acquisition timestamp of a dataset — when the microscope
actually recorded it, as opposed to when the file was written, copied or
converted.

Where the fact comes from
-------------------------
Every ``.msr`` carries it inside the dataset's own embedded zarr store, on the
``mfx`` node's ``.zattrs``:

* **m2410** (modern, MFXDTA container v3) — ``acquisition_date``, an ISO 8601
  string *with* UTC offset, e.g. ``"2026-04-21T13:09:33+02:00"``. It is recorded
  **per dataset**, so a file holding three runs carries three different values.
* **m2205** (early, container v2) — no ``acquisition_date``; the equivalent is
  ``tms``, a Unix epoch float (``1654082503.02894``).

Both were validated against Imspector's own auto-generated dataset labels, which
encode ``YYMMDD-HHMMSS`` (``260618-152713_minflux`` matches
``2026-06-18T15:27:13+02:00``; the ``tms`` above matches label ``220601-132142``),
and against the run chronology of a three-run file, whose runs come out strictly
sequential, non-overlapping, and ending seconds before the file was saved.

.. warning::
   The **MFXDTA container timestamp** (``mfxdta.container_timestamp``) is the
   *save/export* time, not the acquisition time. In a modern multi-dataset file
   every stack shares it and it equals the file mtime, while the datasets were
   acquired hours earlier. Do not use it here.

``tim`` and the span
--------------------
``tim`` is seconds since the acquisition start, i.e. the same origin as
``acquisition_date`` (the first localizations appear at ``tim`` around 20-40 s,
after the search phase). So the run ends at ``acquisition_date + max(tim)`` and
the span is ``max(tim)``.

The span is stored as a metadata fact of its own rather than recomputed on
demand: after a ROI crop, a filtered snapshot export or a channel flatten,
``tim`` no longer spans the original run, so a recomputed span would silently
understate the acquisition. Both keys travel with the data:

``metadata["acquisition_date"]``    ISO 8601 string, offset preserved verbatim
``metadata["acquisition_span_s"]``  float seconds, the run duration
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

#: ``dataset.metadata`` key holding the ISO 8601 acquisition start.
ACQUISITION_DATE_KEY = "acquisition_date"
#: ``dataset.metadata`` key holding the acquisition duration in seconds.
ACQUISITION_SPAN_KEY = "acquisition_span_s"

#: Shown for an unknown acquisition, matching the other Data-window rows.
UNKNOWN_TEXT = "—"

# Fixed month names — ``%b`` is locale-dependent and would render "Jun" as
# "Juni" on a German-locale machine.
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


# ---------------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------------
def parse_acquisition_date(value) -> datetime | None:
    """Parse a recorded acquisition date into a ``datetime``.

    Accepts the m2410 ISO 8601 string (offset preserved), the m2205 ``tms``
    Unix epoch (int/float, rendered in local time so it reproduces Imspector's
    own ``YYMMDD-HHMMSS`` label), or an existing ``datetime``. Returns ``None``
    for anything unparseable — a missing date is a normal state, not an error.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        ts = float(value)
        if not np.isfinite(ts) or ts <= 0:
            return None
        try:
            return datetime.fromtimestamp(ts).astimezone()
        except (OSError, OverflowError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def acquisition_date_from_zattrs(zattrs: dict | None) -> str | None:
    """Return the ISO 8601 acquisition start from an ``mfx`` node's ``.zattrs``.

    Prefers the m2410 ``acquisition_date`` verbatim (so the instrument's own UTC
    offset is preserved); falls back to the m2205 ``tms`` epoch, converted to a
    local-time ISO string with offset.
    """
    if not isinstance(zattrs, dict):
        return None
    raw = zattrs.get("acquisition_date")
    if raw not in (None, ""):
        text = str(raw).strip()
        if text and parse_acquisition_date(text) is not None:
            return text
    when = parse_acquisition_date(zattrs.get("tms"))
    return when.isoformat() if when is not None else None


def span_seconds_from_tim(tim) -> float | None:
    """Acquisition duration from a ``tim`` column: ``max(tim)`` in seconds.

    ``tim`` counts from the acquisition start, so the maximum *is* the span —
    the minimum is the delay until the first localization, not a start offset.
    """
    if tim is None:
        return None
    values = np.asarray(tim, dtype=float).ravel()
    if values.size == 0:
        return None
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    span = float(values.max())
    return span if span > 0 else None


# ---------------------------------------------------------------------------
# Dataset accessors
# ---------------------------------------------------------------------------
def dataset_acquisition_date(ds) -> datetime | None:
    """The dataset's recorded acquisition start, or ``None`` when unknown."""
    meta = getattr(ds, "metadata", None) or {}
    return parse_acquisition_date(meta.get(ACQUISITION_DATE_KEY))


def dataset_acquisition_span(ds) -> float | None:
    """The dataset's acquisition duration in seconds, or ``None`` when unknown.

    Uses the recorded span when present; otherwise falls back to the live ``tim``
    column (correct for a freshly loaded dataset, an understatement once rows
    have been cropped away — which is exactly why the value is recorded).
    """
    meta = getattr(ds, "metadata", None) or {}
    recorded = meta.get(ACQUISITION_SPAN_KEY)
    if recorded is not None:
        try:
            span = float(recorded)
        except (TypeError, ValueError):
            span = float("nan")
        if np.isfinite(span) and span > 0:
            return span
    return span_seconds_from_tim(_dataset_tim(ds))


def _dataset_tim(ds):
    """The dataset's ``tim`` column as a 1-D array, or ``None``.

    The all-iteration ``mfx_raw`` store is preferred: it holds the last row the
    instrument actually wrote, whereas the materialized last-valid view ends a
    couple of seconds earlier (its final localization is not the final row).
    """
    raw = getattr(ds, "mfx_raw", None)
    if raw is not None:
        try:
            values = raw.get("tim")
            if values is not None and len(values):
                return values
        except Exception:
            pass
    try:
        from .loader import attr_values_1d

        values = attr_values_1d(ds, "tim")
        if values is not None:
            return values
    except Exception:
        pass
    try:
        return ds.attr["tim"]
    except Exception:
        return None


def acquisition_metadata(ds) -> dict:
    """The acquisition provenance of *ds* as a plain JSON-ready dict.

    Empty when the dataset carries no acquisition date (a converted table, a
    simulation, an export from a tool that never recorded one).
    """
    when = dataset_acquisition_date(ds)
    if when is None:
        return {}
    out: dict = {"date": when.isoformat()}
    span = dataset_acquisition_span(ds)
    if span is not None:
        out["span_s"] = float(span)
        out["end"] = (when + timedelta(seconds=float(span))).isoformat()
    return out


def stamp_acquisition(metadata: dict, date_iso, span_s=None) -> None:
    """Record an acquisition date (and span) onto a ``metadata`` dict in place.

    Silently does nothing when *date_iso* is unparseable, so a caller can pass a
    possibly-absent zattr without guarding. *span_s* may be a number or a raw
    ``tim`` array.
    """
    when = parse_acquisition_date(date_iso)
    if when is None:
        return
    metadata[ACQUISITION_DATE_KEY] = (
        date_iso if isinstance(date_iso, str) and date_iso.strip() else when.isoformat()
    )
    span = span_s
    if isinstance(span, np.ndarray):
        span = span_seconds_from_tim(span)
    try:
        span = float(span) if span is not None else None
    except (TypeError, ValueError):
        span = None
    if span is not None and np.isfinite(span) and span > 0:
        metadata[ACQUISITION_SPAN_KEY] = float(span)


def stamp_dataset_acquisition(ds, date_iso) -> bool:
    """Record *date_iso* on *ds*, taking the span from its own ``tim`` column.

    The span is captured **at load time**, while ``tim`` still covers the whole
    run — a later crop or filtered export would only see a subset. Returns
    whether a date was recorded.
    """
    metadata = getattr(ds, "metadata", None)
    if metadata is None:
        return False
    stamp_acquisition(metadata, date_iso, span_seconds_from_tim(_dataset_tim(ds)))
    return ACQUISITION_DATE_KEY in metadata


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def format_span(seconds: float | None) -> str:
    """Human span: ``"20 hours"``, ``"1 hour 54 min"``, ``"9 min"``, ``"45 s"``."""
    if seconds is None:
        return UNKNOWN_TEXT
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return UNKNOWN_TEXT
    if not np.isfinite(total) or total < 0:
        return UNKNOWN_TEXT

    if total < 60:                         # sub-minute run: seconds are the truth
        return f"{int(round(total))} s"

    hours = int(total // 3600)
    minutes = int(round((total - hours * 3600) / 60.0))
    if minutes == 60:                      # rounding rolled a full hour
        hours += 1
        minutes = 0

    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour" + ("s" if hours != 1 else ""))
    if minutes:
        parts.append(f"{minutes} min")
    if parts:
        return " ".join(parts)
    return f"{int(round(total))} s"


def _stamp(when: datetime) -> str:
    return f"{when.year:04d}-{_MONTHS[when.month - 1]}-{when.day:02d}"


def format_acquisition_range(start: datetime | None, span_seconds: float | None) -> str:
    """``"2026-Jun-26,13:09:33 ~ 15:03:33 (span 1 hour 54 min)"``.

    The start and end are joined by ``~`` ("to"), not a hyphen, which the date
    fields already use as their own separator.

    The end repeats the date only when the run crosses midnight::

        2026-Jun-26,13:00:00 ~ 2026-Jun-27,09:00:00 (span 20 hours)

    With no span the start alone is returned, so a dataset whose ``tim`` was lost
    still reports the one fact it does have.
    """
    if start is None:
        return UNKNOWN_TEXT
    head = f"{_stamp(start)},{start:%H:%M:%S}"
    if span_seconds is None:
        return head
    try:
        span = float(span_seconds)
    except (TypeError, ValueError):
        return head
    if not np.isfinite(span) or span < 0:
        return head

    end = start + timedelta(seconds=span)
    tail = f"{end:%H:%M:%S}" if end.date() == start.date() else f"{_stamp(end)},{end:%H:%M:%S}"
    return f"{head} ~ {tail} (span {format_span(span)})"


def dataset_acquisition_text(ds) -> str:
    """The Dataset-Information ``Acquisition`` row for *ds*."""
    return format_acquisition_range(dataset_acquisition_date(ds), dataset_acquisition_span(ds))
