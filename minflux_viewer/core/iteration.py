"""
minflux_viewer.core.iteration
==============================
Human-facing iteration-selector labels for the Attribute Plot and Histogram
windows, and the mapping from a label to a retrieval selector + render mode.

Labels are **1-based** (``1st``, ``2nd``, …, ``last (Nth)``) plus the pooled
modes ``all [flatten]``, ``all [stacked]``, ``all [sum]`` and ``all [average]``.
Internally iterations are 0-based in ``mfx_raw``; the last iteration uses the
semantic ``"last"`` selector so it tracks the true final valid iteration per
localization.

Two distinct kinds of pooled mode:

* **render pooling** — ``all [flatten]`` / ``all [stacked]`` keep one value per
  *raw row* and only change how the rows are drawn. Viewer-only.
* **value pooling** — ``all [sum]`` / ``all [average]`` collapse each
  localization's iterations to **one value per localization**, so they behave
  like an ordinary per-loc attribute and can be filtered on.

The fixed dropdown order for a *viewer* iteration selector is: ``last (Nth)``,
``all [flatten]``, ``all [stacked]``, ``all [sum]``, ``all [average]``, then the
individual iterations counting down (``(N-1)th`` … ``2nd``, ``1st``).
The **Filter dialog** uses :func:`filter_iteration_labels` instead: the render
pooling modes are meaningless for a filter, and the value-pooling modes sit at
the bottom (``last (Nth)``, ``(N-1)th`` … ``1st``, ``all [sum]``,
``all [average]``).
"""

from __future__ import annotations


def ordinal(n: int) -> str:
    """English ordinal: 1 -> '1st', 2 -> '2nd', 11 -> '11th', 23 -> '23rd'."""
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


FLATTEN_LABEL = "all [flatten]"
STACKED_LABEL = "all [stacked]"
SUM_LABEL = "all [sum]"
AVERAGE_LABEL = "all [average]"

#: Pooled modes that collapse a localization's iterations to ONE value.
VALUE_POOL_LABELS = (SUM_LABEL, AVERAGE_LABEL)
#: Pooled modes that only change how the raw rows are rendered.
RENDER_POOL_LABELS = (FLATTEN_LABEL, STACKED_LABEL)

#: Short preference keys for the pooled modes -> their dropdown label, in the
#: fixed viewer order. Preferences > Appearance > Histogram Plot stores the
#: keys ("flatten", "stacked", ...) rather than the labels, so the user-facing
#: wording can change without invalidating a saved preference.
POOL_LABEL_BY_KEY: "dict[str, str]" = {
    "flatten": FLATTEN_LABEL,
    "stacked": STACKED_LABEL,
    "sum": SUM_LABEL,
    "average": AVERAGE_LABEL,
}
#: The pooled keys in dropdown order.
POOL_KEYS = tuple(POOL_LABEL_BY_KEY)


def last_label(n_itr: int) -> str:
    """The ``last (Nth)`` label for a dataset with ``n_itr`` iterations."""
    return f"last ({ordinal(max(1, int(n_itr)))})"


def iteration_labels(n_itr: int, allowed: "object | None" = None) -> list[str]:
    """Selector labels for a *viewer* iteration dropdown.

    Returns ``[]`` when there is nothing to browse (``n_itr <= 1``); callers
    should hide the selector in that case and use the default last view.

    ``allowed`` optionally restricts the **pooled** modes to a collection of
    :data:`POOL_KEYS` (``"flatten"``, ``"stacked"``, ``"sum"``, ``"average"``) —
    this is what Preferences > Appearance > Histogram Plot drives. ``None``
    means "all of them" (the default, and what a preference-less caller gets);
    an empty collection means "none of them". ``last (Nth)`` and the individual
    iterations are never gated, and the fixed order is preserved either way.
    """
    if n_itr <= 1:
        return []
    if allowed is None:
        keys = POOL_KEYS
    else:
        wanted = set(allowed)
        keys = tuple(key for key in POOL_KEYS if key in wanted)
    labels = [last_label(n_itr)] + [POOL_LABEL_BY_KEY[key] for key in keys]
    labels += [ordinal(i) for i in range(n_itr - 1, 0, -1)]   # (N-1)th … 1st
    return labels


def filter_iteration_labels(n_itr: int) -> list[str]:
    """Selector labels for the **Filter dialog** ``Iter`` column.

    ``last (Nth)``, the individual iterations counting down, then the two
    value-pooling modes. The render-pooling modes (``all [flatten]`` /
    ``all [stacked]``) are omitted — they do not define a per-localization value
    so they cannot be filtered on. Always returns at least ``["last (1st)"]`` so
    the column shows a meaningful selector for single-iteration data.
    """
    n_itr = max(1, int(n_itr))
    if n_itr <= 1:
        return [last_label(n_itr)]
    labels = [last_label(n_itr)]
    labels += [ordinal(i) for i in range(n_itr - 1, 0, -1)]   # (N-1)th … 1st
    labels += list(VALUE_POOL_LABELS)
    return labels


def parse_iteration_label(label: str) -> tuple["str | int", str]:
    """Map a label to ``(itr_selector, render_mode)``.

    ``itr_selector`` is what to pass to ``mfx_get``: ``"last"``, ``"all"``,
    ``"sum"``, ``"average"``, or a 0-based iteration index. ``render_mode`` is
    ``"single"`` (one series, drawn like the default view), ``"flatten"`` (one
    pooled series over all iterations), or ``"stacked"`` (one colored series
    per iteration). The value-pooling selectors render as ``"single"`` — they
    yield one value per localization, exactly like the default view.
    """
    text = (label or "").strip()
    if text == FLATTEN_LABEL:
        return "all", "flatten"
    if text == STACKED_LABEL:
        return "all", "stacked"
    if text == SUM_LABEL:
        return "sum", "single"
    if text == AVERAGE_LABEL:
        return "average", "single"
    if text.startswith("last") or text == "":
        return "last", "single"
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        return int(digits) - 1, "single"   # 1-based label -> 0-based index
    return "last", "single"


def iteration_selector_label(selector: "str | int | None", n_itr: int) -> str:
    """Inverse of :func:`parse_iteration_label` for a stored filter selector.

    Maps a persisted spec value (``"last"``, ``"sum"``, ``"average"``, a 0-based
    ``int``, or ``None``) back to its dropdown label. ``"effective"`` is not
    resolvable here (it depends on the attribute) — callers resolve it to a
    concrete iteration index first. Returns ``""`` when there is no matching
    label, so the caller can fall back to its own default.
    """
    n_itr = max(1, int(n_itr))
    if selector is None:
        return ""
    if isinstance(selector, bool):
        return ""
    if isinstance(selector, int):
        idx = int(selector)
        if idx == n_itr - 1:
            return last_label(n_itr)
        return ordinal(idx + 1) if 0 <= idx < n_itr else ""
    text = str(selector).strip().lower()
    if text == "sum":
        return SUM_LABEL
    if text == "average":
        return AVERAGE_LABEL
    if text == "last":
        return last_label(n_itr)
    if text.isdigit():
        return iteration_selector_label(int(text), n_itr)
    return ""


def iteration_bold_flags(labels, effective, n_itr: int) -> list[bool]:
    """Which iteration-selector *labels* to render **bold**.

    ``effective`` is a boolean sequence (length ``n_itr``, 0-based) marking the
    iterations that hold real values for the current attribute (see
    ``loader.effective_iterations_for_attr``). An individual ``kth`` label is
    bold when iteration ``k-1`` is effective; ``last (Nth)`` is bold when the
    global-max iteration (``n_itr-1``) is effective. The pooled modes
    (``all [flatten]`` / ``all [stacked]`` / ``all [sum]`` / ``all [average]``)
    are never bold — they span every iteration.
    """
    eff = list(effective)
    out: list[bool] = []
    for label in labels:
        sel, _render = parse_iteration_label(label)
        bold = False
        if isinstance(sel, int):
            bold = 0 <= sel < n_itr and sel < len(eff) and bool(eff[sel])
        elif sel == "last":
            bold = n_itr > 0 and (n_itr - 1) < len(eff) and bool(eff[n_itr - 1])
        out.append(bold)
    return out
