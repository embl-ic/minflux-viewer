"""
minflux_viewer.core.iteration
==============================
Human-facing iteration-selector labels for the Attribute Plot and Histogram
windows, and the mapping from a label to a retrieval selector + render mode.

Labels are **1-based** (``1st``, ``2nd``, …, ``last (Nth)``) plus the two pooled
modes ``all [flatten]`` and ``all [stacked]``. Internally iterations are 0-based
in ``mfx_raw``; the last iteration uses the semantic ``"last"`` selector so it
tracks the true final valid iteration per localization.

The fixed dropdown order — for every viewer that offers an iteration selector —
is: ``last (Nth)``, ``all [flatten]``, ``all [stacked]``, then the individual
iterations counting down (``(N-1)th`` … ``2nd``, ``1st``).
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


def iteration_labels(n_itr: int) -> list[str]:
    """Selector labels for a dataset with ``n_itr`` iterations.

    Returns ``[]`` when there is nothing to browse (``n_itr <= 1``); callers
    should hide the selector in that case and use the default last view.
    """
    if n_itr <= 1:
        return []
    labels = [f"last ({ordinal(n_itr)})", FLATTEN_LABEL, STACKED_LABEL]
    labels += [ordinal(i) for i in range(n_itr - 1, 0, -1)]   # (N-1)th … 1st
    return labels


def parse_iteration_label(label: str) -> tuple["str | int", str]:
    """Map a label to ``(itr_selector, render_mode)``.

    ``itr_selector`` is what to pass to ``mfx_get``: ``"last"``, ``"all"``, or a
    0-based iteration index. ``render_mode`` is ``"single"`` (one series, drawn
    like the default view), ``"flatten"`` (one pooled series over all
    iterations), or ``"stacked"`` (one coloured series per iteration).
    """
    text = (label or "").strip()
    if text == FLATTEN_LABEL:
        return "all", "flatten"
    if text == STACKED_LABEL:
        return "all", "stacked"
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


def iteration_bold_flags(labels, effective, n_itr: int) -> list[bool]:
    """Which iteration-selector *labels* to render **bold**.

    ``effective`` is a boolean sequence (length ``n_itr``, 0-based) marking the
    iterations that hold real values for the current attribute (see
    ``loader.effective_iterations_for_attr``). An individual ``kth`` label is
    bold when iteration ``k-1`` is effective; ``last (Nth)`` is bold when the
    global-max iteration (``n_itr-1``) is effective. The pooled modes
    (``all [flatten]`` / ``all [stacked]``) are never bold.
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
