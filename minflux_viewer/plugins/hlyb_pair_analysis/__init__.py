"""
minflux_viewer.plugins.hlyb_pair_analysis
=========================================
**HlyB/D subunit pair analysis** — a project-specific plugin that tests a 3-D
MINFLUX dataset for a short-range population of labelling sites, without
assuming a molecular template and without fitting a molecular distance.

The workflow is staged: trace centroids with measured standard errors are
consolidated into labelling-site estimates, split into coarse spatial (cell)
components, reduced to a within-component pair-distance profile, and compared
against a conditional rod-surface randomization that preserves each cell's
observed geometry.  See :mod:`minflux_viewer.analysis.hlyb_staged` for the
algorithm, :mod:`minflux_viewer.ui.hlyb_staged_dialog` for the UI and
:mod:`.runner` for the launch path.

Earlier template-matching and parametric pair-fit workflows are retained in
:mod:`minflux_viewer.analysis.hlyb_clustering` and
:mod:`minflux_viewer.analysis.hlyb_pairwise` but are no longer exposed in the
menus; see ``CLAUDE.md`` for why each was retired.
"""

from __future__ import annotations

from .. import PluginEntry, register


def _launch(state, parent=None) -> None:
    from .runner import run_hlyb_pair_analysis

    run_hlyb_pair_analysis(state, parent)


register(PluginEntry(
    name="HlyB/D subunit pair analysis",
    tooltip="Test a 3-D dataset for a short-range population of labelling "
            "sites against a conditional cell-surface null. Model-independent: "
            "it reports where an excess sits, not a fitted subunit distance.",
    launch=_launch,
    keywords=("hlyb", "hlyd", "dimer", "oligomer", "cluster", "clustering",
              "short range", "excess", "surface null", "conditional "
              "randomization", "ecoli", "e. coli", "membrane", "transporter",
              "site consolidation", "sensitivity audit"),
))
