"""
minflux_viewer.api
==================
The published ``mfv`` scripting/plugin API — **the contract with code this
project does not own**.

Everything a user script or a third-party plugin is allowed to rely on lives
here. Two rules make it a contract rather than a convenience:

1. **Nothing in this package may touch a private (leading-underscore) attribute
   of ``MainWindow`` or any other internal object.** Reach through the public
   surface only. The precedent is the Dataset Manager's batch actions, which
   were deliberately routed through public ``close_datasets`` /
   ``duplicate_datasets`` / ``combine_datasets_as_overlay`` for exactly this
   reason (see ``CLAUDE.md`` — Data model). A private name that external code
   depends on is a promise nobody made and that will be broken by accident.
2. **Signatures are frozen once released.** :data:`__api_version__` is what a
   plugin's ``plugin.toml`` declares in ``requires.mfv_api``; the plugin loader
   refuses to run a plugin whose requirement this version does not satisfy.
   Adding a namespace or a keyword-with-a-default is a minor bump; removing or
   renaming anything is a major bump.

Namespaces
----------
``data``     datasets, attributes, coordinates, filters — read *and* write
``roi``      list/add/select ROIs, masks, crops, points-in-region
``results``  the shared results table (the ImageJ ``ResultsTable`` analogue)
``plot``     line / scatter / histogram / image windows
``view``     the dataset-owned viewer windows, through public methods only
``ui``       log, status, parameter dialogs, file pickers, message boxes
``run``      background execution, progress, cancellation
``journal``  record a step so it reaches *Generate Method Text*
``record``   record viewer work as a reusable Python script

Scripts do not import this package directly. ``minflux_viewer.scripting``
binds these namespaces to the live :class:`AppState` and installs the result as
a runtime ``mfv`` module; a plugin receives the same object as its ``ctx``
argument.
"""

from __future__ import annotations

#: Version of the published API contract. Semantic:
#:   major — a signature was removed or changed incompatibly
#:   minor — a namespace, function or defaulted keyword was added
#: Declared by plugins as ``requires.mfv_api = ">=1.0,<2.0"``.
__api_version__ = "1.1"

#: Namespace module names, in documentation order. The facade builds one bound
#: instance of each; the plugin loader and the API reference both read this.
NAMESPACES: tuple[str, ...] = (
    "data",
    "roi",
    "results",
    "plot",
    "view",
    "ui",
    "run",
    "journal",
    "record",
)

__all__ = ["__api_version__", "NAMESPACES"]
