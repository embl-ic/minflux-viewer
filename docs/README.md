# MINFLUX Viewer manual

MINFLUX Viewer is a desktop application for viewing, filtering, processing and
analysing MINFLUX nanoscopy data, including Abberior Imspector `.msr` files. This
manual describes the current version; the [changelog](../CHANGELOG.md) lists what
changed between releases.

## For users

1. [Getting started](user-guide/getting-started.md) — install, open a first
   dataset, find your way around the windows and menus.
2. [Concepts](user-guide/concepts.md) — how the viewer represents MINFLUX data:
   datasets, localizations, traces, iterations, attributes, filters, channels.
3. [Filtering](user-guide/filtering.md) — attribute filters, trace-level
   filters, the iteration selector, and data-driven *Auto* filtering.
4. [Localization precision and resolution](methods/precision-and-resolution.md)
   — what CRLB, StdDev per trace and FRC measure, how they are computed, and
   their limits.
5. [Command reference](reference/commands.md) — every menu command with its
   default shortcut and parameters (generated from the application).

## For script and plugin authors

- [Scripting and plugin API](scripting-api.md) — the `mfv` API used by the
  Script Editor, user plugins and standalone Python.

## For developers

- [Developer guide](developer/README.md) — code layout, layering rules, tests
  and quality gates, building the application, keeping these documents current.
- [MINFLUX Viewer Zarr v2 format](zarr-v2-format.md) — the application's own
  dataset format.
- [Editing the UI with Qt Designer](ui-designer.md).

## Writing up methods

Every processing and analysis step is recorded in the event Log (*View › Log*).
*Plugins › Generate Method Text* turns that record into a methods paragraph
with the parameters used and the references to cite. *Plugins › Macro
Recorder* records the steps you take while it is on as a script that can
repeat them.

## Getting help

Report problems or file formats that do not open on the
[issue tracker](https://github.com/embl-ic/minflux-viewer/issues). Please
include the application version (*Help › About*), the steps that led to the
problem, the relevant lines of the Log, and — for file problems — a small
sample file.
