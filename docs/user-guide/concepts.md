# Concepts

How MINFLUX Viewer represents MINFLUX data, and what that means when you
filter, analyse and export.

## Datasets

A **dataset** is one acquisition — or one channel of a multi-colour
acquisition. Every file format is converted to the same internal form when it
is opened, so all windows and analyses work the same way whatever the source.

Operations that change the data itself — cropping (*Edit › Duplicate*), drift
correction, aggregation, flattening an overlay, particle averaging — create a
**new dataset** and leave the original open, so before and after can be
compared. Filters and regions of interest never change the data.

## Localizations, traces and iterations

- A **localization** is one position estimate of an emitter.
- A **trace** (`tid`) groups the localizations recorded while one emitter was
  followed. For a fixed structure the localizations of a trace scatter around
  one position; in a tracking experiment they form a trajectory.
- MINFLUX reaches its final precision in steps: each localization is the
  result of a sequence of **iterations** (`itr`) with a shrinking excitation
  pattern. The data file keeps every iteration.

By default each localization is shown with the values of its **final valid
iteration**; `cfr` and `efc` are taken from the iteration where the instrument
actually measured them. The histogram and the filter can instead show a chosen
iteration, or all iterations pooled (flattened, stacked, summed or averaged).

Only **valid** localizations (`vld`) are loaded by default
(*Edit › Preferences › Data*).

## Coordinates and the Z scaling factor

Coordinates are stored in metres as recorded and shown in nanometres. For 3-D
data, axial distances are distorted by the refractive-index mismatch between
immersion medium and sample. The **Z scaling factor** corrects this as part of
the display; it is never written into the raw coordinates. By default no factor
is applied. It can be estimated from the data (*Analyze › Trace › Estimate Z
Scaling Factor*) or set to a fixed value (*Edit › Preferences › Data*).

A dataset whose Z values span less than 5 nm is treated as 2-D (a 2-D
acquisition records small non-zero Z values); the threshold is set in the same
preferences page.

## Attributes

Every localization carries **attributes**. Hovering over an attribute name in
any dropdown shows its description. The most used:

| Attribute | Meaning |
|---|---|
| `loc` | position, with the instrument's corrections (for example bead-based stabilisation) applied |
| `lnc` | position without those corrections |
| `tim` | time stamp (s) |
| `tid`, `itr`, `vld` | trace id, iteration, validity flag |
| `efo` | photon emission rate at the outer positions of the excitation pattern (Hz) |
| `efc` | photon emission rate at the pattern centre (Hz) |
| `cfr` | centre-frequency ratio `efc/efo`; high values indicate background or more than one emitter |
| `eco`, `ecc` | photon counts at the outer positions and at the centre |
| `dcr` | detection channel ratio, detector 1 / (detector 1 + detector 2), used to separate colours |
| `fbg` | estimated background rate, when the sequence measured it |

**Computed attributes** are added when a dataset opens, for example `siz`
(localizations per trace), `dur` (trace duration), `len` (path length of a
trace), `dst` and `dt` (distance and time to the previous localization of the
trace), `spd` (speed), `tim_trace` (time since the trace started) and `den`
(local density). Which raw and computed attributes appear is set in
*Edit › Preferences › Attributes*.

Attributes such as `siz`, `dur` and `len` belong to a whole trace and are
**trace-wise**: every localization of a trace carries the same value. Any
attribute can also be read per trace — the trace mean, median, minimum,
maximum, first or last value, standard deviation or range — in the histogram
and in filters.

## Filters

A filter is a set of attribute ranges (see [Filtering](filtering.md)). It
selects localizations without deleting anything: every window of the dataset
shows only what passes, and switching a filter off brings everything back.
Filters are saved with the dataset and offered again when it is reopened.

## Channels and overlays

Each colour channel is its own dataset. Channels are combined only for display,
as a **multi-channel overlay** — for example one render with a colour per
channel. Channels can be aligned with the fiducial beads recorded by the
instrument (MBM) or by hand in the render view.

*Process › Channel* combines datasets into an overlay, splits it again, or
flattens it into one dataset for joint analysis. *Convert Dataset to
Multi-Channel Overlay* separates one dataset into channels — by detection
channel ratio (`dcr`), by acquisition time window, or by any attribute.

## Regions of interest

ROIs are drawn with the toolbar tools on the render or scatter views and
managed in *Process › ROI › ROI Manager*. They highlight and select
localizations for measurement, cropping and analysis; they are not filters.
ROI sets can be saved and loaded, including as ImageJ `.roi`/`.zip` files.

## Tracking data

When a dataset contains trajectories rather than repeated localizations of
fixed emitters, the *Tracking View* plays them back and *Analyze › Tracking ›
MSD Analysis* analyses their motion. The viewer guesses whether a channel is a
moving species from how far its traces travel, and the role can be set by hand
in the Tracking View. Static-structure measures — StdDev per trace and FRC — are
refused for tracking channels, because the movement of an emitter is not
localization precision.

## Reproducibility

Each step is recorded in the event **Log** with its parameters.
*Plugins › Generate Method Text* writes a methods paragraph with references
from it, and *Plugins › Macro Recorder* records the steps as a script.
