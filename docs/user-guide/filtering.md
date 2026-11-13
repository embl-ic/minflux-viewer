# Filtering

*Process › Filter…* (**Shift+F**) opens the filter table of the active
dataset. A filter selects localizations without deleting anything: every view
of the dataset follows it at once, and switching it off restores the full
data. Filters are stored with the dataset (in a `.zarr` store, or in the
processing file written next to other formats) and come back when it is
reopened.

## The filter table

Each row bounds one attribute; a localization passes when it satisfies **every**
switched-on row.

| Column | Meaning |
|---|---|
| **On** | the row takes part in the filter |
| **Attribute** | what is bounded (hover for its description) |
| **Mode** | *per loc* bounds each localization; a *trace* mode (mean, median, min, max, 1st, last, stdev, range) computes one value per trace and keeps or drops the **whole trace** |
| **Iter** | which iteration the value comes from: the final one (default), a single iteration, or all iterations summed or averaged; `cfr` and `efc` default to the iteration where they are measured |
| **Min / Max** | the bounds; the step and number format follow the attribute |
| 👁 | show this filter in the attribute histogram |
| ✕ | remove the row |

Trace-wise attributes such as `siz` (localizations per trace) are always read
per trace.

Below the table:

- **Add filter row** adds a row covering the attribute's full range, switched
  off until you tick it.
- **All filters On / Off** ticks or unticks every row; the rows are kept.
- **Auto** adds bounds computed from the data (next section).
- **Save / Load** writes or reads the filter as a JSON file. A filter file can
  also be dropped onto the dialog. The same file can be applied to another
  dataset.

The line under the table shows how many localizations pass.

### Setting bounds on the histogram

The eye button shows a row's range on the *Attribute Histogram*.
Double-clicking the histogram starts editing the bounds by dragging;
*Update Filter* applies them, *Finish Edit* applies and stops, *Cancel Edit*
restores the previous bounds.

## Auto filtering

**Auto** adds one ordinary filter row per rule — or updates the row already
bounding that attribute — with bounds computed from the data. The rules are
edited in *Edit › Preferences › Process › Filter* and applied top to bottom;
each rule is computed on the localizations that pass the rows already switched
on and the rules before it. The result line lists every bound and how it was
obtained, and the same account goes to the Log and into the generated methods
text. A rule that cannot suggest a bound says why and adds nothing. The rows are
ordinary rows: check them, adjust them or switch them off as with any other.

Three methods are available, for any numeric attribute:

**Single-emitter frequency cut** (default for `efo`). One active emitter emits
at some rate *f1*; two emitters active at the same time emit at twice that rate.
The method fits the logarithm of the rates with two populations — single
emitters, and double emitters held at exactly *2·f1* — and places the upper
bound where a value becomes more likely to come from the double population. It
uses every value rather than a histogram, so no bin width influences the
result. `efo` is computed from photon counts and therefore takes discrete
values; the values are spread within one step before fitting. *f1* is fitted
unless you state it. When the data show no double population, or when more than
35 % of the values would count as double (single emitters then no longer clearly
dominate), the bound is instead set at a stated multiple of *f1* (2 by default;
0 sets no bound), and the result line says so. A bound that removes more than
10 % of the values is flagged.

**Robust spread** (default for `cfr`). The bound is *median + k·σ*, where *σ* is
1.4826 × the median absolute deviation — the robust equivalent of the standard
deviation, unaffected by the outliers it is meant to remove. The default
*k = 2.5* follows Leys et al. (2013). The bound can be upper, lower or both.

**Percentile range** (default for `siz`). Keeps values between two percentiles,
optionally with an absolute floor. The default keeps traces with at least 10
localizations, up to the 99th percentile of trace length.

The **+** button on the preferences page adds a rule, and *Restore default
rules* brings back the three defaults.

### Choosing the iteration for Auto

A rule is computed on the same values its row filters, so the row's **Iter**
setting applies. Use the final iteration (the default) for `efo`: the photon
rate differs between iterations of a sequence, and pooling iterations mixes
populations that have nothing to do with emitter number.

## References

- Leys, C., Ley, C., Klein, O., Bernard, P. & Licata, L. Detecting outliers: do
  not use standard deviation around the mean, use absolute deviation around the
  median. *J. Exp. Soc. Psychol.* 49, 764–766 (2013).
  https://doi.org/10.1016/j.jesp.2013.03.013
