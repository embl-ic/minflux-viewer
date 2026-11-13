# Localization precision and resolution

*Analyze › Loc Precision / Resolution* offers three measures. They answer
different questions:

- **Localization precision** — how well one emitter is localized: the spread
  of repeated position estimates of the same emitter. *CRLB* gives the bound
  the measurement model allows; *StdDev per trace* measures the spread actually
  achieved.
- **Resolution** — how fine a structure the reconstructed image can show. It
  depends on precision **and** on how densely the structure is labelled and
  sampled. *FRC* estimates it from the data.

A small localization precision with sparse labelling does not make a high
resolution image, and a resolution value is not a per-localization uncertainty.

## CRLB — Cramér–Rao lower bound

The lowest standard deviation an unbiased estimator can reach for each
localization under the MINFLUX targeted-excitation model (Balzarotti et al.
2017), in the closed forms of Marin & Ries (2024):

```
lateral  σ = L / √(8N) · 1 / (1 − L² / (8 σq²)) · √((1 + ρ)(1 + ρ(1 + c)))
axial    σ = Lz / (4 √Nz) · √((1 + ρ)(1 + ρ(1 + c)))
```

- `N` — photons of the final-scale iterations of that axis (`eco` + `ecc`).
- `L` — the pattern diameter, `patGeoFactor × 360 nm` from the acquisition
  sequence. Without sequence information 39.6 nm is assumed, and the report
  says so.
- `σq` — the width of the model's doughnut; 131.6 nm (a 310 nm FWHM doughnut)
  is assumed. It changes the result by about 1 % at L = 40 nm.
- `ρ = b / (efo − b)` — background relative to signal, with the background `b`
  taken from the localization's own centre exposure, else an earlier centre
  exposure scaled by laser power, else the instrument's `fbg` estimate;
  `c` — centre dwell divided by the whole-orbit dwell.

Each axis gets a status: ideal only (no background information), background
estimated, background inconsistent (background at or above the measured rate,
so no finite bound exists — these localizations are counted, not hidden),
invalid input, or unsupported pattern geometry. For static data the report
also compares, on the same traces, the measured per-trace spread with the
traces' RMS bound. A spread *below* the bound is reported as unexplained; no
attempt is made to attribute the difference.

The CRLB depends on assumptions — whether the recorded counts include
background is not documented by the instrument — and it is a model bound, not
the experiment's measured precision.

## StdDev per trace

For each trace with at least 5 localizations, the sample standard deviation
(`ddof = 1`) of each coordinate, from the raw coordinates (no Z scaling
factor). The lateral spread of a trace is `σr = √((σx² + σy²) / 2)`. The
reported value is the standard error of the mean position averaged over
traces, `⟨σ / √n⟩`; the medians and means of the per-trace `σ` are shown too,
with their distributions (Ostersehlt et al. 2022).

The spread of a moving emitter is dominated by its motion, so this measure is
refused for channels classified as tracking.

## FRC — Fourier ring correlation

The localizations are split at random into two halves, each half is rendered as
a histogram image, and the two images are compared ring by ring in Fourier
space (Banterle et al. 2013; Nieuwenhuizen et al. 2013):

```
FRC(q) = Σring Re{F1·F2*} / √(Σring |F1|² · Σring |F2|²)
```

The curve is averaged over 5 random splits, smoothed (Savitzky–Golay, window 7,
order 1), and its high-frequency end beyond 0.8 of the maximum frequency is
discarded. The resolution is `1 / q` at the first frequency where the curve
falls below 1/7, the MINFLUX convention of Gwosch et al. (2020) and Ostersehlt
et al. (2022). The image grid covers the data with 1024 to 4096 pixels per
side, aiming at a 2 nm pixel (finer for small fields, coarser for fields wider
than about 8 µm); the pixel can be set and the FRC recomputed. If the curve never falls below 1/7, the result is limited by the
pixel and the window says so.

- **Per localization** (default) uses every localization. When emitters are
  localized many times, repeated localizations of one emitter fall in both
  halves and make the estimate optimistic.
- **Per-trace combined** first replaces each trace by its mean position, so it
  measures the resolution of the emitter positions themselves.

### Resolution over acquisition time

The *Over acquisition time* tab repeats the FRC on cumulative time windows —
all localizations recorded from the start up to the end of each window — to
show how the resolution develops and whether a longer acquisition would still
have helped. Every window uses the same field of view, pixel, mode and number
of splits, so the windows differ only in the localizations they contain, and
the last window reproduces the whole-acquisition value. The error bars are the
standard deviation of the resolutions of the individual random splits. The
computation runs in the background and can be stopped.

FRC assumes a static structure and is refused for tracking channels.

## References

- Balzarotti, F. et al. Nanometer resolution imaging and tracking of
  fluorescent molecules with minimal photon fluxes. *Science* 355, 606–612
  (2017). https://doi.org/10.1126/science.aak9913
- Marin, Z. & Ries, J. arXiv:2410.12427 (2024).
  https://arxiv.org/abs/2410.12427
- Ostersehlt, L. M. et al. DNA-PAINT MINFLUX nanoscopy. *Nat. Methods* 19,
  1072–1075 (2022). https://doi.org/10.1038/s41592-022-01577-1
- Gwosch, K. C. et al. MINFLUX nanoscopy delivers 3D multicolor nanometer
  resolution in cells. *Nat. Methods* 17, 217–224 (2020).
  https://doi.org/10.1038/s41592-019-0688-0
- Banterle, N., Bui, K. H., Lemke, E. A. & Beck, M. Fourier ring correlation as
  a resolution criterion for super-resolution microscopy. *J. Struct. Biol.*
  183, 363–367 (2013). https://doi.org/10.1016/j.jsb.2013.05.004
- Nieuwenhuizen, R. P. J. et al. Measuring image resolution in optical
  nanoscopy. *Nat. Methods* 10, 557–562 (2013).
  https://doi.org/10.1038/nmeth.2448
