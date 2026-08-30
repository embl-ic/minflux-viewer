# Parallel agent session summary of the whole extension-layer task

Date: 2026-08-30

Branch: `feat/ext-base`

Phase reviewed: Phase 0 through Phase 2 D1/D2

## Executive verdict

The extension layer fulfils its intended purpose. MINFLUX Viewer now has a
versioned external API, lazy two-tier plugin discovery, real plugin submenus,
shared result and plot windows, background execution, reproducibility
journaling, and a frozen-safe route for installing compiled Python wheels. The
HlyB/D workflow is no longer a permanent built-in menu item; it is the proving
Tier 2 plugin and exercises the public parameter, background, results, plot and
journal surfaces.

The implementation is suitable for the 0.5.0 release. Its strongest qualities
are containment of third-party failures, deliberate Qt ownership rules, the
append-only external-library path policy, and a genuinely tested frozen path.
The principal remaining work is boundary hardening rather than missing
functionality: correct the documented raw-metre coordinate behaviour, remove
or explicitly classify the facade's public `state` escape hatch, give Python
requirements real version/import semantics, and automate the frozen smoke in
the release process.

## Status found at the start of D1

Phase 2 D2 had already landed as commit `abfe3cd` (`docs: release the extension
layer as 0.5.0`). The API, discovery, results/plots and managed-library tracks
were present, but the D1 integration work was not:

- Tier 2 paths were flattened into labels instead of making real submenus.
- Broken plugin entries were listed but remained enabled.
- HlyB/D still registered as a built-in plugin.
- `pip` had not been vendored into the frozen build.
- The spec did not explicitly collect the dynamically imported API namespaces.
- No frozen compiled-wheel or complete frozen extension smoke had been run.

## D1 implementation

### Real plugin menu structure

`DiscoveredPlugin` and `PluginEntry` now preserve a tuple-valued `menu_path`.
The loader returns the manifest's leaf label unchanged and the main window
creates/reuses `QMenu` objects for every path component. Multiple plugins can
therefore share a submenu. Entries with a discovery or requirement error are
visible but disabled, with their explanatory tooltip intact. Command Finder
metadata now points to the discovered plugin's actual entry file.

The old `MENU_SEPARATOR` flattening convention was removed. A stale Phase 0
test that assumed `discover({})` must return an empty list was also corrected;
that assertion had become order-dependent once the repository contained a real
Tier 2 proving plugin.

### HlyB/D proving plugin

The customer-facing orchestration moved to:

- `plugins/hlyb_pair_analysis/plugin.toml`
- `plugins/hlyb_pair_analysis/main.py`

It appears as **Plugins > HlyB/D > Staged pair analysis...** and uses:

- `ctx.ui.ask` for scope and numerical parameters;
- `ctx.data` and `ctx.roi` to capture plain arrays on the GUI thread;
- `ctx.run.background` for the numerical analysis;
- `ctx.results` for tabular output;
- `ctx.plot` for the pair profile and inferred-site view;
- `ctx.journal` and `ctx.ui.log` for reproducibility and user feedback.

Judgement call: the reusable, heavily tested scientific algorithms remain in
`minflux_viewer.analysis`. Customer-specific parameter choices, scope
selection, menu placement and result presentation belong to the external
plugin. Reporting helpers formerly embedded in the built-in runner moved to
`analysis/hlyb_reporting.py`. The retired HlyB variants remain importable and
tested; only their old permanent menu registration was removed, so no
scientific capability was deleted.

This is a meaningful port, but not total physical independence: the plugin
imports `minflux_viewer.analysis.hlyb_staged`. Its manifest consequently pins
MINFLUX Viewer to `>=0.5,<0.6`. It proves the `mfv` application boundary while
reusing the application's analysis engine. A later fully distributable plugin
should either ship that engine itself or consume a separately versioned
analysis service/package.

### Frozen packaging and managed libraries

`tools/vendor_pip.py --version 26.0.1` generated `resources/pip/`; the directory
is intentionally ignored because it is a reproducible build input, not source.
The PyInstaller spec includes it as real files, which is required because pip
cannot run correctly when frozen into the PYZ archive.

The spec also collects every `minflux_viewer.api` submodule. This was found by
an actual frozen boot failure: the facade imports namespace modules from
strings, so static analysis omitted `minflux_viewer.api.roi` until the hidden
imports were added.

The first clean build exposed a separate release-environment hazard: injected
Codex runtime directories on `PATH` caused PyInstaller to collect foreign ICU
and CRT DLLs, after which Qt failed at boot. The verified build used a sanitized
PATH containing only the project venv, CPython, Windows and PowerShell. Its
`Analysis-00.toc` contains zero Codex/OpenAI runtime paths.

## Verification evidence

| Check | Result |
|---|---|
| Focused loader/HlyB/collection/rod tests | 118 passed |
| Full suite, pass 1 | 2,197 passed, 4 skipped |
| Full suite, pass 2 | 2,197 passed, 4 skipped |
| Mixed extension + lifecycle + HlyB suite | 245 passed |
| Python compilation | `compileall` passed for `minflux_viewer` and `plugins` |
| Targeted Ruff check | passed for new/changed production code and primary tests |
| Diff hygiene | `git diff --check` passed |
| PyInstaller | 6.19.0, clean Windows onedir build passed |
| Build provenance | zero foreign Codex/OpenAI runtime paths |
| Vendored pip | 26.0.1 present as filesystem data |
| Frozen package install | Preferences installed compiled `xxhash==3.5.0` into the managed `pylibs` root |
| Frozen restart/import | `xxhash 3.5.0`, hash `06a10c3f2345e9ad` |
| Frozen dataset/plugin smoke | `frozen_sample.csv`, 30 rows; Tier 1 and Tier 2 actions ran |
| Frozen output smoke | results table, plot and dataset render opened; HlyB menu action was present and enabled |
| Frozen shutdown | orderly window close, exit code 0 |

The desktop automation host stopped permitting foreground/cursor input partway
through the run. The required package installation was nevertheless exercised
through the real frozen Preferences UI. The final action smoke used a
test-only PyInstaller runtime hook in a separate `.tmp` build to trigger the
real registered `QAction`s in-process. It used the same sanitized spec and
installed `xxhash`; the production spec was restored to `runtime_hooks=[]`
immediately afterward. No smoke hook or test plugin is part of source or the
release bundle.

## Whole extension-layer assessment

| Area | Assessment | Notes |
|---|---|---|
| `mfv` API 1.0 | Strong | Eight coherent namespaces, backwards-compatible aliases, explicit version contract and actionable errors. |
| Plugin discovery | Strong with identity limitation | Lazy import, namespaced modules, one-time trust, rediscovery and contained errors work well. |
| Menus/Command Finder | Strong | Real nested menus, shared branches, disabled broken entries and source traceability. |
| Results and plots | Strong | Same-name table reuse, model/view separation and explicit lifecycle handling match the product's Qt rules. |
| Background execution | Strong | Viewer-free worker contract, progress/cancellation checkpoint and process-owned retention are appropriate. |
| External libraries | Strong safety posture | Paths append rather than prepend; wheel-only and `--no-deps` protect the bundled numerical/Qt ABI. |
| Frozen delivery | Good, manually proven | Vendored pip and dynamic API collection work; build-environment reproducibility needs automation. |
| HlyB/D proving plugin | Good partial decoupling | Viewer interaction is public-API-only; numerical engine remains app-version-coupled by design. |
| Documentation/release | Strong | `CLAUDE.md`, scripting reference, changelog and 0.5.0 metadata agree with the implementation. |

## Findings and sensible improvements

### Priority 1 — API correctness and boundary

1. **Correct `mfv.data.loc(unit="m")`.** The documentation promises raw metres,
   but the implementation starts from `ds.loc_nm` and divides by `1e9`; Z has
   therefore already received the display `z_scaling_factor`. Either construct
   raw metres from `loc_x/loc_y/loc_z`, or rename/document this as calibrated
   metres. Add a regression with a non-1.0 Z factor. The HlyB plugin correctly
   avoids this ambiguity by reading the three raw attributes.

2. **Close or explicitly bless the facade escape hatch.**
   `MinfluxViewerFacade.state` publicly exposes `AppState`, despite the stated
   rule that external code must use the eight namespaces. Store it privately
   and make namespace bindings use the private reference, or document that it
   is intentionally unsupported and enforce this with an external-API test.

3. **Define the long-term HlyB analysis boundary.** Move the staged engine into
   a separately versioned package/service exposed by `mfv`, or ship it with the
   plugin. Until then, keep the strict app-version range and treat every engine
   signature change as a plugin compatibility event.

### Priority 2 — plugin and packaging robustness

4. **Key discovered entries by stable identity, not display name.** The registry
   currently de-duplicates by `PluginEntry.name`. Two vendors cannot publish
   the same leaf label even in different submenus. Carry the manifest ID/source
   into the registry key; reserve display names for presentation.

5. **Give `requires.python` unambiguous semantics.** Version clauses are stripped
   and only `find_spec()` is checked, so `numpy>=2` does not verify the version.
   Distribution names can also differ from import names (`Pillow`/`PIL`). Use
   `packaging.Requirement` plus distribution metadata, or split the manifest
   into distribution requirements and import probes.

6. **Make sanitized builds the only release path.** Add a checked-in build
   wrapper that constructs a known PATH and fails when `Analysis-00.toc`
   contains DLLs outside approved roots. Record PyInstaller/Python versions and
   generated-pip checksum in the release artifact.

7. **Promote the frozen smoke to automation.** Keep a release-only test that
   installs a compiled wheel into an isolated profile, restarts, imports it,
   loads a small dataset, and triggers Tier 1/Tier 2 results/plot/render actions.
   This is the only test that detects both vendored-pip layout errors and
   dynamic hidden-import omissions.

### Priority 3 — maintainability and user experience

8. **Explain `--no-deps` at the point of installation.** It is the correct ABI
   safeguard, but users need an actionable message when a pure-Python
   dependency must be installed separately.

9. **Resolve frozen API help.** `optimize=1` strips namespace docstrings, so
   `help(mfv.data)` is empty in the bundle. Either build with `optimize=0` or
   ship/open the API reference as frozen data. This is already identified in
   `CLAUDE.md` and remains valid.

10. **Retain the mixed-order regression.** The D1 mixed run found the stale
    `discover({}) == []` assumption that two full suites happened to mask.
    Keep extension-foundation, discovery, lifecycle, results/plot, managed-lib
    and proving-plugin tests together in at least one CI process.

## Files changed by D1

- `.gitignore`
- `minflux_viewer.spec`
- `minflux_viewer/plugins/__init__.py`
- `minflux_viewer/plugins/loader.py`
- removed `minflux_viewer/plugins/hlyb_pair_analysis/`
- `minflux_viewer/analysis/hlyb_reporting.py`
- `plugins/hlyb_pair_analysis/{plugin.toml,main.py}`
- `minflux_viewer/ui/main_window.py`
- `minflux_viewer/ui/command_meta.py`
- `minflux_viewer/ui/hlyb_collection_dialog.py`
- affected HlyB, loader, extension-foundation and rod tests

Generated and ignored verification inputs remain under `resources/pip/`,
`build/`, `dist/` and `.tmp/`. They are not source deliverables.

## Conclusion

The project now has an extension layer rather than merely a script console and
a plugin-shaped menu. A customer workflow can live outside the built-in
registry, negotiate compatibility, collect parameters, run computation safely,
produce reusable UI output, journal its method and install a compiled
dependency in the frozen application. That is the intended architectural
proof. The recommended follow-up is to harden the boundary and release
automation, not to redesign the layer.
