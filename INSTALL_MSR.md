# Enabling .msr file support

Opening `.msr` raw acquisition files requires **specpy**, Abberior Instruments'
Python SDK.  specpy is **not on PyPI** and is not bundled in this repository —
it is a proprietary binary that Abberior distributes as part of Imspector.

## Step 1 — Find your specpy wheel

On the Windows PC that runs Imspector, the wheels are at:

```
C:\Imspector\Versions\<your-imspector-version>\python\specpy\
```

Each subfolder matches a Python × NumPy combination, for example:

```
SpecPy-Python3.12-NumPy2.0.0\
    specpy-1.2.3-cp312-cp312-win_amd64.whl
```

Find the subfolder for **your Python version** (`cp312` = Python 3.12, etc.)
and for **your NumPy major version** (1.x or 2.x).

If you are unsure which versions you have:

```powershell
poetry run python --version
poetry run python -c "import numpy; print(numpy.__version__)"
```

## Step 2 — Install

```powershell
# Option A — install directly from the Imspector path
poetry run pip install "C:\Imspector\Versions\<ver>\python\specpy\SpecPy-Python3.12-NumPy2.0.0\specpy-1.2.3-cp312-cp312-win_amd64.whl"

# Option B — copy the wheel to vendor/ first, then install
copy "C:\Imspector\...\specpy-1.2.3-cp312-cp312-win_amd64.whl" vendor\
poetry run pip install vendor\specpy-1.2.3-cp312-cp312-win_amd64.whl
```

No restart needed — the viewer detects specpy at runtime.

## Step 3 — Verify

```powershell
poetry run python -c "import specpy; print('specpy OK, version:', specpy.__version__)"
```

Expected:
```
specpy OK, version: 1.2.3
```

## Uninstall

```powershell
poetry run pip uninstall specpy -y
```

---

## Troubleshooting

**`ERROR: … is not a supported wheel on this platform`**
The wheel's Python tag does not match your interpreter.
Check `poetry run python --version` and pick the matching subfolder in the
Imspector specpy directory.

**`ImportError: DLL load failed while importing _specpy`**
The Microsoft Visual C++ 2015–2022 Redistributable is missing.
Download and install it from:
<https://aka.ms/vs/17/release/vc_redist.x64.exe>

**Opening `.msr` on Linux / macOS**
specpy is Windows-only.  The viewer shows an informative message and suggests
exporting as `.mat` from Imspector instead.  All other functionality works
normally on any platform.

---

## Note on redistribution

specpy is proprietary software owned by Abberior Instruments GmbH.
**Do not commit the `.whl` file to a public repository.**
`vendor/*.whl` and `specpy-*/` are listed in `.gitignore` for this reason.
