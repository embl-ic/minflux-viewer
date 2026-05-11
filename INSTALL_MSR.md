# Enabling .msr file support

`.msr` files require **specpy**, Abberior Instruments' Python SDK.
It is a Windows-only binary (`cp312-cp312-win_amd64`) bundled in `vendor/`.

## Requirements

| Requirement | Value |
|-------------|-------|
| OS | Windows 10 / 11 (64-bit) |
| Python | **3.12.x** (must match `cp312`) |
| numpy | **≥ 2.0** |

> **Note:** If you installed the project with `poetry install`, Python 3.12
> and numpy ≥ 2.0 are already satisfied — Poetry's constraint is
> `python = ">=3.10,<3.13"` and numpy is pinned at `>=1.24`. If you are
> on Python 3.10 or 3.11, the specpy wheel will not install; upgrade to
> 3.12 first.

## Install (one command)

Open **PowerShell** inside the project directory and run:

```powershell
poetry run pip install vendor\specpy-1.2.3-cp312-cp312-win_amd64.whl
```

That's it. No restart needed — the viewer detects specpy at runtime.

## Verify

```powershell
poetry run python -c "import specpy; print('specpy OK, version:', specpy.__version__)"
```

Expected output:
```
specpy OK, version: 1.2.3
```

## Uninstall

```powershell
poetry run pip uninstall specpy -y
```

## Troubleshooting

**`ERROR: specpy-1.2.3-cp312-cp312-win_amd64.whl is not a supported wheel
on this platform`**
You are not on Python 3.12. Check with `poetry run python --version`.
If needed, tell Poetry to use a specific interpreter:
```powershell
poetry env use C:\Python312\python.exe
poetry install
poetry run pip install vendor\specpy-1.2.3-cp312-cp312-win_amd64.whl
```

**`ImportError: DLL load failed while importing _specpy`**
The Microsoft Visual C++ 2015-2022 Redistributable is not installed.
Download and install it from:
https://aka.ms/vs/17/release/vc_redist.x64.exe
Then re-launch the viewer.

**Opening `.msr` on Linux / macOS**
specpy is Windows-only. The viewer will show an informative message and
suggest exporting as `.mat` from Imspector instead. All other functionality
(loading `.mat`, filtering, scatter plot, histogram, etc.) works normally.
