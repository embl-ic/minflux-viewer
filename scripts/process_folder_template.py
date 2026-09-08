"""
Process Folder — batch template for the MINFLUX Viewer Script Editor.

The MINFLUX counterpart of Fiji's ``Process_Folder.py``: pick an input folder,
a filter and an output folder, then run the same processing over every file
that matches. Open it in *Plugins > Script Editor* and press Run.

Fiji declares its parameters with ``#@`` annotations; the equivalent here is
``mfv.ui.ask()``, which builds one dialog from a dict of defaults and returns
``None`` when the user cancels.

    +--------------------------------------------------------------+
    |  Put your processing commands in process_one() -- see the     |
    |  "YOUR PROCESSING GOES HERE" block. Everything around it is   |
    |  the folder walk, the filtering and the saving.               |
    +--------------------------------------------------------------+
"""

# `mfv` is injected by the Script Editor at run time, the way Fiji injects its
# #@ parameters, so a static checker cannot see it.
# ruff: noqa: F821

import os

# ---------------------------------------------------------------------------
# 1. Parameters -- one dialog, the mfv equivalent of Fiji's #@ annotations.
# ---------------------------------------------------------------------------
# Leave a directory blank to get a folder picker instead of typing the path.
params = mfv.ui.ask(
    {
        "Input directory": "",
        "Output directory": "",
        "File extension": ".mat",
        "File name include": "",
        "File name exclude": "",
        "Search subfolders": True,
        "Keep directory structure": True,
    },
    title="Process Folder",
    descriptions={
        "Input directory": "Folder to read from. Blank opens a folder picker.",
        "Output directory": "Folder to write to. Blank opens a folder picker.",
        "File extension": "Only files ending in this are processed, e.g. .mat",
        "File name include": "Comma-separated. A file is kept when its name "
                             "contains ANY of these. Blank keeps all.",
        "File name exclude": "Comma-separated. A file is skipped when its name "
                             "contains ANY of these. Blank excludes nothing.",
        "Search subfolders": "Walk subfolders too, instead of the top level only.",
        "Keep directory structure": "Mirror the input folder tree under the "
                                    "output folder, instead of writing flat.",
    },
)
if params is None:
    raise SystemExit("Cancelled.")


def _tokens(text):
    """Split a comma-separated field into lowercase tokens, dropping blanks."""
    return [t.strip().lower() for t in str(text).split(",") if t.strip()]


def _folder(value, caption):
    """A typed path, or a picker when the field was left blank."""
    value = str(value).strip().strip('"')
    if not value:
        value = mfv.ui.choose_dir(caption=caption)
    if not value:
        raise SystemExit("Cancelled.")
    return os.path.abspath(value)


SRC_DIR = _folder(params["Input directory"], "Input directory")
DST_DIR = _folder(params["Output directory"], "Output directory")
EXT = str(params["File extension"]).strip()
INCLUDE = _tokens(params["File name include"])
EXCLUDE = _tokens(params["File name exclude"])
RECURSIVE = bool(params["Search subfolders"])
KEEP_TREE = bool(params["Keep directory structure"])


# ---------------------------------------------------------------------------
# 2. Which files to process
# ---------------------------------------------------------------------------
def wanted(filename):
    """Whether *filename* passes the extension / include / exclude filters."""
    name = filename.lower()
    if EXT and not name.endswith(EXT.lower()):
        return False
    if INCLUDE and not any(token in name for token in INCLUDE):
        return False
    if EXCLUDE and any(token in name for token in EXCLUDE):
        return False
    return True


def find_files():
    """Every matching file under SRC_DIR, sorted, as (folder, filename)."""
    found = []
    if RECURSIVE:
        for root, _dirs, filenames in os.walk(SRC_DIR):
            for filename in sorted(filenames):
                if wanted(filename):
                    found.append((root, filename))
    else:
        for filename in sorted(os.listdir(SRC_DIR)):
            if os.path.isfile(os.path.join(SRC_DIR, filename)) and wanted(filename):
                found.append((SRC_DIR, filename))
    return found


def output_folder(current_dir):
    """Where a file from *current_dir* is written, honouring KEEP_TREE."""
    if KEEP_TREE:
        relative = os.path.relpath(current_dir, SRC_DIR)
        target = DST_DIR if relative == "." else os.path.join(DST_DIR, relative)
    else:
        target = DST_DIR
    os.makedirs(target, exist_ok=True)
    return target


# ---------------------------------------------------------------------------
# 3. Opening and saving
# ---------------------------------------------------------------------------
# The published mfv API has no file I/O yet, so these two reach into the core
# modules. Keep them together: when mfv gains an io namespace, only this
# section changes.
def open_dataset(path):
    """Load one file into a dataset, without adding it to the viewer."""
    from minflux_viewer.core.particle_extract import load_any_dataset

    if path.lower().endswith((".zarr", ".zarr.zip")):
        from minflux_viewer.core.loader import load_zarr

        return load_zarr(path)
    # Handles .mat, .npy, .csv, .tsv, .txt and .json.
    #
    # NOTE: .msr is deliberately not opened here. One .msr can hold several
    # acquisitions, so it is not a single-dataset file. To batch-convert .msr,
    # use the MSR Reader's own Folder (batch) mode and point this script at
    # what it exported.
    return load_any_dataset(path)


def save_dataset(ds, folder, filename):
    """Write *ds* into *folder*. Change fmt/content to suit your workflow."""
    from minflux_viewer.core.save import save_processed

    stem = os.path.splitext(filename)[0]
    out_path = os.path.join(folder, stem + ".mat")
    save_processed(
        ds,
        data_path=out_path,
        fmt="mat",              # zarr, msr, npy, mat, json, csv
        content="snapshot",     # "snapshot" bakes filters + Z scaling; "raw" does not
        filter_mode="apply",    # "apply" drops filtered rows; "flag" keeps them
    )
    return out_path


# ---------------------------------------------------------------------------
# 4. Per-file processing
# ---------------------------------------------------------------------------
def process_one(src_folder, filename, dst_folder):
    """Open one file, process it, save the result."""
    src_path = os.path.join(src_folder, filename)
    mfv.ui.log("Opening " + filename)
    ds = open_dataset(src_path)

    # =======================================================================
    # ===                 YOUR PROCESSING GOES HERE                       ===
    # =======================================================================
    #
    # `ds` is the dataset just opened. A few things you can do with it:
    #
    #     n    = ds.prop.num_loc                        # localization count
    #     xyz  = mfv.data.loc(dataset=ds)               # (N, 3) nm coordinates
    #     efo  = mfv.data.attr("efo", dataset=ds)       # any attribute
    #     mfv.data.set_filter("cfr", hi=0.8, dataset=ds)
    #     mfv.data.add_attr("my_value", values, dataset=ds)
    #
    # To collect one row of numbers per file instead of writing datasets, add
    # to the results table created below the loop and delete the save call:
    #
    #     table.add_row(file=filename, n_locs=ds.prop.num_loc)
    #
    # =======================================================================

    out_path = save_dataset(ds, dst_folder, filename)
    mfv.ui.log("Saved " + os.path.basename(out_path))


# ---------------------------------------------------------------------------
# 5. The run
# ---------------------------------------------------------------------------
files = find_files()
if not files:
    mfv.ui.warn(
        f"No files matched under:\n{SRC_DIR}\n\n"
        f"extension {EXT!r}, include {INCLUDE or '(any)'}, "
        f"exclude {EXCLUDE or '(none)'}"
    )
    raise SystemExit

# One row per file, if you want measurements instead of (or besides) files.
table = mfv.results.table("Batch")

mfv.ui.log(f"Process Folder: {len(files)} file(s) from {SRC_DIR}")
failed = []
for index, (src_folder, filename) in enumerate(files, start=1):
    mfv.ui.status(
        f"Processing {index}/{len(files)}: {filename}",
        index / len(files),
    )
    try:
        process_one(src_folder, filename, output_folder(src_folder))
    except Exception as exc:          # one bad file must not stop the batch
        failed.append((filename, str(exc)))
        mfv.ui.log(f"FAILED {filename}: {exc}", "WARN")

mfv.ui.status("Process Folder: done", 1.0)
if len(table):
    table.show()
mfv.ui.log(
    f"Process Folder finished: {len(files) - len(failed)} done, "
    f"{len(failed)} failed."
)
for filename, message in failed:
    mfv.ui.log(f"  failed: {filename} ({message})", "WARN")
