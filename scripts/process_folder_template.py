"""
Process Folder — CPU-only background batch template for the Script Editor.

Each input gets an owned output directory containing data, processing recipe
and a hash-verified receipt. Existing outputs are never replaced. Resume checks
the versioned operation, parameters, input and output bytes. Change OPERATION
whenever you change process_one(). See the scripting reference for limits.
"""
# `mfv` is injected by the Script Editor.
# ruff: noqa: F821

import os
from pathlib import Path

OPERATION = "process-folder-example@1"

params = mfv.ui.ask(
    {
        "Input directory": "",
        "Output directory": "",
        "File extension": ".mat",
        "File name include": "",
        "File name exclude": "",
        "Search subfolders": True,
        "Resume verified outputs": True,
        "Minimum EFO (Hz)": 0.0,
    },
    title="Process Folder",
    descriptions={
        "Input directory": "Blank opens a folder picker.",
        "Output directory": "A separate output tree. Never overwrite input files.",
        "File extension": "Single-dataset files only. MSR/projects need explicit channel selection.",
        "File name include": "Comma-separated: match any token. Blank keeps all.",
        "File name exclude": "Comma-separated: skip any token.",
        "Search subfolders": "Walk subfolders without following directory links.",
        "Resume verified outputs": "Reuse only matching receipts and unchanged file hashes.",
        "Minimum EFO (Hz)": "Example filter; zero leaves the data unchanged.",
    },
)
if params is None:
    raise SystemExit("Cancelled.")


def folder(value, caption):
    value = str(value).strip().strip('"')
    if not value:
        value = mfv.ui.choose_dir(caption=caption)
    if not value:
        raise SystemExit("Cancelled.")
    return Path(value).resolve()


SRC = folder(params["Input directory"], "Input directory")
DST = folder(params["Output directory"], "Output directory")
EXT = str(params["File extension"]).strip().lower()
INCLUDE = tuple(t.strip().lower() for t in params["File name include"].split(",") if t.strip())
EXCLUDE = tuple(t.strip().lower() for t in params["File name exclude"].split(",") if t.strip())
RECURSIVE = bool(params["Search subfolders"])
RESUME = bool(params["Resume verified outputs"])
PROCESSING = {"minimum_efo_hz": float(params["Minimum EFO (Hz)"])}


def find_files(task):
    """Run discovery off-thread; exclude the output tree and sibling recipes."""
    found = []
    for root, directories, filenames in os.walk(SRC, followlinks=False):
        task.check_cancelled()
        directories[:] = sorted(
            name for name in directories
            if (Path(root) / name).resolve() != DST and not (Path(root) / name).is_symlink()
        )
        for filename in sorted(filenames):
            name = filename.lower()
            if EXT and not name.endswith(EXT):
                continue
            if name.endswith("_viewer_metadata.json"):
                continue
            if INCLUDE and not any(token in name for token in INCLUDE):
                continue
            if EXCLUDE and any(token in name for token in EXCLUDE):
                continue
            found.append(Path(root) / filename)
        if not RECURSIVE:
            break
    return found


def process_one(ds, parameters, check_cancelled):
    """YOUR PROCESSING GOES HERE: detached data only, no mfv.data/view/ui calls."""
    import numpy as np

    check_cancelled()
    minimum = parameters["minimum_efo_hz"]
    if minimum > 0:
        efo = ds.attr.get("efo")
        if efo is None:
            raise ValueError("This dataset has no EFO column for the requested filter")
        ds.filter_mask = ds.filter_mask & (np.asarray(efo) >= minimum)
    # Modify ds or return a replacement detached dataset. Call check_cancelled()
    # between expensive steps. Never access widgets or a live viewer dataset.


def work(task):
    files = find_files(task)
    if not files:
        raise ValueError("No input files matched the requested filters")
    return mfv.io.batch(
        files, DST, process_one,
        operation=OPERATION, parameters=PROCESSING,
        fmt="npy", content="snapshot", filter_mode="apply",
        apply_sidecar=True, resume=RESUME,
        check_cancelled=task.check_cancelled, progress=task.progress,
    )


def done(rows):
    table = mfv.results.table("Batch")
    table.add_rows(rows)
    table.show()
    mfv.ui.status("Process Folder: complete", 1.0)
    mfv.ui.log(f"Process Folder: {len(rows)} verified output(s).")


mfv.run.background(
    work, on_done=done, name="Process Folder",
    on_error=lambda error: mfv.ui.error(
        f"Batch stopped: {error}\nEarlier committed items remain safe; rerun with resume enabled."
    ),
    on_cancel=lambda: mfv.ui.log("Batch stopped; committed items remain resumable."),
)
