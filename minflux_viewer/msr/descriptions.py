from __future__ import annotations

from typing import Any


#: The one group in a .msr that is ours rather than Abberior's.
VIEWER_PATH = "viewer"

PATH_DESCRIPTIONS = {
    "mfx": {
        "dtype": "Structured MINFLUX raw localization table",
        "description": "Raw MINFLUX measurement array. This is the main structured localization table exported by iMSPECTOR.",
    },
    "grd/mbm/points": {
        "dtype": "Structured MBM point table",
        "description": "Reference bead or point-of-reference table used for active drift correction (MBM).",
    },
    "grd/search_0/points": {
        "dtype": "Structured grid-search point table",
        "description": "Grid-search point table describing the searched grid positions used during MINFLUX search grid acquisition.",
    },
    # Written by THIS application, not by Imspector -- a .msr we saved carries
    # its processing state here so the file is a complete viewer document as
    # well as a measurement. Described so it does not read as an unknown
    # Abberior parameter in the reader tree.
    "viewer": {
        "dtype": "MINFLUX Viewer processing state",
        "description": (
            "MINFLUX Viewer processing state, written by this application when "
            "the file was saved, not part of the Abberior format. Holds this "
            "channel's view and processing settings (filters, ROIs, overlay "
            "membership and LUT, transform, Z scaling factor) plus the manifest "
            "of the channels it was saved with, so re-opening restores the "
            "session. Imspector ignores it."
        ),
    },
    "grd/mbm/points/gri": {"description": "Grid reference."},
    "grd/mbm/points/xyz": {"description": "Bead position in metres."},
    "grd/mbm/points/tim": {"description": "Time when bead position was measured in seconds."},
    "grd/mbm/points/str": {"description": "Strength."},
    "grd/search_0/points/gri": {"description": "Grid reference."},
    "grd/search_0/points/xyz": {"description": "Bead position in metres."},
    "grd/search_0/points/tim": {"description": "Time when bead position was measured in seconds."},
    "grd/search_0/points/str": {"description": "Strength."},
    "grd": {
        "description": "Grid-related data. According to the Abberior MINFLUX Zarr layout, this contains search grid positions and MBM measurements.",
    },
    "grd/mbm": {
        "description": "MBM subgroup with information about the points of reference used for drift correction.",
    },
    "mbm": {
        "description": "Information about which points of reference (POR) were used to perform the active drift correction.",
    },
    "grd/search_0": {
        "description": "Search-grid subgroup containing the searched grid positions for the first search grid.",
    },
}


FIELD_DESCRIPTIONS = {
    "gri": "Grid reference.",
    "xyz": "Bead position in metres.",
    "tim": "Time stamp in seconds.",
    "str": "Strength.",
    "tid": "Trace ID. Each burst or molecule track gets a unique ID.",
    "itr": "Iteration index. This is the localization step defined in the MINFLUX sequence that iteratively improves localization precision.",
    "loc": "Calculated fluorophore coordinates in metres for each iteration. These localizations may include applied corrections.",
    "vld": "Valid localization flag. True means a valid localization was obtained.",
    "lnc": "Final localization coordinates without applied corrections. When MBM is active, this stores the raw uncorrected coordinates.",
    "sqi": "Sequence index.",
    "sta": "State or abort-status code.",
    "thi": "Thread value.",
    "bot": "Boolean flag marking the beginning of a trace.",
    "eot": "Boolean flag marking the end of a trace.",
    "fnl": "Boolean flag indicating a final event.",
    "eco": "Effective counts offset: background-corrected photons counted on the outer EOD pattern positions.",
    "ecc": "Effective counts center: background-corrected photons counted at the center EOD position.",
    "efo": "Effective frequency offset: fluorophore emission frequency on the outer EOD pattern positions.",
    "efc": "Effective frequency center: fluorophore emission frequency at the center EOD position.",
    "fbg": "Estimated background emission frequency, if background correction sensing was enabled.",
    "cfr": "Center frequency ratio computed as efc/efo.",
    "dcr": "Detection channel ratio used for multicolor imaging, computed as detector 1 photon count divided by detector 1 plus detector 2 photon count.",
    "act": "Activation-laser active flag.",
    "dmz": "Deformable mirror z-position.",
    "dos": "Digital outputs value.",
    "eox": "EOD x-position in the field of view, relative to the galvo center.",
    "eoy": "EOD y-position in the field of view, relative to the galvo center.",
    "ext": "Physical extension of the EOD pattern diameter in the field of view.",
    "gvx": "Galvo x-position in the field of view.",
    "gvy": "Galvo y-position in the field of view.",
    "lcx": "Localization x-coordinate relative to the center of the EOD pattern.",
    "lcy": "Localization y-coordinate relative to the center of the EOD pattern.",
    "lcz": "Localization z-coordinate relative to the center of the EOD pattern.",
    "sky": "Retry counter for repeated localization attempts within an iteration.",
    "tic": "FPGA ticks at 40 MHz.",
    "measurement": "Measurement metadata subtree from Zarr attributes.",
    "threads": "Measurement thread definitions stored in metadata.",
    "sequences": "MINFLUX sequence definitions stored in metadata.",
}


def describe_path(path: str, *, is_array: bool = False, dtype: Any = None) -> str:
    entry = PATH_DESCRIPTIONS.get(path)
    if entry and entry.get("description"):
        return entry["description"]
    leaf = path.rsplit("/", 1)[-1]
    field_desc = FIELD_DESCRIPTIONS.get(leaf)
    if field_desc:
        return field_desc
    if path == VIEWER_PATH or path.startswith(VIEWER_PATH + "/"):
        # Anything under our own group is ours, whatever it is called; pointing
        # the user at Abberior documentation for it would be actively wrong.
        return PATH_DESCRIPTIONS[VIEWER_PATH]["description"]
    if path:
        return "Unknown parameter, consider checking Abberior documentation."
    if is_array:
        return "Unknown array node, consider checking Abberior documentation."
    return "Unknown zarr node, consider checking Abberior documentation."


def describe_dtype(path: str, dtype: Any, length: int | None = None) -> str:
    entry = PATH_DESCRIPTIONS.get(path)
    if entry and entry.get("dtype"):
        if length is not None:
            return f"{entry['dtype']} ({length} rows)"
        return entry["dtype"]
    names = getattr(dtype, "names", None)
    if names:
        preview = ", ".join(str(n) for n in list(names)[:6])
        suffix = "" if len(names) <= 6 else ", ..."
        return f"structured fields: {preview}{suffix}"
    return str(dtype)
