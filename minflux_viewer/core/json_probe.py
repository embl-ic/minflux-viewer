"""Bounded, string-aware look at a ``.json`` file -- enough to classify it.

Several kinds of JSON share the extension: localization data (often hundreds
of megabytes), a processing sidecar, a ROI set, a filter preset, an MBM bead
table, a MINFLUX acquisition sequence. Classifying one must not decode a
multi-gigabyte localization array; it used to, twice, before the loader decoded
it a third time.

:func:`probe_json` reads at most :data:`PREFIX_BYTES` -- the **fast path** --
and tokenizes them with string awareness (``json.decoder.scanstring``, escapes,
nesting depth), so a ``"}"`` inside a string cannot end a record early. It
reports the top-level type, an object's top-level keys **with the kind of each
value** (so "``rois`` is a list" needs no decoding), or an array's first-record
keys, and whether the whole document fitted.

A prefix that cannot decide -- whitespace only, a first record longer than the
prefix, an object whose deciding keys may still follow -- is **inconclusive**
(:attr:`JsonProbe.conclusive` False), never "data" and never "invalid".
:func:`deep_probe_json` then reads in growing steps until the question is
decided, still without building the document; the application runs it off the
UI thread. Only a document that fits the fast-path budget is ever decoded.

**Precedence that a bounded read can decide.** Kinds are told apart by top-level
keys, and a key can follow a gigabyte-long record array; any rule in which such
a late key changes the answer would make the answer depend on the file's size.
So, in this order:

1. the sidecar marker as the **first** key (where the writer always puts it):
   a processing sidecar;
2. a localization record array under a wrapper key (``RECORD_WRAPPER_KEYS``,
   first element a localization record): **data**, whatever else the object
   holds -- no annotation kind holds localizations;
3. everything else by the predicates' own order (sidecar marker elsewhere,
   ROI set, filter preset, MBM table, sequence), and data last.

Rules 1 and 2 are decided where they occur, so a prefix that reaches them is
conclusive; rule 3 needs the whole object.

**The one case where key order matters.** An object holding both the sidecar
marker and a localization array under a wrapper key is a sidecar when the
marker comes first (rule 1) and data when the array comes first (rule 2).
Re-serialising such an object with reordered keys -- ``json.dumps(...,
sort_keys=True)`` puts ``data`` before the marker -- therefore changes its
kind. Our writer never produces such an object and always writes the marker
first. Every other kind, including a sidecar without a localization array,
is decided the same in any key order.

A probe is meant to live for one routing decision: no global cache, so a file
changed on disk between two opens is always read afresh.

Qt-free and pure.
"""

from __future__ import annotations

import json
import json.decoder
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: The JSON string scanner (C-accelerated): ``scanstring(text, end) -> (str, end)``
#: where *end* is just past the opening quote. Public but absent from the stubs.
scanstring: Callable[[str, int], tuple[str, int]] = json.decoder.scanstring  # type: ignore[attr-defined]

__all__ = ["PREFIX_BYTES", "RECORD_WRAPPER_KEYS", "SIDECAR_MARKER", "JsonProbe",
           "probe_json", "deep_probe_json", "as_probe", "looks_like_localization"]

#: Bytes read on the fast path, and the largest document ever decoded while
#: classifying. Every sidecar, filter preset and sequence seen so far fits; a
#: localization array is decided by its first record.
PREFIX_BYTES = 64 * 1024

#: Keys under which a JSON object may wrap a localization record array. Once
#: such an array's first element is read and is a localization record, the
#: object is data (precedence rule 2), so classifying it never needs the
#: (large) rest of the file.
RECORD_WRAPPER_KEYS = ("mfx", "records", "localizations", "localisations", "data")

#: The processing sidecar's own marker (``core.save.METADATA_JSON_MARKER``),
#: the one kind that names itself; repeated here so the probe stays import-free.
SIDECAR_MARKER = "minflux_viewer_metadata"

_WHITESPACE = " \t\r\n"
_UNDECODED = object()


def looks_like_localization(keys) -> bool:
    """Whether a record with these keys is a MINFLUX localization: a
    ``loc``/``lnc``/``ext`` vector or an x/y coordinate pair (metres or nm).
    The JSON loader accepts exactly these."""
    names = set(keys)
    return bool({"loc", "lnc", "ext"} & names
                or {"x", "y"} <= names or {"xnm", "ynm"} <= names
                or {"x_nm", "y_nm"} <= names or {"loc_x", "loc_y"} <= names
                or {"loc_x_nm", "loc_y_nm"} <= names)


class JsonProbe:
    """What a bounded read of one ``.json`` file says about it."""

    def __init__(self, path: Path, *, top: str, keys: tuple[str, ...] = (),
                 kinds: dict[str, str] | None = None, complete: bool,
                 first_closed: bool = False, size: int,
                 wrapped: dict[str, tuple[str, ...] | None] | None = None,
                 document: Any = _UNDECODED) -> None:
        self.path = path
        #: ``"array"``, ``"object"``, ``"invalid"``, ``"empty"`` (nothing but
        #: whitespace in the whole file) or ``"undetermined"`` (nothing but
        #: whitespace in what was read).
        self.top = top
        #: An object's top-level keys, in order, as far as the read went; or
        #: an array's first-record keys (empty if that record did not close).
        self.keys = keys
        #: For an object: the kind of each top-level key's value
        #: (``object``/``array``/``string``/``number``/``literal``).
        self.kinds = dict(kinds or {})
        #: The whole file was read, so :attr:`keys` is final.
        self.complete = complete
        #: For an array: its first element ended inside what was read (or the
        #: array is empty, or the first element is not an object).
        self.first_closed = first_closed
        self.size = size
        #: For an object: each wrapper-key array seen, with its first element's
        #: keys (``()`` for an empty array or one of non-objects, ``None`` when
        #: the read ended before that element closed).
        self.wrapped = dict(wrapped or {})
        self._document = document

    @property
    def marker_first(self) -> bool:
        """The sidecar marker is the object's first key (precedence rule 1)."""
        return self.top == "object" and bool(self.keys) and self.keys[0] == SIDECAR_MARKER

    @property
    def wraps_localizations(self) -> bool | None:
        """The object holds a localization record array under a wrapper key
        (precedence rule 2: data). None when a wrapper array's first element
        was not read completely and no other one decides."""
        unknown = False
        for first_keys in self.wrapped.values():
            if first_keys is None:
                unknown = True
            elif looks_like_localization(first_keys):
                return True
        return None if unknown else False

    @property
    def conclusive(self) -> bool:
        """Whether every ``.json`` kind can be decided from this probe."""
        if self.top in ("invalid", "empty"):
            return True
        if self.top == "array":
            return self.complete or self.first_closed
        if self.top == "object":
            return self.complete or self.marker_first or self.wraps_localizations is True
        return False                          # "undetermined"

    def has_key(self, key: str) -> bool | None:
        """Whether an object has top-level *key*: True, False, or None when
        the read ended before every key was seen."""
        if self.top != "object":
            return False
        if key in self.keys:
            return True
        return False if self.complete else None

    def kind(self, key: str) -> str | None:
        """The kind of an object's top-level *key* value, if it was seen."""
        return self.kinds.get(key)

    @property
    def decoded(self) -> bool:
        """The document was small enough to be decoded while probing."""
        return self._document is not _UNDECODED

    def document(self) -> Any:
        """The decoded document; decoded at most once per probe.

        Raises ``ValueError`` for text that is not valid JSON. Callers that
        must stay bounded check :attr:`decoded` first.
        """
        if self._document is _UNDECODED:
            with open(self.path, "rb") as handle:
                self._document = json.loads(handle.read().decode("utf-8-sig"))
        return self._document

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"JsonProbe({self.path.name!r}, top={self.top!r}, keys={self.keys!r}, "
                f"complete={self.complete}, conclusive={self.conclusive})")


def as_probe(source) -> JsonProbe:
    """*source* itself when it already is a probe, else a fast-path probe of it."""
    return source if isinstance(source, JsonProbe) else probe_json(source)


def deep_probe_json(path) -> JsonProbe:
    """Read further, in growing steps, until the file's kind is decided.

    Never builds the document of a file larger than the fast-path budget; at
    worst it scans the whole text. Meant to run off the UI thread.
    """
    budget = PREFIX_BYTES * 4
    while True:
        probe = probe_json(path, budget=budget)
        if probe.conclusive or probe.complete:
            return probe
        budget *= 4


def probe_json(path, *, budget: int | None = PREFIX_BYTES) -> JsonProbe:
    """Read at most *budget* bytes of *path* (all of it for None) and classify
    its shape. ``OSError`` propagates; anything not JSON-shaped is
    ``top="invalid"``."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as handle:
        head = handle.read() if budget is None else handle.read(budget)
    complete = size <= len(head)
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        if complete:
            return JsonProbe(path, top="invalid", complete=True, size=size)
        # The read may cut a multi-byte character: drop the partial tail.
        text = head.decode("utf-8-sig", errors="ignore")
    if complete and len(head) <= PREFIX_BYTES:
        return _probe_document(path, text, size)
    start = _skip_ws(text, 0)
    if start >= len(text):
        top = "empty" if complete else "undetermined"
        return JsonProbe(path, top=top, complete=complete, size=size)
    if text[start] == "{":
        wrapped: dict[str, tuple[str, ...] | None] = {}
        keys, kinds, _closed = _object_keys(text, start, wrapped=wrapped)
        return JsonProbe(path, top="object", keys=keys, kinds=kinds, complete=complete,
                         size=size, wrapped=wrapped)
    if text[start] == "[":
        element = _skip_ws(text, start + 1)
        if element >= len(text):
            return JsonProbe(path, top="array", complete=complete, size=size)
        if text[element] != "{":
            # An empty array, or rows that are not objects: no record kind.
            return JsonProbe(path, top="array", complete=complete, first_closed=True,
                             size=size)
        found, _kinds, closed = _object_keys(text, element)
        return JsonProbe(path, top="array", keys=found if closed else (),
                         complete=complete, first_closed=closed, size=size)
    return JsonProbe(path, top="invalid", complete=complete, size=size)


def _probe_document(path: Path, text: str, size: int) -> JsonProbe:
    """A whole small document: decode it once and keep it."""
    if not text.strip():
        return JsonProbe(path, top="empty", complete=True, size=size)
    try:
        document = json.loads(text)
    except ValueError:
        return JsonProbe(path, top="invalid", complete=True, size=size)
    if isinstance(document, dict):
        wrapped: dict[str, tuple[str, ...] | None] = {
            str(k): (tuple(map(str, v[0])) if v and isinstance(v[0], dict) else ())
            for k, v in document.items()
            if k in RECORD_WRAPPER_KEYS and isinstance(v, list)}
        return JsonProbe(path, top="object", keys=tuple(map(str, document)),
                         kinds={str(k): _kind_of(v) for k, v in document.items()},
                         complete=True, size=size, wrapped=wrapped, document=document)
    if isinstance(document, list):
        first_record = document[0] if document and isinstance(document[0], dict) else {}
        return JsonProbe(path, top="array", keys=tuple(map(str, first_record)),
                         complete=True, first_closed=True, size=size, document=document)
    return JsonProbe(path, top="invalid", complete=True, size=size)


def _kind_of(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool) or value is None:
        return "literal"
    return "number"


def _kind_at(text: str, i: int) -> str:
    char = text[i] if i < len(text) else ""
    return {"{": "object", "[": "array", '"': "string"}.get(
        char, "literal" if char in "tfn" else "number")


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WHITESPACE:
        i += 1
    return i


def _object_keys(text: str, start: int, *,
                 wrapped: dict[str, tuple[str, ...] | None] | None = None,
                 ) -> tuple[tuple[str, ...], dict[str, str], bool]:
    """Top-level keys (and value kinds) of the object at ``text[start] == "{"``.

    Returns ``(keys, kinds, closed)``: ``closed`` is True when the object's
    closing brace was reached inside *text*. Stops early (closed False) at the
    end of the text or at anything malformed. A key is recorded with its
    value's kind as soon as the value starts, before the value is skipped.
    With *wrapped*, each wrapper-key array's first-element keys are recorded
    there too (see :attr:`JsonProbe.wrapped`).
    """
    keys: list[str] = []
    kinds: dict[str, str] = {}
    i = _skip_ws(text, start + 1)
    if i < len(text) and text[i] == "}":
        return (), kinds, True
    while i < len(text):
        if text[i] != '"':
            break
        try:
            key, i = scanstring(text, i + 1)
        except ValueError:                       # truncated or malformed string
            break
        i = _skip_ws(text, i)
        if i >= len(text) or text[i] != ":":
            break
        value_start = _skip_ws(text, i + 1)
        if value_start >= len(text):
            break
        keys.append(key)
        kinds[key] = _kind_at(text, value_start)
        if wrapped is not None and key in RECORD_WRAPPER_KEYS and kinds[key] == "array":
            element = _skip_ws(text, value_start + 1)
            if element >= len(text):
                wrapped[key] = None
            elif text[element] == "{":
                record_keys, _kinds, closed = _object_keys(text, element)
                wrapped[key] = record_keys if closed else None
            else:
                wrapped[key] = ()             # empty, or elements that are not objects
        end = _skip_value(text, value_start)
        if end is None:
            break
        i = _skip_ws(text, end)
        if i >= len(text):
            break
        if text[i] == "}":
            return tuple(keys), kinds, True
        if text[i] != ",":
            break
        i = _skip_ws(text, i + 1)
    return tuple(keys), kinds, False


def _skip_value(text: str, i: int) -> int | None:
    """Index just past the JSON value starting at *i*, or None if the text
    ends first (or the value is malformed)."""
    if i >= len(text):
        return None
    char = text[i]
    if char == '"':
        try:
            return scanstring(text, i + 1)[1]
        except ValueError:
            return None
    if char in "{[":
        depth = 0
        while i < len(text):
            char = text[i]
            if char == '"':
                try:
                    i = scanstring(text, i + 1)[1]
                except ValueError:
                    return None
                continue
            if char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None
    # A number or a literal (true / false / null): it ends at a delimiter. If
    # the text ends first, the value may be cut, so it is undetermined.
    while i < len(text) and text[i] not in ",}] \t\r\n":
        i += 1
    return i if i < len(text) else None
