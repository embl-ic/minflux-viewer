"""Qt-free macro-recording primitives.

The processing journal remains the single event stream.  :class:`Recorder`
only marks a recording boundary and appends structured journal entries while
recording is enabled; :func:`emit_script` is the second renderer for those
entries (the method-text generator is the first).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class GuiClass(str, Enum):
    """Whether replaying a recorded step needs a graphical interaction."""

    GUI_FREE = "gui_free"
    GUI_RESULT = "gui_result"
    GUI_ONLY = "gui_only"

    @classmethod
    def coerce(cls, value: GuiClass | str | None) -> GuiClass | None:
        """Normalize persisted/user-facing spellings without guessing."""
        if value is None or isinstance(value, cls):
            return value
        text = str(value).strip()
        if not text:
            return None
        lowered = text.lower()
        for member in cls:
            if lowered in {member.value, member.name.lower()}:
                return member
        choices = ", ".join(member.name for member in cls)
        raise ValueError(f"Unknown GUI class {value!r}; expected one of: {choices}.")


#: Journal categories that describe work a macro would have to reproduce.
#:
#: An entry in one of these surfaces in the recording even when nothing has
#: instrumented it yet -- as a TODO naming what happened. That is deliberate and
#: it is the difference between a recorder a user can act on and one that looks
#: broken: loading a dataset, filtering it and running an analysis are exactly
#: the things somebody records, and showing nothing at all for them is the worst
#: possible feedback. ``other`` is the chatter category and stays out.
MACRO_CATEGORIES = frozenset(
    {"load", "filter", "transform", "analysis", "export", "plugin"}
)


@dataclass(frozen=True)
class RecordedStep:
    """One structured, script-renderable event from the processing journal."""

    timestamp: str = ""
    category: str = "other"
    summary: str = ""
    code: str | None = None
    gui_class: GuiClass | None = None
    command: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", str(self.timestamp or ""))
        object.__setattr__(self, "category", str(self.category or "other"))
        object.__setattr__(self, "summary", str(self.summary or ""))
        code = str(self.code).strip() if self.code is not None else ""
        command = str(self.command).strip() if self.command is not None else ""
        object.__setattr__(self, "code", code or None)
        object.__setattr__(self, "command", command or None)
        object.__setattr__(self, "gui_class", GuiClass.coerce(self.gui_class))
        object.__setattr__(self, "details", dict(self.details or {}))

    @classmethod
    def from_entry(cls, entry: Any) -> RecordedStep:
        """Copy the structured portion of a journal entry."""
        return cls(
            timestamp=getattr(entry, "timestamp", ""),
            category=getattr(entry, "category", "other"),
            summary=getattr(entry, "summary", ""),
            code=getattr(entry, "code", None),
            gui_class=getattr(entry, "gui_class", None),
            command=getattr(entry, "command", None),
            details=getattr(entry, "details", {}),
        )

    @property
    def structured(self) -> bool:
        """Whether this entry was deliberately emitted for macro recording."""
        return bool(self.code or self.command or self.gui_class is not None)


def _as_step(value: RecordedStep | Any) -> RecordedStep:
    return value if isinstance(value, RecordedStep) else RecordedStep.from_entry(value)


def _deduplicate(steps: Iterable[RecordedStep]) -> list[RecordedStep]:
    """Collapse instrumentation duplicates without removing data operations.

    A command hook may first record only the QAction identity and a command
    handler may then record the runnable call.  Consecutive entries for that
    same command are coalesced in favour of the runnable one.  Exact repeated
    GUI-only or identity-only events are also harmless presentation duplicates.
    Executable GUI_FREE/GUI_RESULT calls are deliberately preserved: running
    the same analysis twice can be meaningful.
    """
    out: list[RecordedStep] = []
    for step in steps:
        if not out:
            out.append(step)
            continue
        previous = out[-1]
        same_command = bool(step.command and step.command == previous.command)
        # A step that declares itself the newer state of the same command
        # replaces its predecessor: tuning a filter re-applies on every edit,
        # and a macro wants the setting the user settled on.
        if same_command and step.details.get("supersedes_previous"):
            out[-1] = step
            continue
        if same_command and bool(step.code) != bool(previous.code):
            if step.code:
                out[-1] = step
            continue
        same_event = (
            same_command
            and step.code == previous.code
            and step.gui_class == previous.gui_class
            and step.summary == previous.summary
            and dict(step.details) == dict(previous.details)
        )
        if same_event and (not step.code or step.gui_class is GuiClass.GUI_ONLY):
            continue
        out.append(step)
    return out


def _todo_line(step: RecordedStep) -> str:
    label = " ".join(step.summary.split()) or "An undescribed command"
    # Do not repeat the identity when the summary already is it (a self-recorded
    # API call is summarised by its own name).
    identity = (
        f" ({step.command})"
        if step.command and step.command != step.summary
        else ""
    )
    # A caller that knows *why* the call could not be written down says so; that
    # is the difference between a gap somebody can fix and a mystery.
    explicit = str(step.details.get("unrecordable", "")).strip()
    if explicit:
        reason = explicit
    elif not step.command and not step.gui_class:
        reason = (
            f"this {step.category} step is not instrumented for recording yet"
        )
    elif step.gui_class is None:
        reason = "its GUI class and runnable mfv call are not declared yet"
    else:
        reason = "no runnable mfv call was recorded"
    return f"# TODO {label}{identity} ran here, but {reason}."


def _default_header() -> str:
    from minflux_viewer import __version__
    from minflux_viewer.api import __api_version__

    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    return (
        f"# Recorded with MINFLUX Viewer {__version__} · mfv API "
        f"{__api_version__} · {stamp}\n"
        "# GUI_RESULT values reproduce accepted outcomes; review TODOs before reuse."
    )


def emit_script(
    steps: Iterable[RecordedStep | Any],
    *,
    silent: bool = False,
    header: bool | str = True,
) -> str:
    """Render structured journal entries as a runnable ``mfv`` Python script.

    Silent mode drops only steps explicitly classified as :attr:`GUI_ONLY`.
    Unclassified commands remain visible as TODOs so an incomplete recording
    can never masquerade as a complete silent workflow -- and so does any
    journal entry in a :data:`MACRO_CATEGORIES` category, because a recorder
    that shows nothing for a dataset the user just loaded reads as broken.
    """
    normalized = [
        step for step in map(_as_step, steps)
        if step.structured or step.category in MACRO_CATEGORIES
    ]
    normalized = _deduplicate(normalized)
    if silent:
        normalized = [
            step for step in normalized if step.gui_class is not GuiClass.GUI_ONLY
        ]

    lines: list[str] = []
    if header:
        heading = _default_header() if header is True else str(header).strip()
        if heading:
            for line in heading.splitlines():
                lines.append(line if line.lstrip().startswith("#") else f"# {line}")
    lines.append("import mfv")

    for step in normalized:
        body = step.code or _todo_line(step)
        lines.extend(("", *body.rstrip().splitlines()))
    return "\n".join(lines).rstrip() + "\n"


class Recorder:
    """A recording cursor over the shared :class:`ProcessingJournal`."""

    def __init__(self, journal: Any | None = None) -> None:
        if journal is None:
            # Lazy import keeps ``processing_journal`` free to type its entries
            # with ``GuiClass`` without creating an import cycle.
            from .processing_journal import ProcessingJournal

            journal = ProcessingJournal()
        self._journal = journal
        self._enabled = False
        self._cursor = len(journal)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self, *, clear: bool = True) -> None:
        """Start recording; by default begin a fresh visible recording."""
        if clear:
            self.clear()
        self._enabled = True

    def stop(self) -> None:
        """Pause recording without discarding the captured steps."""
        self._enabled = False

    pause = stop

    def clear(self) -> None:
        """Forget the visible recording without deleting journal history."""
        self._cursor = len(self._journal)

    def append(
        self,
        summary: str,
        *,
        code: str | None = None,
        gui_class: GuiClass | str | None = None,
        command: str | None = None,
        category: str = "other",
        **details: Any,
    ) -> Any | None:
        """Append one structured journal event while recording is enabled."""
        if not self._enabled:
            return None
        return self._journal.add(
            category,
            summary,
            code=code,
            gui_class=gui_class,
            command=command,
            **details,
        )

    def record_command(
        self,
        command: str,
        *,
        summary: str,
        code: str | None = None,
        gui_class: GuiClass | str | None = None,
        **details: Any,
    ) -> Any | None:
        """Record a QAction identity, with runnable code when it is known."""
        return self.append(
            summary,
            code=code,
            gui_class=gui_class,
            command=command,
            category="other",
            **details,
        )

    def steps(self) -> list[RecordedStep]:
        entries = self._journal.entries
        start = self._cursor if self._cursor <= len(entries) else 0
        return [
            step
            for step in (RecordedStep.from_entry(entry) for entry in entries[start:])
            if step.structured or step.category in MACRO_CATEGORIES
        ]

    def script(self, *, silent: bool = False, header: bool | str = True) -> str:
        return emit_script(self.steps(), silent=silent, header=header)


__all__ = ["GuiClass", "RecordedStep", "Recorder", "emit_script"]
