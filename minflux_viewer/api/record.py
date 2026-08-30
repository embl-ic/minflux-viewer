"""``mfv.record`` -- control and extend the session macro recorder.

The recorder is a view over the shared processing journal, not a second event
stream.  Scripts and plugins may therefore contribute reproducible code while
the recorder is active without depending on any Qt or main-window internals.
"""

from __future__ import annotations

from typing import Any

from ..core.recorder import GuiClass, RecordedStep
from ._base import ApiError, Namespace


class Record(Namespace):
    """Record viewer work as a runnable ``mfv`` Python script."""

    name = "record"

    def start(self, *, clear: bool = True) -> None:
        """Start recording, beginning a fresh recording by default."""
        self._state.recorder.start(clear=bool(clear))

    def stop(self) -> None:
        """Pause recording without discarding captured steps."""
        self._state.recorder.stop()

    def steps(self) -> list[RecordedStep]:
        """Return immutable snapshots of the currently captured steps."""
        return self._state.recorder.steps()

    def script(self, *, silent: bool = False, header: bool = True) -> str:
        """Render the current recording as Python.

        With ``silent=True``, presentation-only steps are omitted.  Unknown
        commands remain visible as ``TODO`` comments so gaps cannot masquerade
        as a complete automated workflow.
        """
        return self._state.recorder.script(silent=bool(silent), header=bool(header))

    def step(
        self,
        code: str,
        gui_class: GuiClass | str = GuiClass.GUI_FREE,
        *,
        summary: str = "",
        category: str = "plugin",
        command: str | None = None,
        **details: Any,
    ) -> None:
        """Contribute one exact replay step while recording is active.

        ``gui_class`` is ``"gui_free"``, ``"gui_result"`` or ``"gui_only"``.
        Record accepted values and outcomes, never mouse/keyboard gestures.
        Calling this while recording is paused is intentionally a no-op, so a
        plugin need not branch around ordinary execution.
        """
        source = str(code).strip()
        if not source:
            raise ApiError("mfv.record.step() needs a non-empty Python statement.")
        try:
            classification = GuiClass.coerce(gui_class)
        except ValueError as exc:
            raise ApiError(str(exc)) from None
        if classification is None:
            raise ApiError("mfv.record.step() needs an explicit GUI class.")

        label = str(summary).strip()
        if not label:
            label = source.splitlines()[0].strip()
        self._state.recorder.append(
            label,
            code=source,
            gui_class=classification,
            command=command,
            category=str(category or "plugin"),
            **details,
        )


__all__ = ["Record"]
