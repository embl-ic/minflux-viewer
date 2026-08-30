"""
Self-recording for the published ``mfv`` namespaces (design §3.3).

A script or plugin that calls the API records itself: the call is written down
as the same line of Python somebody could have typed. Nothing on the caller's
side changes.

Three rules keep the output usable rather than merely complete:

**Only calls that change something are recorded.** ``mfv.data.attr()`` is a
read; a script that reads forty attributes must not emit forty lines. The
registry below is the whole list of recordable calls, and adding to it is a
deliberate act.

**Arguments are rendered as round-trippable code, or the call is not recorded
at all.** A step that looks runnable and is not is worse than an honest TODO —
the same rule the command hook follows. NumPy arrays are rendered as literals
only while they are small enough to read; a 60,000-element array in a script
helps nobody, so such a call records as a TODO naming what could not be
captured.

**Objects become variables.** ``mfv.roi.add(...)`` returns a record that a
later ``mfv.roi.points_in(roi)`` refers to, so the returned object is bound to
a generated name and remembered. Without that, the second call could not be
written down at all.

Suppression, and why
--------------------
Recording is suppressed while a **plugin** is running. A plugin calling
``ctx.run.background`` is *one* logical step, not the twenty API calls inside
it; its own ``ctx.journal.record()`` is the step worth keeping. It is also
suppressed while a recorded script is replayed, so replaying a macro does not
append a second copy of itself.
"""

from __future__ import annotations

import functools
import threading
import weakref
from typing import Any

from ..core.recorder import GuiClass

#: Longest NumPy array still written into a script as a literal. Above this the
#: call records as a TODO: an unreadable wall of numbers is not a macro, and
#: silently truncating it would produce a script that runs and is wrong.
MAX_INLINE_ARRAY = 64

#: ``namespace.method`` -> (GUI class, binds-a-variable, variable stem).
#:
#: Only what is here is recorded. A read is absent on purpose. ``view`` entries
#: are GUI_ONLY so silent mode drops them; everything that changes data is
#: GUI_FREE, and a call whose *arguments* came from a human gesture (a drawn
#: ROI) is GUI_RESULT — the outcome is data even though producing it was not.
RECORDABLE: dict[str, tuple[GuiClass, bool, str]] = {
    # data — changes the dataset or its filter
    "data.set_filter":      (GuiClass.GUI_FREE, False, ""),
    "data.clear_filter":    (GuiClass.GUI_FREE, False, ""),
    "data.add_attr":        (GuiClass.GUI_FREE, False, ""),
    "data.create":          (GuiClass.GUI_FREE, True, "ds"),
    # roi — geometry a human usually drew, so the value is the record
    "roi.add":              (GuiClass.GUI_RESULT, True, "roi"),
    "roi.remove":           (GuiClass.GUI_FREE, False, ""),
    "roi.select":           (GuiClass.GUI_FREE, False, ""),
    "roi.crop":             (GuiClass.GUI_FREE, True, "cropped"),
    # results / plot — output the macro should recreate
    "results.table":        (GuiClass.GUI_FREE, True, "table"),
    "results.close":        (GuiClass.GUI_FREE, False, ""),
    "plot.line":            (GuiClass.GUI_FREE, True, "plot"),
    "plot.scatter":         (GuiClass.GUI_FREE, True, "plot"),
    "plot.hist":            (GuiClass.GUI_FREE, True, "plot"),
    "plot.image":           (GuiClass.GUI_FREE, True, "plot"),
    # view — presentation, dropped in silent mode
    "view.render":          (GuiClass.GUI_ONLY, False, ""),
    "view.scatter":         (GuiClass.GUI_ONLY, False, ""),
    "view.histogram":       (GuiClass.GUI_ONLY, False, ""),
    "view.attribute_plot":  (GuiClass.GUI_ONLY, False, ""),
    "view.snapshot":        (GuiClass.GUI_ONLY, False, ""),
}


class _Unrenderable(Exception):
    """An argument that cannot be written as code, with the reason."""


#: Nesting depth per thread, so a recordable method that calls another records
#: only the outermost -- that is the call the user made.
_DEPTH = threading.local()


def _wrap(original, key: str):
    """Class-level wrapper for one recordable namespace method."""
    gui_class, binds, stem = RECORDABLE[key]

    @functools.wraps(original)
    def wrapper(namespace_self, *args, **kwargs):
        depth = getattr(_DEPTH, "value", 0)
        _DEPTH.value = depth + 1
        try:
            result = original(namespace_self, *args, **kwargs)
        finally:
            _DEPTH.value = depth
        if depth == 0:
            calls = getattr(getattr(namespace_self, "_facade", None), "calls", None)
            if calls is not None:
                calls._record(key, gui_class, binds, stem, args, kwargs, result)
        return result

    wrapper._mfv_recorded = True
    return wrapper


class CallRecorder:
    """
    Wraps the recordable namespace methods of one facade.

    Re-entrant by design: a recordable method that calls another one records
    only the outermost, because that is the call the user made.
    """

    def __init__(self, facade: Any) -> None:
        # ⚠ WEAK. The facade holds this object, each namespace instance holds a
        # bound wrapper, and each wrapper holds this object -- so a strong
        # reference here closes the cycle
        # ``facade → namespace → wrapper → CallRecorder → facade`` on *every*
        # AppState. Python then breaks those cycles during GC, and collecting a
        # graph that reaches Qt objects while queued events are in flight is
        # this project's documented native-abort hazard. Measured: the strong
        # form aborted the suite reproducibly, early and away from any recorder
        # test.
        self._facade_ref = weakref.ref(facade)
        self._suppress = 0
        self._names: dict[int, str] = {}      # id(object) -> variable name
        self._counts: dict[str, int] = {}

    # -- suppression ---------------------------------------------------------

    @property
    def _facade(self) -> Any:
        """The bound facade, or ``None`` once it has been collected."""
        return self._facade_ref()

    def suppressed(self) -> bool:
        return self._suppress > 0

    def suppress(self) -> _Suppression:
        """Context manager: nothing inside is recorded (plugins, replay)."""
        return _Suppression(self)

    # -- installation --------------------------------------------------------

    def install(self) -> None:
        """
        Wrap every recordable method **on its class**, once per process.

        ⚠ Not on the instance. ``setattr(namespace, name, wrapper.__get__(ns))``
        stores a *bound* method in the instance dict, so the namespace holds a
        method that holds the namespace -- one reference cycle per namespace per
        AppState, for Python's GC to break later while Qt objects are reachable.
        A class-level wrapper is a plain function bound by the descriptor
        protocol at call time and owns nothing.

        The wrapper therefore takes its policy from the namespace it is called
        on (``namespace_self._facade.calls``) rather than from a captured
        recorder, which is also what lets two facades coexist.
        """
        for key in RECORDABLE:
            namespace_name, _, method_name = key.partition(".")
            namespace = getattr(self._facade, namespace_name, None)
            if namespace is None:
                continue
            cls = type(namespace)
            original = cls.__dict__.get(method_name)
            if original is None or getattr(original, "_mfv_recorded", False):
                continue
            setattr(cls, method_name, _wrap(original, key))

    # -- recording -----------------------------------------------------------

    def _record(self, key, gui_class, binds, stem, args, kwargs, result) -> None:
        facade = self._facade
        if facade is None:
            return
        recorder = getattr(facade._state, "recorder", None)
        if recorder is None or not recorder.enabled or self.suppressed():
            return

        summary = f"mfv.{key}"
        try:
            rendered = self._render_call(key, args, kwargs)
        except _Unrenderable as exc:
            recorder.append(
                summary, gui_class=gui_class, command=f"mfv.{key}",
                unrecordable=str(exc),
            )
            return

        if binds and result is not None:
            name = self._bind(result, stem)
            rendered = f"{name} = {rendered}"
        recorder.append(summary, code=rendered, gui_class=gui_class,
                        command=f"mfv.{key}")

    def _render_call(self, key: str, args: tuple, kwargs: dict) -> str:
        parts = [self._render(value) for value in args]
        # A ``None`` keyword is the default the caller did not set -- writing
        # it back would be noise, and ``dataset=None`` already means "active".
        parts += [f"{name}={self._render(value)}" for name, value in kwargs.items()
                  if value is not None]
        return f"mfv.{key}({', '.join(parts)})"

    def _bind(self, obj: Any, stem: str) -> str:
        existing = self._names.get(id(obj))
        if existing:
            return existing
        self._counts[stem] = self._counts.get(stem, 0) + 1
        count = self._counts[stem]
        name = stem if count == 1 else f"{stem}_{count}"
        self._names[id(obj)] = name
        return name

    def _render(self, value: Any) -> str:
        """One argument as code, or raise :class:`_Unrenderable` with why."""
        import numpy as np

        # Something we already bound to a variable.
        known = self._names.get(id(value))
        if known:
            return known

        if value is None or isinstance(value, (bool, int, float, str)):
            return repr(value)

        if isinstance(value, np.ndarray):
            if value.size > MAX_INLINE_ARRAY:
                raise _Unrenderable(
                    f"an array of {value.size} values was passed; too large to "
                    "write into a script"
                )
            return repr(value.tolist())
        if isinstance(value, (np.integer, np.floating)):
            return repr(value.item())

        if isinstance(value, (list, tuple)):
            inner = ", ".join(self._render(item) for item in value)
            return f"[{inner}]" if isinstance(value, list) else f"({inner}{',' if len(value) == 1 else ''})"
        if isinstance(value, dict):
            inner = ", ".join(f"{self._render(k)}: {self._render(v)}"
                              for k, v in value.items())
            return "{" + inner + "}"

        # A dataset addresses itself by name; that is how a script refers to one.
        name = getattr(value, "name", None)
        if name and hasattr(value, "prop") and hasattr(value, "attr"):
            return f"mfv.data.get({name!r})"

        # A ROI we did not create during this recording cannot be addressed:
        # its id is a per-session UUID, so writing it into a script would
        # produce a line that runs and refers to nothing. Say so instead.
        if hasattr(value, "geometry") and hasattr(value, "id"):
            raise _Unrenderable(
                "a ROI created before recording started cannot be referenced; "
                "draw it while recording, or add it with mfv.roi.add(...)"
            )

        raise _Unrenderable(
            f"an argument of type {type(value).__name__} cannot be written as code"
        )


class _Suppression:
    def __init__(self, owner: CallRecorder) -> None:
        self._owner = owner

    def __enter__(self) -> None:
        self._owner._suppress += 1

    def __exit__(self, *_exc) -> bool:
        self._owner._suppress -= 1
        return False
