"""
minflux_viewer.plugins.method
=============================
The ``[method]`` block of ``plugin.toml`` — how a plugin declares, once, the
prose that describes what it does, so a run of it can be written up.

A plugin author writes the method *once*, with the parameter values left as
``{placeholders}``; at write-up time the values the run actually used are
substituted into them.  That is the whole point: the description, the workflow
and the parameter meanings are properties of the **method** and belong with the
plugin, while the numbers are properties of the **run** and come from the
journal.  Without this split, either every plugin hard-codes a paragraph it
cannot keep in step with its own parameters, or the method-text generator grows
a bespoke renderer per plugin — which is exactly what it had.

Qt-free and side-effect-free, like :mod:`~minflux_viewer.plugins.manifest`:
reading a declaration must never import the plugin.

Substitution is deliberately **not** :meth:`str.format`.  A template is author
text that may legitimately contain a brace, and ``format`` would both crash on
it and expose attribute access (``{x.__class__}``) on recorded values.  A plain
``{name}`` scan is enough, and it can report what was missing instead of
failing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

#: A ``{name}`` slot in a template. Names follow the parameter names a plugin
#: records, so they are Python-identifier-ish.
_SLOT_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: What a slot renders as when the run recorded no such value. Visible on
#: purpose: a method paragraph that quietly drops a number it promised to state
#: is worse than one that says the number is missing.
MISSING_TEMPLATE = "<{name} not recorded>"

#: Section order of a rendered method entry.
SECTIONS = ("description", "workflow", "parameters", "outputs", "limitations")


class MethodError(ValueError):
    """A ``[method]`` block is present but malformed."""


@dataclass(frozen=True)
class MethodField:
    """One declared input or output of the method."""

    name: str
    label: str = ""
    unit: str = ""
    description: str = ""
    #: Whether this input is an answer a replay should feed back to the
    #: plugin's parameter dialog. False for a setting the run *derived* rather
    #: than asked for: it belongs in the write-up, but a recorded script that
    #: appears to set it would misdescribe where it came from.
    replay: bool = True

    @property
    def title(self) -> str:
        """What to call it in prose: the label if given, else the raw name."""
        return self.label or self.name


@dataclass(frozen=True)
class MethodSpec:
    """A plugin's declared method text, with its parameter slots unfilled."""

    title: str = ""
    description: str = ""
    workflow: tuple[str, ...] = ()
    limitations: str = ""
    inputs: tuple[MethodField, ...] = ()
    outputs: tuple[MethodField, ...] = ()
    citations: tuple[tuple[str, str], ...] = ()
    #: Free-form extras kept verbatim so a newer manifest key is not lost.
    extra: dict = field(default_factory=dict)

    @property
    def fields(self) -> dict[str, MethodField]:
        """Every declared field by name; an output never shadows an input."""
        merged = {item.name: item for item in self.outputs}
        merged.update({item.name: item for item in self.inputs})
        return merged

    def slots(self) -> set[str]:
        """Every ``{name}`` the templates reference."""
        found: set[str] = set()
        for text in (self.description, self.limitations, *self.workflow):
            found.update(_SLOT_RE.findall(text or ""))
        return found


def _text(value: Any, what: str) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(_text(item, what) for item in value)
    if not isinstance(value, str):
        raise MethodError(f"[method] {what} must be text, not {type(value).__name__}.")
    return value.strip()


def _fields(raw: Any, what: str) -> tuple[MethodField, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise MethodError(
            f"[[method.{what}]] must be a list of tables, each with a 'name'.")
    out: list[MethodField] = []
    seen: set[str] = set()
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise MethodError(
                f"[[method.{what}]] entry {position} must be a table.")
        name = str(item.get("name", "") or "").strip()
        if not name:
            raise MethodError(
                f"[[method.{what}]] entry {position} needs a 'name'.")
        if not name.isidentifier():
            raise MethodError(
                f"[[method.{what}]] name {name!r} must be a plain identifier — "
                f"it is what a {{{name}}} slot refers to.")
        if name in seen:
            raise MethodError(
                f"[[method.{what}]] declares {name!r} twice.")
        seen.add(name)
        replay = item.get("replay", True)
        if not isinstance(replay, bool):
            raise MethodError(
                f"[[method.{what}]] {name!r}: 'replay' must be true or false.")
        out.append(MethodField(
            name=name,
            label=str(item.get("label", "") or "").strip(),
            unit=str(item.get("unit", "") or "").strip(),
            description=str(item.get("description", "") or "").strip(),
            replay=replay,
        ))
    return tuple(out)


def parse_method_spec(raw: Any) -> MethodSpec | None:
    """Validate a decoded ``[method]`` table, or ``None`` when there is none.

    Raises :class:`MethodError` for a block that is present but wrong, so the
    author is told rather than silently getting no method text.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise MethodError("[method] must be a table.")
    if not raw:
        return None

    workflow = raw.get("workflow", ()) or ()
    if isinstance(workflow, str):
        workflow = (workflow,)
    if not isinstance(workflow, (list, tuple)):
        raise MethodError("[method] workflow must be a list of steps.")
    steps = tuple(_text(step, "workflow step") for step in workflow)
    if any(not step for step in steps):
        raise MethodError("[method] workflow contains an empty step.")

    citations = raw.get("citations", ()) or ()
    if isinstance(citations, str):
        citations = (citations,)
    pairs: list[tuple[str, str]] = []
    for item in citations:
        if isinstance(item, str):
            pairs.append((item.strip(), ""))
        elif isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            if not text:
                raise MethodError("[method] a citation needs 'text'.")
            pairs.append((text, str(item.get("url", "") or "").strip()))
        else:
            raise MethodError("[method] citations must be strings or tables.")

    spec = MethodSpec(
        title=_text(raw.get("title"), "title"),
        description=_text(raw.get("description"), "description"),
        workflow=steps,
        limitations=_text(raw.get("limitations"), "limitations"),
        inputs=_fields(raw.get("inputs"), "inputs"),
        outputs=_fields(raw.get("outputs"), "outputs"),
        citations=tuple(pairs),
        extra={k: v for k, v in raw.items() if k not in {
            "title", "description", "workflow", "limitations",
            "inputs", "outputs", "citations"}},
    )
    if not (spec.description or spec.workflow):
        raise MethodError(
            "[method] needs at least a 'description' or a 'workflow' — an "
            "empty block would generate an empty method paragraph.")
    return spec


def format_value(value: Any, field: MethodField | None = None) -> str:
    """Render one recorded value for prose, with its declared unit.

    Numbers lose a trailing ``.0`` because a method section reads ``4 nm``,
    not ``4.0 nm``; a sequence becomes a comma list; everything else is its
    own string.
    """
    unit = (field.unit if field is not None else "") or ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        text = f"{value:,}"
    elif isinstance(value, float):
        text = f"{value:g}" if abs(value) < 1e6 else f"{value:,.6g}"
    elif isinstance(value, (list, tuple)):
        inner = ", ".join(format_value(item) for item in value)
        return f"{inner} {unit}".strip() if unit else inner
    elif value is None:
        return "not set"
    else:
        text = str(value)
    return f"{text} {unit}".strip() if unit else text


def substitute(template: str, values: dict, fields: dict, missing: set) -> str:
    """Fill ``{name}`` slots from *values*, recording any that are absent."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in values:
            missing.add(name)
            return MISSING_TEMPLATE.format(name=name)
        return format_value(values[name], fields.get(name))

    return _SLOT_RE.sub(replace, template or "")


def render_method_text(
    spec: MethodSpec,
    values: dict | None = None,
    *,
    dataset: str = "",
) -> tuple[str, list[str]]:
    """Render *spec* with *values*, returning ``(text, missing_slot_names)``.

    The caller decides what to do about missing slots; nothing is hidden here.
    """
    values = dict(values or {})
    fields = spec.fields
    missing: set[str] = set()
    lines: list[str] = []

    title = spec.title or "Plugin analysis"
    if dataset:
        lines.append(f"{title} was applied to dataset {dataset!r}.")
    elif title:
        lines.append(f"{title}.")

    if spec.description:
        lines.append(substitute(spec.description, values, fields, missing))
    if spec.workflow:
        lines.append("Workflow.")
        for index, step in enumerate(spec.workflow, start=1):
            lines.append(
                f"  {index}. {substitute(step, values, fields, missing)}")

    reported = _reported_fields(spec.inputs, values, missing)
    if reported:
        lines.append("Parameters.")
        lines.extend(reported)
    produced = _reported_fields(spec.outputs, values, missing)
    if produced:
        lines.append("Results.")
        lines.extend(produced)

    if spec.limitations:
        lines.append("Limitations.")
        lines.append(f"  {substitute(spec.limitations, values, fields, missing)}")

    return "\n".join(lines), sorted(missing)


def _reported_fields(
    declared: tuple[MethodField, ...], values: dict, missing: set,
) -> list[str]:
    """One ``label: value — description`` line per declared field.

    A declared field the run did not record is **listed as not recorded**
    rather than dropped: a parameter block that silently shrinks cannot be
    checked against the method it claims to describe. It is also added to
    *missing*, so a caller that only inspects the returned list still learns
    about it -- the block and the report must not disagree.
    """
    lines = []
    for item in declared:
        if item.name in values:
            shown = format_value(values[item.name], item)
        else:
            missing.add(item.name)
            shown = MISSING_TEMPLATE.format(name=item.name)
        detail = f" — {item.description}" if item.description else ""
        lines.append(f"  {item.title}: {shown}{detail}")
    return lines


__all__ = [
    "MISSING_TEMPLATE",
    "SECTIONS",
    "MethodError",
    "MethodField",
    "MethodSpec",
    "format_value",
    "parse_method_spec",
    "render_method_text",
    "substitute",
]
