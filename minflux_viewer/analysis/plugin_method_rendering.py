'Plugin method rendering'

from __future__ import annotations


def plugin_method_entry(ev):
    """Render one plugin run from its declared ``[method]`` block.

    Returns ``(stage, sentence, citations)`` or ``None`` when the event is not
    a plugin run or its plugin ships no declaration.

    Matched on the structured payload the plugin runner attaches rather than on
    the wording of the log line: a plugin owns its own prose, and a regex over
    it would break the moment its author reworded a sentence -- which is
    exactly how the built-in rules used to lose track of this plugin.
    """
    payload = (ev.get("method_data") or {}).get("mfv_plugin")
    if not payload:
        return None
    from minflux_viewer.core.plugin_registry import available
    from minflux_viewer.utils.plugin_method import render_method_text

    wanted = payload.get("id")
    entry = next((e for e in available() if e.plugin_id == wanted), None)
    spec = getattr(entry, "method", None)
    if spec is None:
        return None
    text, missing = render_method_text(
        spec, payload.get("values") or {}, dataset=payload.get("dataset", ""))
    if missing:
        # Named, not hidden: a write-up that silently omits a parameter the
        # method promised cannot be checked against the run it describes.
        text += ("\n  (not recorded by this run: " + ", ".join(missing) + ")")
    return "analysis", text, tuple(spec.citations)
