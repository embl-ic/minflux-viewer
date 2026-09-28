"""
``mfv.plugins`` -- listing, describing and re-running the installed plugins.

This is what makes a recorded plugin run reproducible. A plugin normally asks
for its parameters through :meth:`mfv.ui.ask`; :meth:`Plugins.run` stages the
answers first, so the dialog is answered from the recording instead of by hand
and the run repeats exactly. The same declaration -- the ``[method]`` block of
``plugin.toml`` -- is what :meth:`Plugins.method_text` renders for a write-up.
"""

from __future__ import annotations

from typing import Any

from ._base import ApiError, Namespace


class Plugins(Namespace):
    """The installed plugins, and how to re-run one."""

    name = "plugins"

    def list(self) -> list[dict]:
        """Every registered plugin: id, name, menu path, and whether it runs."""
        from ..plugins import available

        rows = []
        for entry in available():
            rows.append({
                "id": entry.plugin_id or entry.name,
                "name": entry.name,
                "menu_path": list(entry.menu_path),
                "error": entry.error,
                "has_method": getattr(entry, "method", None) is not None,
            })
        return rows

    def find(self, plugin_id: str):
        """The registry entry for *plugin_id*, or a readable error."""
        from ..plugins import available

        wanted = str(plugin_id)
        entries = list(available())
        for entry in entries:
            if entry.plugin_id == wanted or entry.name == wanted:
                return entry
        known = ", ".join(
            repr(e.plugin_id or e.name) for e in entries) or "(none installed)"
        raise ApiError(f"No plugin {plugin_id!r}. Installed: {known}")

    def run(self, plugin_id: str, *, answers: dict | None = None) -> None:
        """Launch a plugin, optionally pre-answering its parameter dialog.

        *answers* is staged for the **next** :meth:`mfv.ui.ask` the plugin
        makes, which is how a recorded run replays: the values that were
        confirmed the first time are supplied instead of the dialog. Keys the
        plugin does not ask for are ignored, and keys it asks for but the
        recording lacks keep the plugin's own defaults -- a plugin that gains a
        parameter therefore still replays, with the new one at its default,
        rather than failing.

        Staging is one-shot, so a later, unrelated question in the same run is
        still asked.
        """
        entry = self.find(plugin_id)
        if entry.error:
            raise ApiError(f"Plugin {plugin_id!r} cannot run: {entry.error}")
        stage = getattr(self._facade, "stage_preset_answers", None)
        if answers is not None and stage is None:
            raise ApiError(
                "This build cannot pre-answer a plugin dialog; run it from the "
                "Plugins menu instead.")
        if answers is not None:
            stage(answers)
        try:
            entry.launch(self._state, self._facade._main_window)
        finally:
            # Never leave answers staged for whatever asks next.
            if answers is not None:
                self._facade.take_preset_answers()

    def method(self, plugin_id: str):
        """The plugin's declared :class:`MethodSpec`, or ``None``."""
        return getattr(self.find(plugin_id), "method", None)

    def method_text(
        self,
        plugin_id: str,
        values: dict | None = None,
        *,
        dataset: str = "",
    ) -> str:
        """Render the plugin's declared method text with *values* filled in.

        With no *values*, the most recent recorded run of that plugin is used,
        so this answers "write up what I just did" directly.
        """
        from ..plugins.method import render_method_text

        spec = self.method(plugin_id)
        if spec is None:
            raise ApiError(
                f"Plugin {plugin_id!r} declares no [method] block, so there is "
                f"no method text to render.")
        if values is None:
            values, dataset = self._last_run(plugin_id, dataset)
        text, missing = render_method_text(spec, values, dataset=dataset)
        return text

    def last_run(self, plugin_id: str) -> dict | None:
        """The values the most recent recorded run of *plugin_id* used."""
        values, _dataset = self._last_run(str(plugin_id), "")
        return values or None

    def _last_run(self, plugin_id: str, dataset: str) -> tuple[dict, str]:
        for event in reversed(self._state.log_history):
            payload = (event.get("method_data") or {}).get("mfv_plugin")
            if payload and payload.get("id") == plugin_id:
                return dict(payload.get("values") or {}), (
                    dataset or payload.get("dataset", ""))
        return {}, dataset


__all__ = ["Plugins"]
