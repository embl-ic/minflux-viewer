'Session'

from __future__ import annotations


class _StateLogAdapter:
    """Callable log adapter used by the MSR parser and older tests."""

    def __init__(self, state) -> None:
        self._state = state

    def __call__(self, message: str) -> None:
        text = str(message)
        lowered = text.lower().lstrip()
        if lowered.startswith("[error]") or lowered.startswith("error"):
            level = "ERROR"
        elif lowered.startswith("[warn]") or lowered.startswith("warn"):
            level = "WARN"
        else:
            level = "INFO"
        self._state.log(text, level)


def state_namespace_for(parsed: dict):
    """A stand-in for the module-global ``msr.state`` over ONE parse result.

    Exposes ``mfx_map`` / ``mbm_map`` / ``mbm_meta_map`` plus ``mfx`` / ``mbm``
    (the first array).  Used so a reader dialog — and the batch worker, which
    has no dialog at all — stay independent of the shared global, which only
    ever reflects the most-recently-parsed file.
    """
    from types import SimpleNamespace

    mfx_map = dict((parsed or {}).get("mfx_map") or {})
    mbm_map = dict((parsed or {}).get("mbm_map") or {})
    return SimpleNamespace(
        mfx_map=mfx_map,
        mbm_map=mbm_map,
        mbm_meta_map=dict((parsed or {}).get("mbm_meta_map") or {}),
        mfx=next(iter(mfx_map.values()), None),
        mbm=next(iter(mbm_map.values()), None),
    )
