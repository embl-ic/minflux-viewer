"""Registry contract for optional tracking state-segmentation adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .tracking_stats import PreparedTrajectories, TrackingResult


@dataclass(frozen=True)
class TrackingStateAdapterSpec:
    """One optional, versioned state-inference backend.

    Dependencies remain with the registering plugin.  The viewer supplies only
    prepared trajectories and requires the adapter to return the common
    :class:`TrackingResult` envelope with model/version provenance.
    """

    adapter_id: str
    label: str
    version: str
    description: str
    runner: Callable[..., TrackingResult]
    availability: Callable[[], tuple[bool, str]] | None = None
    citations: tuple[str, ...] = ()


_STATE_ADAPTERS: dict[str, TrackingStateAdapterSpec] = {}


def register_tracking_state_adapter(
    spec: TrackingStateAdapterSpec,
    *,
    replace: bool = False,
) -> None:
    key = str(spec.adapter_id).strip()
    if not key:
        raise ValueError("A tracking state adapter needs a non-empty adapter_id.")
    if key in _STATE_ADAPTERS and not replace:
        raise ValueError(f"Tracking state adapter {key!r} is already registered.")
    _STATE_ADAPTERS[key] = spec


def tracking_state_adapters() -> tuple[TrackingStateAdapterSpec, ...]:
    return tuple(_STATE_ADAPTERS[key] for key in sorted(_STATE_ADAPTERS))


def get_tracking_state_adapter(adapter_id: str) -> TrackingStateAdapterSpec:
    try:
        return _STATE_ADAPTERS[str(adapter_id)]
    except KeyError as exc:
        raise KeyError(f"Unknown tracking state adapter {adapter_id!r}.") from exc


def tracking_state_adapter_availability(adapter_id: str) -> tuple[bool, str]:
    spec = get_tracking_state_adapter(adapter_id)
    if spec.availability is None:
        return True, "available"
    available, reason = spec.availability()
    return bool(available), str(reason)


def run_tracking_state_adapter(
    adapter_id: str,
    data: PreparedTrajectories,
    **parameters,
) -> TrackingResult:
    spec = get_tracking_state_adapter(adapter_id)
    available, reason = tracking_state_adapter_availability(adapter_id)
    if not available:
        raise RuntimeError(
            f"Tracking state adapter {spec.label!r} is unavailable: {reason}")
    result = spec.runner(data, **parameters)
    if not isinstance(result, TrackingResult):
        raise TypeError("A tracking state adapter must return TrackingResult.")
    return result


__all__ = [
    "TrackingStateAdapterSpec",
    "get_tracking_state_adapter",
    "register_tracking_state_adapter",
    "run_tracking_state_adapter",
    "tracking_state_adapter_availability",
    "tracking_state_adapters",
]
