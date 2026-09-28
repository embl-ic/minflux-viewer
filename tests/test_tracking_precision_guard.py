"""Static-emitter precision estimators must not claim trajectory motion as precision."""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("runner_name", ["run_frc", "run_stddev_per_trace"])
def test_static_precision_methods_refuse_a_tracking_role(monkeypatch, runner_name) -> None:
    pytest.importorskip("PyQt6")
    from minflux_viewer.analysis import localization_precision as precision

    dataset = SimpleNamespace(
        name="cargo",
        state={
            "tracking_role": "Tracking",
            "tracking_role_auto": "Tracking",
            "tracking_role_auto_reason": "traces move 100 nm",
        },
    )
    state = SimpleNamespace(active_dataset=dataset)
    shown = []
    monkeypatch.setattr(
        precision, "_show_info_dialog",
        lambda _parent, title, text: shown.append((title, text)))

    getattr(precision, runner_name)(None, state)

    assert len(shown) == 1
    assert "tracking channel" in shown[0][0].lower()
