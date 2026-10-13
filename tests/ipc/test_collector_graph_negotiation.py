from __future__ import annotations

from typing import Any

import pytest

from uni_rl.env_contract import negotiate_collector_graph


class _Env:
    def __init__(self, mode: str = "eager") -> None:
        self.requested: list[tuple[str, bool]] = []
        self.mode = mode

    def configure_collector_graph(self, mode: str, *, allow_eager_fallback: bool) -> None:
        self.requested.append((mode, allow_eager_fallback))
        self.mode = mode

    def get_collector_graph_manifest(self) -> dict[str, Any]:
        return {"mode": self.mode, "enabled": False}


def test_collector_graph_negotiation_uses_public_env_protocol() -> None:
    env = _Env()

    result = negotiate_collector_graph(env, " EAGER ", allow_eager_fallback=True)

    assert result.mode == "eager"
    assert result.manifest == {"mode": "eager", "enabled": False}
    assert env.requested == [("eager", True)]


def test_collector_graph_disabled_is_compatible_without_protocol() -> None:
    result = negotiate_collector_graph(object(), "disabled")

    assert result.mode == "disabled"
    assert result.manifest is None


def test_non_disabled_mode_requires_public_protocol() -> None:
    with pytest.raises(NotImplementedError, match="public collector graph mode protocol"):
        negotiate_collector_graph(object(), "eager")


@pytest.mark.parametrize("mode", [True, 1, "capture"])
def test_collector_graph_mode_fails_closed(mode: object) -> None:
    with pytest.raises((TypeError, ValueError), match="collector graph mode"):
        negotiate_collector_graph(_Env(), mode)  # type: ignore[arg-type]
