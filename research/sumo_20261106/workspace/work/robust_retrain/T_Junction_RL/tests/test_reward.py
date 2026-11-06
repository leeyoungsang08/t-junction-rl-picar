import pytest
from src.reward import reward_components


def test_incremental_waiting_is_not_lifetime_recounted(base):
    cfg = base["reward"]
    a, terms = reward_components([1, 0, 0], 1, 0, [1, 0, 0], 0, cfg)
    b, _ = reward_components([1, 0, 0], 1, 0, [2, 0, 0], 0, cfg)
    assert a == b
    assert terms["waiting_cost"] == pytest.approx(1 / cfg["waiting_scale"])


def test_simple_reward_and_switch_penalty(base):
    cfg = base["reward"]
    x, _ = reward_components([0, 0, 0], 0, 0, [0, 0, 0], 1, cfg)
    assert x == -cfg["switch_weight"]
    cfg["mode"] = "simple"
    y, _ = reward_components([3, 0, 0], 100, 100, [200, 0, 0], 2, cfg)
    assert y == pytest.approx(-3 / cfg["queue_scale"])
