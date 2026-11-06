import numpy as np
import pytest
from src.dqn_agent import DQNAgent
from src.inference import Policy, export_model
from src.state_adapter import StateAdapter


def sample():
    return {"vehicles": [3, 2, 1], "mean_waiting_seconds": [4, 5, 6], "internal_vehicles": 1,
            "phase": 0, "green_elapsed": 0, "transitioning": False}


def test_export_cpu_inference_and_mask(agent_config, base, tmp_path):
    a = DQNAgent(agent_config)
    path = tmp_path / "best_model.pt"
    export_model(a, {**base, "observation_schema": 1}, path, {"test": True})
    p = Policy(path)
    state = sample()
    out = p.predict(state)
    assert out["action"] == "KEEP" and out["valid_actions"] == [True, False]
    state["green_elapsed"] = base["signal"]["max_green"]
    assert p.predict(state)["action"] == "SWITCH"


def test_state_shape_range_and_reject_missing_nonfinite(base):
    adapter = StateAdapter(base["normalization"], base["signal"])
    state = sample()
    state["vehicles"] = [1000, 0, 0]
    vector = adapter.encode(state)
    assert vector.shape == (12,) and vector.dtype == np.float32 and vector[0] == 1
    state["vehicles"] = [np.nan, 0, 0]
    with pytest.raises(ValueError):
        adapter.encode(state)
    with pytest.raises(TypeError):
        adapter.encode({})
