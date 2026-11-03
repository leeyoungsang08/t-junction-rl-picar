import numpy as np
import pytest
from src.dqn_agent import DQNAgent
from src.environment import TJunctionEnv
from src.inference import Policy, export_model
from src.reward import reward_components
from src.state_adapter import StateAdapter
from src.utils import write_json


def test_schema_v2_distinguishes_queues_and_loads_policy(base, agent_config, tmp_path):
    adapter = StateAdapter(base["normalization"], base["signal"], 2)
    raw = dict(vehicles=[2, 0, 0], mean_waiting_seconds=[5, 0, 0], internal_vehicles=0,
               phase=0, green_elapsed=20, transitioning=False, queues=[1, 0, 0],
               pending=[0, 0, 0], longest_wait=[10, 0, 0], recent_requests=[2, 0, 0])
    assert not np.array_equal(adapter.encode(raw), adapter.encode({**raw, "queues": [2, 0, 0]}))
    assert adapter.encode({**raw, "pending": [200, 0, 0]})[15] > adapter.encode({**raw, "pending": [100, 0, 0]})[15]
    with pytest.raises(ValueError, match="pending"):
        adapter.encode({**raw, "pending": None})
    agent = DQNAgent({**agent_config, "state_dim": len(adapter.features)})
    path = tmp_path / "v2.pt"
    export_model(agent, base, path, {})
    assert Policy(path).predict(raw)["valid_actions"] == [True, True]


@pytest.mark.integration
def test_pending_is_observed_and_penalized_without_future_leak(base, tmp_path):
    horizon = 15
    base["simulation"]["horizon_seconds"] = horizon
    vehicles = {f"v{i}": {"origin": "W", "requested_depart": 0} for i in range(30)}
    vehicles["future"] = {"origin": "S", "requested_depart": 14}
    route = tmp_path / "burst.rou.xml"
    route.write_text('<routes><route id="we" edges="W_in E_out"/><route id="sw" edges="S_in W_out"/>'
                     + ''.join(f'<vehicle id="{v}" route="we" depart="0"/>' for v in vehicles if v != "future")
                     + '<vehicle id="future" route="sw" depart="14"/></routes>')
    write_json(route.with_suffix(".json"), {"vehicles": vehicles, "horizon_seconds": horizon})
    env = TJunctionEnv(base, route, output_dir=tmp_path / "run")
    try:
        _, info = env.reset()
        assert info["raw_state"]["pending"] == [0, 0, 0]
        with pytest.raises(ValueError, match="discrete"):
            env.step(0.7)
        _, _, _, _, info = env.step(0)
        assert info["raw_state"]["pending"][0] > 0
        assert info["raw_state"]["recent_requests"] == [30, 0, 0]
        assert info["reward_components"]["waiting_cost"] > 0
    finally:
        env.close()


def test_active_internal_and_long_wait_costs(base):
    r, costs = reward_components([0, 0, 0], 0, 1, [80, 0, 0], 0, base["reward"],
                                 pending=3, internal_count=3)
    assert r < 0
    assert costs["waiting_cost"] == pytest.approx(3 / base["reward"]["waiting_scale"])
    assert costs["internal_cost"] > 0 and costs["starvation_cost"] > 0


def test_truncation_bootstraps_but_true_termination_does_not(agent_config):
    import torch

    cfg = {**agent_config, "batch_size": 1, "learning_start": 1}
    for terminated in (False, True):
        agent = DQNAgent(cfg)
        with torch.no_grad():
            for model in (agent.online, agent.target):
                for param in model.parameters():
                    param.zero_()
            agent.target.layers[-1].bias[:] = torch.tensor([2.0, 5.0])
        state = np.zeros(12, dtype=np.float32)
        agent.replay.add(state, 0, 0.0, state, terminated, not terminated, [True, False])
        agent.global_step = 1
        expected = 0 if terminated else 2 * cfg["gamma"] - 0.5
        assert agent.update()["loss"] == pytest.approx(expected)


@pytest.mark.integration
def test_stopped_vehicle_inside_junction_activates_cost(base, tmp_path):
    base["simulation"].update(horizon_seconds=45, decision_seconds=1)
    route = tmp_path / "blocked.rou.xml"
    route.write_text('<routes><route id="we" edges="W_in E_out"/>'
                     '<vehicle id="v" route="we" depart="0" departPos="220" departSpeed="max"/></routes>')
    write_json(route.with_suffix(".json"), {"vehicles": {"v": {"origin": "W", "requested_depart": 0}},
                                           "horizon_seconds": 45})
    env = TJunctionEnv(base, route, output_dir=tmp_path / "blocked")
    try:
        _, info = env.reset()
        for _ in range(20):
            _, _, _, _, info = env.step(0)
            if info["raw_state"]["internal_vehicles"]:
                env.conn.vehicle.setSpeedMode("v", 0)
                env.conn.vehicle.setSpeed("v", 0)
                break
        else:
            pytest.fail("Vehicle never entered the junction")
        _, _, _, _, info = env.step(0)
        assert info["reward_components"]["internal_cost"] > 0
    finally:
        env.close()
