import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate
from src.inference import Policy, export_model
from src.dqn_agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.signal_controller import DirectPhaseSignalController
from src.train import resolved_config, train
from src.utils import ROOT


@pytest.mark.parametrize("origin,target", [(a, b) for a in range(3) for b in range(3) if a != b])
def test_every_direct_pair_has_exact_protection(base, links, origin, target):
    controller = DirectPhaseSignalController(links, base["signal"])
    controller.phase = origin
    controller.green_elapsed = base["signal"]["min_green"]
    controller.request(target)
    output = []
    for _ in range(5):
        signal = controller.before_second()
        output.append(signal)
        controller.after_second()
    directions = "WES"
    assert output == [
        "".join("y" if group == directions[origin] else "r" for group in controller.groups)
    ] * 3 + ["r" * 6, "".join("G" if group == directions[target] else "r" for group in controller.groups)]


def test_per_weights_use_buffer_max_and_duplicate_updates_use_max():
    replay = ReplayBuffer(3, 3, prioritized=True, alpha=1)
    for _ in range(3):
        replay.add(np.zeros(3), 0, 0, np.zeros(3), False, False, [True, False])
    replay.update_priorities([0, 1, 2, 2], [1, 2, 8, 4])
    batch = replay.sample(3, np.random.default_rng(0), beta=1)
    assert replay.priorities[2] == pytest.approx(8 + replay.priority_epsilon)
    assert batch["weights"] == pytest.approx(replay.priorities.min() / replay.priorities[batch["indices"]])


@pytest.mark.parametrize("terminal,truncated,bootstrap,duration,expected", [
    (True, False, True, 0.4, 1.5),
    (False, True, False, 0.4, 1.5),
    (False, True, True, 0.5, 2.5),
])
def test_bootstrap_uses_duration_and_skips_terminal_masks(agent_config, terminal, truncated, bootstrap, duration, expected):
    import torch

    agent_config.update(batch_size=1, learning_start=1, gamma=0.25, bootstrap_truncated=bootstrap)
    agent = DQNAgent(agent_config)
    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.zero_()
        for parameter in agent.target.parameters():
            parameter.zero_()
        agent.target.layers[-1].bias.fill_(2)
    agent.replay.add(np.zeros(12), 0, 2, np.zeros(12), terminal, truncated,
                     [bootstrap and not terminal, False], duration)
    assert agent.update()["loss"] == pytest.approx(expected)


@pytest.mark.integration
def test_direct_evaluation_loads_model_and_preserves_baselines(tmp_path):
    fixed = {}
    for mode in ["cycle", "direct"]:
        base, cfg = resolved_config("smoke", algorithm="dqn", control_mode=mode)
        model = tmp_path / mode / "checkpoints/best_model.pt"
        export_model(DQNAgent(cfg), base, model, {})
        assert len(Policy(model).payload["features"]) == 25
        rows = evaluate(model, tmp_path / mode / "eval", seeds=1, horizon=80, scenarios=["medium"],
                        controllers=["dqn", "fixed_time", "rule_based", "max_pressure_proxy"])
        assert len(rows) == 4 and rows["route_sha256"].nunique() == 1
        fixed[mode] = rows.loc[rows.controller != "dqn", "cohort_delay_s"].tolist()
    assert fixed["cycle"] == fixed["direct"]


@pytest.mark.integration
def test_sumo_interrupted_and_resumed_matches_uninterrupted(tmp_path):
    settings = dict(mode="smoke", algorithm="rainbow_lite", control_mode="direct", training_seed=42)
    original = train(output=tmp_path / "continuous", **settings)
    resumed = train(output=tmp_path / "resumed", stop_after=1, **settings)
    train(output=resumed, resume=resumed / "checkpoints/training_checkpoint.pt")
    for name in ["updates.csv", "validation.csv", "training.csv"]:
        a = pd.read_csv(original / "logs" / name)
        b = pd.read_csv(resumed / "logs" / name)
        timing = [key for key in a.columns if key.endswith("wall_seconds") or key == "elapsed_seconds"]
        pd.testing.assert_frame_equal(a.drop(columns=timing), b.drop(columns=timing))
    for folder in [original, resumed]:
        status = json.loads((folder / "training_status.json").read_text())
        assert status["complete"] and status["updates"] > 0
        policy = Policy(folder / "checkpoints/best_model.pt")
        assert policy.payload["metadata"]["algorithm"] == "rainbow_lite"


def test_supplied_legacy_policies_load_without_converting_weights():
    v1 = Policy(ROOT / "outputs/reference/legacy_v1.pt")
    v2 = Policy(ROOT / "outputs/revision/seed42/checkpoints/best_model.pt")
    assert len(v1.adapter.features) == 12 and len(v2.adapter.features) == 25
    assert v1.payload.get("action_dim", 2) == v2.payload.get("action_dim", 2) == 2


@pytest.mark.integration
def test_forced_switch_log_distinguishes_requested_hold(base, tmp_path):
    from src.environment import TJunctionEnv
    from src.traffic_generator import generate_traffic

    base["simulation"]["horizon_seconds"] = 47
    base["signal"].update(control_mode="direct", max_green=43)
    route = generate_traffic("low", 1000, 47, directory=tmp_path / "routes")
    env = TJunctionEnv(base, route, 1000, tmp_path / "episode")
    try:
        _, info = env.reset()
        for _ in range(9):
            _, _, _, _, info = env.step(0)
        assert info["forced_switches"] == 1
        row = env.decisions[-1]
        assert row["requested_action"] == 0 and row["requested_switch_target"] is None
        assert row["executed_switch_target"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["cycle", "direct"])
def test_switch_mandated_at_decision_boundary_counts_as_forced(base, links, mode):
    from src.signal_controller import SignalController

    cls = SignalController if mode == "cycle" else DirectPhaseSignalController
    controller = cls(links, base["signal"])
    controller.green_elapsed = base["signal"]["max_green"]
    controller.request(1)
    assert controller.forced_switches == 1 and controller.stage == "yellow"


@pytest.mark.integration
def test_libsumo_matches_traci_metrics_and_signal_trace(base, tmp_path):
    pytest.importorskip("libsumo")
    from src.evaluate import run_episode
    from src.traffic_generator import generate_traffic

    base["simulation"]["horizon_seconds"] = 180
    route = generate_traffic("south_bias", 2000, 180, "validation", tmp_path)
    traci = run_episode(base, route, 2000, "max_pressure_proxy", tmp_path / "traci")
    base["simulation"]["backend"] = "libsumo"
    library = run_episode(base, route, 2000, "max_pressure_proxy", tmp_path / "libsumo")
    assert {k: v for k, v in traci.items() if k != "wall_seconds"} == {
        k: v for k, v in library.items() if k != "wall_seconds"
    }
    for filename in ("timeseries.csv", "decisions.csv"):
        assert (tmp_path / "traci" / filename).read_bytes() == (tmp_path / "libsumo" / filename).read_bytes()
