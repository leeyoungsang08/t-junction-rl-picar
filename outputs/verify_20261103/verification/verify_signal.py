from pathlib import Path
from src.environment import TJunctionEnv
from src.traffic_generator import generate_traffic
from src.utils import load_yaml
from src.inference import Policy
import numpy as np

root = Path("outputs/verify_20261103/verification/controlled_signal")
cfg = load_yaml("base.yaml")
cfg["simulation"].update(horizon_seconds=97, decision_seconds=7)
route = generate_traffic("medium", 3031, 97, "test", root / "routes")
env = TJunctionEnv(cfg, route, 3031, root)
try:
    state, info = env.reset(seed=3031)
    assert state.shape == (25,) and state.dtype == np.float32
    done = False
    while not done:
        action = int(not info["action_mask"][0])
        state, _, terminal, done, info = env.step(action)
        assert not terminal and env.observation_space.contains(state)
        if not done:
            assert env.conn.trafficlight.getRedYellowGreenState("J") == env.timeseries[-1]["tls_state"]
    assert info["metrics"]["forced_switches"] >= 1
    assert info["time"] == 97
    assert any(t["stage"] == "all_red" for t in env.timeseries)
    print("Live TLS readbacks matched the timeline; 97s truncation with valid float32 observation.")
    print(info["metrics"])
finally:
    env.close()
