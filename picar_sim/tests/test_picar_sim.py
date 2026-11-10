"""Tests for the PiCar Real-to-Sim package (requires SUMO; no PyTorch needed)."""
import numpy as np
import pytest

pytest.importorskip("sumolib")

from picar_sim.dqn import NumpyAgent, Policy, Replay, save_policy  # noqa: E402
from picar_sim.env import PicarEnv  # noqa: E402
from picar_sim.map_builder import build, load_config  # noqa: E402
from picar_sim.observation import FEATURES, CarLedger, Encoder  # noqa: E402
from picar_sim.real_bridge import replay  # noqa: E402
from picar_sim.supervisor import IntersectionSupervisor  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_map_builds_with_six_controlled_movements(cfg, tmp_path):
    net, info = build(cfg, tmp_path)
    assert net.exists()
    assert len(info["junction_internal_m"]) == 6
    assert all(v > 0 for v in info["lanes"].values())


def test_supervisor_never_grants_two_approaches_and_waits_for_empty_box(cfg):
    sup = IntersectionSupervisor(cfg["signal"])
    occupancy = 1
    for step in range(600):
        if sup.mask()[1]:
            sup.request(1)
        permits = sup.tick(0.1, occupancy, 0.0, 0.0)
        assert sum(v == "GO" for v in permits.values()) <= 1
        if sup.stage == "clear" and step < 300:
            assert all(v == "STOP" for v in permits.values())
        if step == 300:
            occupancy = 0
    assert sup.switches >= 1


def test_supervisor_faults_on_stale_occupancy_and_link_loss(cfg):
    sup = IntersectionSupervisor(cfg["signal"])
    assert sup.tick(0.1, 0, 5.0, 0.0) == {"W": "STOP", "E": "STOP", "N": "STOP"}
    assert sup.fault
    assert sup.tick(0.1, None, 0.0, 0.0)["W"] == "STOP"
    assert sup.tick(0.1, 0, 0.0, 9.0)["W"] == "STOP"
    assert sup.tick(0.1, 0, 0.0, 0.0)["W"] == "GO"


def test_mask_respects_min_and_max_green(cfg):
    sup = IntersectionSupervisor(cfg["signal"])
    assert sup.mask().tolist() == [True, False]
    for _ in range(int(cfg["signal"]["min_green_s"] / 0.1) + 1):
        sup.tick(0.1, 0, 0, 0)
    assert sup.mask().tolist() == [True, True]
    for _ in range(int(cfg["signal"]["max_green_s"] / 0.1)):
        sup.tick(0.1, 0, 0, 0)
    assert sup.forced_switches == 1 and sup.phase == 1   # max green forces W -> E through yellow/clear


def test_unknown_values_are_not_zero(cfg):
    led = CarLedger()
    snap = led.snapshot(10.0)
    assert snap["occupancy_age"] == float("inf")          # never reported -> supervisor faults
    enc = Encoder(cfg["signal"]["max_green_s"], cfg["signal"]["clear_timeout_s"])
    assert enc.encode(snap, IntersectionSupervisor(cfg["signal"]).status(), [2, 2, 2]).shape == (len(FEATURES),)


def test_no_entries_during_occupancy_fault_and_no_unsafe_events():
    env = PicarEnv(faults=[(30, 60, "occupancy")])
    obs, info = env.reset(seed=1234, cars=5, bias="connector_heavy", horizon=120)
    rng = np.random.default_rng(0)
    done, during = False, 0
    while not done:
        c0 = env.crossings
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(info["mask"]))))
        if 31 <= env.t <= 60:
            during += env.crossings - c0
        done = term or trunc
    m = info["metrics"]
    assert during == 0
    assert m["collisions"] == m["conflict_steps"] == m["stopline_violations"] == 0


def test_numpy_dqn_learns_a_trivial_target():
    rng = np.random.default_rng(0)
    agent = NumpyAgent(4, 2, {"hidden": [16], "lr": 1e-2, "gamma": 0.0, "grad_clip": 10, "double": True}, rng)
    rb = Replay(256, 4, 2, rng)
    for _ in range(256):
        o = rng.random(4).astype(np.float32)
        a = int(rng.integers(2))
        rb.add(o, a, float(a), o, 1.0, np.array([True, True]))
    for _ in range(400):
        agent.update(rb.sample(64))
    q = agent.q(np.full(4, .5, np.float32))
    assert q[1] - q[0] > 0.8


def test_sim_event_log_replays_to_identical_decisions(cfg, tmp_path):
    """Sim-to-real contract: the central-PC bridge fed with the SUMO status events reproduces the
    exact action sequence of the policy running inside SUMO."""
    rng = np.random.default_rng(3)
    env = PicarEnv(log_decisions=True)
    agent = NumpyAgent(env.obs_size, env.n_actions, {"hidden": [32, 32], "lr": 1e-3, "gamma": .9, "grad_clip": 10}, rng)
    path = tmp_path / "p.npz"
    save_policy(path, agent.weights(), {"schema": "picar_v1", "signal": cfg["signal"], "control_mode": "cycle"})
    pol = Policy(path)
    obs, info = env.reset(seed=1500, cars=4, bias="uniform", horizon=60)
    start, done = env.t, False
    while not done:
        obs, r, term, trunc, info = env.step(pol.act(obs, info["mask"]))
        done = term or trunc
    sim_actions = [d["action"] for d in env.obs_log]
    res = replay(str(path), None, events=env.event_log, start_t=start, end_t=env.t - 1.0, mode="active")
    bridge_actions = [e["decision"]["applied"] for e in res["log"] if e["decision"]]
    n = min(len(sim_actions), len(bridge_actions))
    assert n >= 50
    assert sim_actions[:n] == bridge_actions[:n]


def test_box_blocking_guard_prevents_connector_gridlock():
    """6 cars on the 1-car connector used to gridlock both T junctions; the exit-space check stops it."""
    from picar_sim.train import controller_act, run_episode
    env = PicarEnv()
    act = controller_act(env, "queue_rule")
    runs = [run_episode(env, 1500 + s, 6, "connector_heavy", act, horizon=200) for s in range(4)]
    assert not any(m["deadlock"] for m in runs)
    assert all(m["collisions"] == m["conflict_steps"] == m["stopline_violations"] == 0 for m in runs)


def test_max_wait_guard_forces_starving_approach(cfg):
    c = {**cfg["signal"], "control_mode": "direct", "max_wait_guard_s": 30}
    sup = IntersectionSupervisor(c)
    for _ in range(int(c["min_green_s"] / 0.1) + 2):
        sup.tick(0.1, 0, 0, 0)
    sup.observe_waits([0.0, 10.0, 31.0])          # N starving
    assert sup.mask().tolist() == [False, False, True]
    sup.request(2)
    assert sup.guard_switches == 1
    sup2 = IntersectionSupervisor({**c, "max_wait_guard_s": 0})
    for _ in range(int(c["min_green_s"] / 0.1) + 2):
        sup2.tick(0.1, 0, 0, 0)
    sup2.observe_waits([0.0, 10.0, 31.0])
    assert sup2.mask().all()                      # guard off: policy free
