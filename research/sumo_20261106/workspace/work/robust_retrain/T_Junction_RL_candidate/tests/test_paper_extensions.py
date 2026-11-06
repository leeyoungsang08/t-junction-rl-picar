"""V3 CPU tests. Full pytest additionally checks these against a real SUMO install."""
import numpy as np
import pytest
import torch
from src.signal_controller import DirectPhaseSignalController
from src.baselines import baseline_action
from src.dqn_network import QNetwork
from src.replay_buffer import ReplayBuffer
from src.dqn_agent import DQNAgent
from src.inference import Policy, export_model
from src.state_adapter import StateAdapter


def test_direct_skips_cyclic_phase_with_safe_clearance(base, links):
    base['signal']['control_mode'] = 'direct'
    c = DirectPhaseSignalController(links, base['signal'])
    for _ in range(10):
        c.before_second(); c.after_second()
    c.request(2)
    history = []
    for _ in range(4):
        history.append((c.stage, c.before_second()))
        c.after_second()
    assert [x[0] for x in history] == ['yellow', 'yellow', 'yellow', 'all_red']
    assert c.phase == 2 and c.stage == 'green'
    assert all(not ('G' in value and 'y' in value) for _, value in history)
    assert c.switches == 1


def test_direct_max_green_forces_transition(base, links):
    base['signal']['control_mode'] = 'direct'
    c = DirectPhaseSignalController(links, base['signal'])
    for _ in range(45):
        c.before_second(); c.after_second()
    assert c.mask().tolist() == [False, True, True]
    c.before_second()
    assert c.stage == 'yellow' and c.forced_switches == 1


def test_dueling_double_per_training_and_checkpoint(base, agent_config, tmp_path):
    conf = dict(agent_config, state_dim=25, action_dim=3, dueling=True,
                double_dqn=True, prioritized_replay=True)
    a = DQNAgent(conf)
    for i in range(10):
        a.replay.add(np.ones(25, np.float32) * i / 10, i % 3, -float(i),
                     np.zeros(25, np.float32), False, i % 3 == 0, [True]*3)
    a.global_step = 10
    assert a.update()['loss'] >= 0
    assert a.replay.priorities[:a.replay.size].max() > 0
    assert QNetwork(state_dim=25, actions=3, dueling=True)(torch.zeros(2, 25)).shape == (2, 3)


def test_prioritized_sampling_is_nonuniform():
    buffer = ReplayBuffer(4, 3, 3, prioritized=True)
    for i in range(4):
        buffer.add(np.zeros(3), i % 3, 0, np.ones(3), False, False, [True] * 3)
    buffer.update_priorities(np.arange(4), [0.01, 0.01, 0.01, 100.0])
    selected = np.concatenate([buffer.sample(2, np.random.default_rng(i))['indices'] for i in range(100)])
    assert (selected == 3).mean() > 0.7


@pytest.mark.integration
def test_direct_phase_sumo_episode(base, tmp_path):
    from src.traffic_generator import generate_traffic
    from src.environment import TJunctionEnv
    base['signal']['control_mode'] = 'direct'
    base['simulation'].update(horizon_seconds=120, decision_seconds=5)
    route = generate_traffic('low', 1000, 120, 'train', directory=tmp_path / 'routes')
    env = TJunctionEnv(base, route, 1000, tmp_path / 'test_direct_env')
    try:
        _, info = env.reset(seed=1000)
        while True:
            valid = np.flatnonzero(info['action_mask'])
            selected = 2 if 2 in valid else int(valid[0])
            _, _, done, truncated, info = env.step(selected)
            if done or truncated:
                break
        assert info['metrics']['collision_events'] == 0
        assert info['metrics']['observed_seconds'] == 120
        assert info['metrics']['requested_vehicles'] >= info['metrics']['completed_vehicles']
    finally:
        env.close()
