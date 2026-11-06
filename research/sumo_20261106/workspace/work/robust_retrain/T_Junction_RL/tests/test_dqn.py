import numpy as np
import torch
from src.dqn_agent import DQNAgent, atomic_torch_save, masked_bootstrap


def fill(agent):
    for i in range(10):
        agent.replay.add(np.ones(12, np.float32) * i / 10, i % 2, -float(i), np.ones(12, np.float32) / 2,
                         False, i % 3 == 0, [True, False])
    agent.global_step = 10


def test_network_update_and_target_sync(agent_config):
    a = DQNAgent(agent_config)
    assert a.online(torch.zeros(3, 12)).shape == (3, 2)
    fill(a)
    before = {k: v.clone() for k, v in a.online.state_dict().items()}
    loss = a.update()
    assert loss["loss"] > 0
    assert any(not torch.equal(before[k], v) for k, v in a.online.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in a.target.state_dict().items())
    a.update()
    assert all(torch.equal(a.online.state_dict()[k], v) for k, v in a.target.state_dict().items())


def test_target_mask_excludes_unavailable_high_q():
    online = torch.tensor([[2., 100.], [100., 3.]])
    target = torch.tensor([[7., 800.], [900., 8.]])
    mask = torch.tensor([[True, False], [False, True]])
    for double in (False, True):
        assert torch.equal(masked_bootstrap(online, target, mask, double), torch.tensor([7., 8.]))


def test_checkpoint_restores_buffer_rng_optimizer_and_next_update(agent_config, tmp_path):
    a = DQNAgent(agent_config)
    fill(a)
    a.update()
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save(a.checkpoint(4, {"test": True}), path)
    first_loss = a.update()["loss"]
    b = DQNAgent(agent_config)
    restored = b.restore(path)
    second_loss = b.update()["loss"]
    assert restored["next_episode"] == 4 and b.replay.size == a.replay.size
    assert first_loss == second_loss
    assert a.epsilon == b.epsilon
    assert all(torch.equal(v, b.online.state_dict()[k]) for k, v in a.online.state_dict().items())
