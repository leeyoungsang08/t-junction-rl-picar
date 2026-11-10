"""Verify the PyTorch backend (CPU or GPU) end to end. Run once on Colab / your GPU PC.

python -m picar_sim.check_torch_backend
Checks: torch agent trains a few episodes on the SUMO env, the exported .npz policy gives the same Q values
in numpy (the code that runs on the central PC) as in torch, and a resume from checkpoint works.
"""
import json
import tempfile
from pathlib import Path

import numpy as np


def main(episodes=3):
    import torch
    from picar_sim.dqn import Policy, TorchAgent
    from picar_sim.train import train

    report = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
              "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    with tempfile.TemporaryDirectory() as d:
        out = train(seed=11, episodes=episodes, out=d, backend="torch", control_mode="direct",
                    agent_overrides={"learning_start": 64, "validation_every": episodes, "validation_seeds_per": 1})
        out2 = train(seed=11, episodes=episodes + 1, out=d, resume=True, backend="torch", control_mode="direct",
                     agent_overrides={"learning_start": 64, "validation_every": 1, "validation_seeds_per": 1})
        pol = Policy(Path(out2) / "last_policy.npz")
        status = json.loads((Path(out2) / "training_status.json").read_text())
        rng = np.random.default_rng(0)
        agent = TorchAgent(len(pol.meta["features"]), pol.meta["n_actions"],
                           {**pol.meta["agent"], "device": "auto"}, rng)
        # load the exported numpy weights back into torch and compare outputs on random observations
        lin = [m for m in agent.online if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for k, layer in enumerate(lin, 1):
                layer.weight.copy_(torch.as_tensor(pol.p[f"W{k}"].T))
                layer.bias.copy_(torch.as_tensor(pol.p[f"b{k}"]))
        obs = rng.random((32, len(pol.meta["features"]))).astype(np.float32)
        q_np = np.stack([pol.q(o) for o in obs])
        q_t = np.stack([agent.q(o) for o in obs])
        report.update(episodes_done=status["episodes_done"], backend=status["backend"],
                      max_abs_q_diff=float(np.abs(q_np - q_t).max()))
    report["ok"] = report["max_abs_q_diff"] < 1e-4 and report["episodes_done"] == episodes + 1
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
