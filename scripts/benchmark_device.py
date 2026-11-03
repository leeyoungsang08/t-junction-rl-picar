import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dqn_agent import DQNAgent
from src.train import resolved_config


if __name__ == "__main__":
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        _, cfg = resolved_config("quick")
        cfg.update(device=device, learning_start=64, replay_capacity=1024)
        agent = DQNAgent(cfg)
        rng = np.random.default_rng(0)
        for i in range(1024):
            agent.replay.add(rng.random(cfg["state_dim"]), i % 2, -1, rng.random(cfg["state_dim"]), False, False, [True, True])
        agent.global_step = 2000
        for _ in range(20):
            agent.update()
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            agent.update()
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{device}: {(time.perf_counter() - start) * 1000 / 200:.3f} ms/update (SUMO excluded)")
