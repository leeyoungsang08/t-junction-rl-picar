import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from src.dqn_network import QNetwork
from src.replay_buffer import ReplayBuffer


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def masked_bootstrap(online_values, target_values, masks, double_dqn=False):
    if not masks.any(dim=1).all():
        raise ValueError("Empty target action mask")
    if double_dqn:
        actions = online_values.masked_fill(~masks, -torch.inf).argmax(dim=1)
        return target_values.gather(1, actions[:, None]).squeeze(1)
    return target_values.masked_fill(~masks, -torch.inf).max(dim=1).values


class DQNAgent:
    def __init__(self, config):
        self.config = config
        torch.set_num_threads(config.get("torch_threads", 1))
        device = config["device"]
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.rng = np.random.default_rng(seed)
        state_dim = config.get("state_dim", 12)
        self.actions = int(config.get("action_dim", 2))
        self.online = QNetwork(config["hidden"], state_dim, self.actions, config.get("dueling", False)).to(self.device)
        self.target = QNetwork(config["hidden"], state_dim, self.actions, config.get("dueling", False)).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.replay = ReplayBuffer(config["replay_capacity"], state_dim, actions=self.actions,
                                   prioritized=config.get("prioritized_replay", False),
                                   alpha=config.get("per_alpha", 0.6), epsilon=config.get("per_epsilon", 1e-5))
        self.global_step = self.updates = 0

    @property
    def epsilon(self):
        c = self.config
        f = min(1, self.global_step / c["epsilon_decay_steps"])
        return c["epsilon_start"] + f * (c["epsilon_end"] - c["epsilon_start"])

    def select(self, state, mask, explore=True):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.actions,) or not mask.any():
            raise ValueError("Invalid action mask")
        with torch.inference_mode():
            q = self.online(torch.as_tensor(state, device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()
        if explore and self.rng.random() < self.epsilon:
            action = int(self.rng.choice(np.flatnonzero(mask)))
        else:
            action = int(np.where(mask, q, -np.inf).argmax())
        return action, q

    def update(self):
        c = self.config
        if self.replay.size < max(c["batch_size"], c["learning_start"]) or self.global_step % c["update_every"]:
            return None
        beta_start = c.get("per_beta_start", 0.4)
        beta_end = c.get("per_beta_end", 1.0)
        beta_steps = max(int(c.get("per_beta_steps", 100000)), 1)
        beta = beta_start + (beta_end - beta_start) * min(1.0, self.global_step / beta_steps)
        batch = self.replay.sample(c["batch_size"], self.rng, beta=beta)
        b = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items() if k != "indices"}
        q = self.online(b["states"]).gather(1, b["actions"][:, None]).squeeze(1)
        with torch.no_grad():
            done = b["terminated"] if c["bootstrap_truncated"] else b["terminated"] | b["truncated"]
            live = ~done
            target = b["rewards"].clone()
            if live.any():
                states = b["next_states"][live]
                target_values = self.target(states)
                online_values = self.online(states) if c["double_dqn"] else target_values
                next_q = masked_bootstrap(online_values, target_values, b["next_masks"][live], c["double_dqn"])
                target[live] += c["gamma"] ** b["duration_steps"][live] * next_q
        td_error = target - q
        per_item_loss = torch.nn.functional.smooth_l1_loss(q, target, reduction="none")
        loss = (b["weights"] * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), c["gradient_clip"])
        self.optimizer.step()
        self.replay.update_priorities(batch["indices"], td_error.detach().abs().cpu().numpy())
        self.updates += 1
        if self.updates % c["target_update_steps"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"global_step": self.global_step, "update": self.updates, "loss": float(loss.item()),
                "q_mean": float(q.mean().item()), "q_std": float(q.std(unbiased=False).item()), "gradient_norm": float(norm)}

    def checkpoint(self, next_episode, extra):
        return {"online": self.online.state_dict(), "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict(),
                "config": self.config, "global_step": self.global_step, "updates": self.updates, "epsilon": self.epsilon,
                "next_episode": next_episode, "replay": self.replay.state_dict(),
                "rng": self.rng.bit_generator.state, "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "extra": extra}

    def restore(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"] != self.config:
            raise ValueError("Training config differs from checkpoint; resume with the original config")
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_step, self.updates = state["global_step"], state["updates"]
        self.replay.load_state_dict(state["replay"])
        self.rng.bit_generator.state = state["rng"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda_rng"]])
        return state
