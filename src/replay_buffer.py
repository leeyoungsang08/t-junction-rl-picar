import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, state_dim=12):
        self.capacity = int(capacity)
        self.position = self.size = 0
        self.arrays = {
            "states": np.zeros((capacity, state_dim), np.float32),
            "actions": np.zeros(capacity, np.int64),
            "rewards": np.zeros(capacity, np.float32),
            "next_states": np.zeros((capacity, state_dim), np.float32),
            "terminated": np.zeros(capacity, bool), "truncated": np.zeros(capacity, bool),
            "next_masks": np.ones((capacity, 2), bool),
        }

    def add(self, state, action, reward, next_state, terminated, truncated, next_mask):
        if not np.asarray(next_mask).any():
            raise ValueError("At least one action must be feasible")
        for k, v in zip(self.arrays, (state, action, reward, next_state, terminated, truncated, next_mask)):
            self.arrays[k][self.position] = v
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, rng):
        if self.size < count:
            raise ValueError("Insufficient replay samples")
        indices = rng.choice(self.size, count, replace=False)
        return {k: v[indices] for k, v in self.arrays.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "position": self.position, "size": self.size,
                "arrays": {k: v[:self.size].copy() for k, v in self.arrays.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        self.position, self.size = state["position"], state["size"]
        for k, v in state["arrays"].items():
            self.arrays[k][:self.size] = v
