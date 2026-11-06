import numpy as np


class ReplayBuffer:
    """Uniform or proportional prioritized replay with importance-sampling weights.

    Kept compatible with old replay checkpoints which have no priorities field.
    """

    def __init__(self, capacity, state_dim=12, actions=2, prioritized=False, alpha=0.6, epsilon=1e-5):
        self.capacity = int(capacity)
        self.actions = int(actions)
        self.prioritized = bool(prioritized)
        self.alpha = float(alpha)
        self.priority_epsilon = float(epsilon)
        if self.capacity < 1 or self.actions < 2 or not 0 <= self.alpha <= 1 or self.priority_epsilon <= 0:
            raise ValueError("Invalid replay configuration")
        self.position = self.size = 0
        self.arrays = {
            "states": np.zeros((capacity, state_dim), np.float32),
            "actions": np.zeros(capacity, np.int64),
            "rewards": np.zeros(capacity, np.float32),
            "next_states": np.zeros((capacity, state_dim), np.float32),
            "terminated": np.zeros(capacity, bool), "truncated": np.zeros(capacity, bool),
            "next_masks": np.ones((capacity, self.actions), bool),
            "duration_steps": np.ones(capacity, np.float32),
        }
        self.priorities = np.ones(self.capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, terminated, truncated, next_mask, duration_steps=1.0):
        mask = np.asarray(next_mask, dtype=bool)
        if mask.shape != (self.actions,) or (not mask.any() and not (terminated or truncated)):
            raise ValueError("Invalid next-action mask")
        if not np.isfinite(duration_steps) or duration_steps <= 0:
            raise ValueError("Transition duration must be finite and positive")
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.actions:
            raise ValueError("Action outside replay action space")
        for k, v in zip(self.arrays, (state, action, reward, next_state, terminated, truncated, mask, duration_steps)):
            self.arrays[k][self.position] = v
        if self.prioritized:
            maximum = float(self.priorities[:self.size].max()) if self.size else 1.0
            self.priorities[self.position] = max(maximum, self.priority_epsilon)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, rng, beta=0.4):
        if self.size < count:
            raise ValueError("Insufficient replay samples")
        if self.prioritized:
            if not 0 <= beta <= 1:
                raise ValueError("PER beta must be in [0, 1]")
            scaled = self.priorities[:self.size].astype(np.float64) ** self.alpha
            probabilities = scaled / scaled.sum()
            indices = rng.choice(self.size, count, replace=True, p=probabilities)
            weights = (self.size * probabilities[indices]) ** (-beta)
            weights /= (self.size * probabilities.min()) ** (-beta)
        else:
            indices = rng.choice(self.size, count, replace=False)
            weights = np.ones(count, dtype=np.float32)
        return {**{k: v[indices] for k, v in self.arrays.items()},
                "indices": np.asarray(indices, dtype=np.int64), "weights": np.asarray(weights, dtype=np.float32)}

    def update_priorities(self, indices, absolute_errors):
        if not self.prioritized:
            return
        indices = np.asarray(indices, dtype=np.int64)
        errors = np.asarray(absolute_errors, dtype=np.float32)
        if indices.shape != errors.shape or not np.all(np.isfinite(errors)) or np.any(errors < 0):
            raise ValueError("Invalid TD errors")
        unique, inverse = np.unique(indices, return_inverse=True)
        maximum = np.zeros(len(unique), dtype=np.float32)
        np.maximum.at(maximum, inverse, errors)
        self.priorities[unique] = maximum + self.priority_epsilon

    def state_dict(self):
        return {"capacity": self.capacity, "position": self.position, "size": self.size,
                "actions": self.actions, "prioritized": self.prioritized, "alpha": self.alpha,
                "priority_epsilon": self.priority_epsilon,
                "priorities": self.priorities[:self.size].copy(),
                "arrays": {k: v[:self.size].copy() for k, v in self.arrays.items()}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state.get("actions", 2) != self.actions or state.get("prioritized", False) != self.prioritized:
            raise ValueError("Replay configuration differs from checkpoint")
        self.position, self.size = state["position"], state["size"]
        for k, v in state["arrays"].items():
            if self.arrays[k][:self.size].shape != v.shape:
                raise ValueError(f"Replay shape mismatch: {k}")
            self.arrays[k][:self.size] = v
        if "priorities" in state:
            self.priorities[:self.size] = state["priorities"]
        elif self.prioritized:
            raise ValueError("Prioritized checkpoint lacks priorities")
