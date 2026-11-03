from dataclasses import dataclass
import numpy as np
from src.signal_controller import action_mask


@dataclass
class SensorState:
    vehicles: list
    mean_waiting_seconds: list
    internal_vehicles: float
    phase: int
    green_elapsed: float
    transitioning: bool
    queues: list | None = None
    pending: list | None = None
    longest_wait: list | None = None
    recent_requests: list | None = None
    clearance_remaining: float = 0


EXTRA_GROUPS = ("queues", "pending", "longest_wait", "recent_requests")
FEATURES_V1 = [
    "vehicles_W", "vehicles_E", "vehicles_S", "mean_wait_W", "mean_wait_E", "mean_wait_S",
    "internal_vehicles", "phase_W", "phase_E", "phase_S", "green_elapsed", "transitioning",
]
FEATURES_V2 = FEATURES_V1 + [f"{key}_{d}" for key in EXTRA_GROUPS for d in "WES"] + ["clearance_remaining"]


class StateAdapter:
    def __init__(self, normalization, signal_config, schema=1):
        if schema not in (1, 2):
            raise ValueError(f"Unknown observation schema {schema}")
        self.schema = schema
        self.features = FEATURES_V2 if schema == 2 else FEATURES_V1
        self.normalization = normalization
        self.signal_config = signal_config
        if any(not np.isfinite(v) or v <= 0 for v in normalization.values()):
            raise ValueError("Normalization scales must be finite and positive")

    def encode(self, state):
        if isinstance(state, dict):
            state = SensorState(**state)
        values = np.asarray([*state.vehicles, *state.mean_waiting_seconds, state.internal_vehicles, state.green_elapsed], dtype=float)
        if len(state.vehicles) != 3 or len(state.mean_waiting_seconds) != 3 or values.shape != (8,):
            raise ValueError("Expected three directions in W, E, S order")
        if not np.isfinite(values).all() or (values < 0).any() or state.phase not in (0, 1, 2):
            raise ValueError("Missing, negative, non-finite values or invalid phase")
        n = self.normalization
        vector = np.r_[np.asarray(state.vehicles) / n["vehicles"],
                       np.asarray(state.mean_waiting_seconds) / n["waiting_seconds"],
                       state.internal_vehicles / n["internal_vehicles"],
                       np.eye(3)[state.phase], state.green_elapsed / n["green_seconds"], float(state.transitioning)]
        if self.schema == 1:
            return np.clip(vector, 0, 1).astype(np.float32)
        extras = []
        for key in EXTRA_GROUPS:
            group = np.asarray(getattr(state, key), dtype=float)
            if group.shape != (3,) or not np.isfinite(group).all() or (group < 0).any():
                raise ValueError(f"{key} must contain three nonnegative values in W, E, S order")
            extras.extend(group / n[key])
        remaining = float(state.clearance_remaining)
        if not np.isfinite(remaining) or remaining < 0:
            raise ValueError("Invalid clearance_remaining")
        vector = np.r_[vector, extras, remaining / (self.signal_config["yellow"] + self.signal_config["all_red"])]
        # Smoothly scale open-ended traffic counts and times without a hard ceiling.
        continuous = [*range(7), *range(12, 24)]
        vector[continuous] = 1 - 1 / (1 + vector[continuous])
        return vector.astype(np.float32)

    def mask(self, state):
        if isinstance(state, dict):
            state = SensorState(**state)
        self.encode(state)
        return action_mask(state.green_elapsed, state.transitioning, self.signal_config)
