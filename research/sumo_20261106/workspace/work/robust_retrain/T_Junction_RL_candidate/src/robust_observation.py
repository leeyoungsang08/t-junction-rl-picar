from copy import deepcopy
import numpy as np


def perturb_raw(raw_state, random_seed, step, noise=True):
    """Deterministic independent measurement jitter; signal phase and mask stay exact."""
    state = deepcopy(raw_state)
    if not noise:
        return state
    rng = np.random.default_rng(np.random.SeedSequence([int(random_seed),int(step),97531]))
    for key in ("vehicles", "queues", "pending", "recent_requests"):
        values = np.asarray(state[key], dtype=float)
        # Stochastic rounding retains the intended mean while allowing errors
        # for low counts; nearest-integer rounding makes 1, 2 and 3 exact.
        perturbed = np.maximum(0, values * rng.uniform(.85, 1.15, 3))
        lower = np.floor(perturbed)
        state[key] = (lower + (rng.random(3) < perturbed - lower)).tolist()
    for key in ("mean_waiting_seconds", "longest_wait"):
        state[key] = (np.asarray(state[key], dtype=float) * rng.uniform(.85,1.15,3)).tolist()
    state["internal_vehicles"] = max(0.0, float(state["internal_vehicles"]) * float(rng.uniform(.85,1.15)))
    return state


def observe(info, adapter, random_seed, step, noise=True):
    return adapter.encode(perturb_raw(info["raw_state"], random_seed, step, noise))


def tail_score(values, fraction=0.2):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not np.isfinite(values).all() or not 0 < fraction <= 1:
        raise ValueError("tail score requires finite values and fraction in (0,1]")
    count = max(1, int(np.ceil(len(values) * fraction)))
    return .5*float(values.mean()) + .5*float(values[-count:].mean())
