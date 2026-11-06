import numpy as np


def baseline_action(name, info, config):
    mask = info["action_mask"]
    if len(mask) == 3:
        return direct_phase_baseline(name, info, config)
    if not mask[1]:
        return 0
    if not mask[0]:
        return 1
    state = info["raw_state"]
    if name == "fixed_time":
        return int(state["green_elapsed"] >= config["signal"]["fixed_green"])
    if name == "rule_observed":
        q = np.asarray(state["queues"]) + np.asarray(state["pending"])
        waits = np.asarray(state["longest_wait"])
    elif name == "rule_based":
        q = np.asarray(info["queues"])
        waits = np.asarray(info["max_direction_wait"])
    elif name == "max_pressure_proxy":
        q = np.asarray(state["queues"]) + np.asarray(state["pending"])
        waits = np.zeros(3)
    else:
        raise ValueError(f"Unknown baseline {name}")
    p = state["phase"]
    other = [(p + 1) % 3, (p + 2) % 3]
    return int(max(waits[other]) >= config["baseline"]["starvation_seconds"]
               or q[other[0]] > q[p] + config["baseline"]["queue_margin"]
               or (q[p] == 0 and q[other].sum() > 0))


def direct_phase_baseline(name, info, config):
    """Safety-masked direction request; outgoing links are uncongested in this toy map.

    This is an *upstream queue proxy*, NOT full downstream-aware Max-Pressure.
    """
    mask = np.asarray(info["action_mask"], dtype=bool)
    raw = info["raw_state"]
    current = int(raw["phase"])
    if mask.sum() == 1:
        return int(np.flatnonzero(mask)[0])
    if name == "fixed_time":
        if mask[current] and raw["green_elapsed"] < config["signal"]["fixed_green"]:
            return current
        return (current + 1) % 3
    if name in ("max_pressure_proxy", "rule_based", "rule_observed"):
        queues = np.asarray(raw["queues"], dtype=float) + np.asarray(raw["pending"], dtype=float)
        if name in ("rule_based", "rule_observed"):
            waits = np.asarray(raw["longest_wait"], dtype=float)
            scores = queues + 0.05 * waits
        else:
            scores = queues
        scores = np.where(mask, scores, -np.inf)
        best = int(np.argmax(scores))
        # Prefer the current phase when there is no advantage to switching.
        if mask[current] and scores[current] >= scores[best]:
            return current
        return best
    raise ValueError(f"Unknown direct phase baseline {name}")
