import numpy as np


def baseline_action(name, info, config):
    mask = info["action_mask"]
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
    else:
        raise ValueError(f"Unknown baseline {name}")
    p = state["phase"]
    other = [(p + 1) % 3, (p + 2) % 3]
    return int(max(waits[other]) >= config["baseline"]["starvation_seconds"]
               or q[other[0]] > q[p] + config["baseline"]["queue_margin"]
               or (q[p] == 0 and q[other].sum() > 0))
