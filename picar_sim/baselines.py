"""Non-learning controllers. They use exactly the same observable snapshot as the DQN."""
import numpy as np


def fixed_time(snap, status, mask, mode, green_s=5.0):
    """Fixed green; 5 s was the best of {5, 8, 12} s on the validation split."""
    if mask.sum() == 1:
        return int(np.flatnonzero(mask)[0])
    if mode == "cycle":
        return 1 if status.green_elapsed >= green_s else 0
    return status.phase if status.green_elapsed < green_s else (status.phase + 1) % len(mask)


def queue_rule(snap, status, mask, mode, starvation_s=20.0, margin=2):
    """Serve demand: leave an empty approach, rescue starving approaches, follow the larger queue.
    margin=2 was the best of {0,1,2} on the validation split (same score as DQN model selection).
    J2 uses this rule with the same defaults."""
    if mask.sum() == 1:
        return int(np.flatnonzero(mask)[0])
    n = np.asarray(snap["approach_n"], float)
    wait = np.asarray(snap["wait"], float)
    p = status.phase
    if mode == "direct":
        score = np.where(mask, n + 0.1 * wait + 100 * (wait >= starvation_s), -np.inf)
        best = int(np.argmax(score))
        return p if mask[p] and score[p] >= score[best] - margin else best
    others = [(p + 1) % 3, (p + 2) % 3]
    switch = (n[p] == 0 and n[others].sum() > 0) or wait[others].max() >= starvation_s or n[others[0]] > n[p] + margin
    return int(switch)


CONTROLLERS = {"fixed_time": fixed_time, "queue_rule": queue_rule}
