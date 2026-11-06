from numbers import Integral


class DirectionServiceGuard:
    """Give the oldest continuously backlogged red direction the next legal green."""

    def __init__(self, threshold_s):
        if isinstance(threshold_s, bool) or not isinstance(threshold_s, Integral) or threshold_s <= 0:
            raise ValueError("threshold_s must be a positive integer")
        self.threshold_s = threshold_s
        self.reset()

    def reset(self):
        self.ages = (0, 0, 0)

    def observe_second(self, phase, stage, queues, pending):
        """Use the signal actually shown during the tick and the post-step traffic."""
        self.ages = tuple(
            age + 1 if (stage != "green" or d != phase) and queues[d] + pending[d] > 0 else 0
            for d, age in enumerate(self.ages)
        )

    def choose(self, preferred, mask, phase, transitioning, queues, pending):
        """Preserve existing clearance and min/max-green masking."""
        if len(mask) != 3 or not 0 <= preferred < 3 or not mask[preferred]:
            raise ValueError("preferred action must be legal under the three-phase mask")
        eligible = [
            d for d in range(3)
            if d != phase and mask[d] and self.ages[d] >= self.threshold_s and queues[d] + pending[d] > 0
        ]
        if transitioning or not eligible:
            return preferred, None
        # Current queue breaks same-age ties. Cyclic order is a stable last tie-break.
        target = max(eligible, key=lambda d: (self.ages[d], queues[d] + pending[d], -((d - phase) % 3)))
        return target, target
