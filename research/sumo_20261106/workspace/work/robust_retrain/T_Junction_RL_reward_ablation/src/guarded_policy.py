from dataclasses import dataclass
from src.inference import Policy
from src.service_guard import DirectionServiceGuard


@dataclass(frozen=True)
class GuardedDecision:
    action: int
    preferred: int
    urgent: int | None
    ages: tuple[int, int, int]

    @property
    def overridden(self):
        return self.action != self.preferred


class GuardedDirectPolicy:
    """Wrap a frozen Direct checkpoint. Use observer=policy.guard when constructing the env."""

    def __init__(self, model_path, threshold_s):
        self.base = Policy(model_path)
        if self.base.payload.get("action_dim") != 3:
            raise ValueError("GuardedDirectPolicy requires a three-action Direct checkpoint")
        self.guard = DirectionServiceGuard(threshold_s)

    def predict(self, info):
        state = info["raw_state"]
        base = self.base.predict(state)
        mask = list(info["action_mask"])
        if base["valid_actions"] != mask:
            raise ValueError("Model and environment masks differ")
        preferred = base["action_id"]
        action, urgent = self.guard.choose(preferred, mask, state["phase"], state["transitioning"],
                                          state["queues"], state["pending"])
        return GuardedDecision(action, preferred, urgent, self.guard.ages)
