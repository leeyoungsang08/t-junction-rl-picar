"""Experimental guarded traffic-signal arbitration, calibrated in SUMO only."""
from dataclasses import dataclass
from src.baselines import direct_phase_baseline
from src.guarded_policy import GuardedDirectPolicy

@dataclass(frozen=True)
class AdaptiveDecision:
    action: int
    proposed: int
    model_action: int
    proxy_action: int
    selected_source: str
    priority_target: int | None
    dominance: float

class AdaptiveDirectPolicy:
    def __init__(self, model_path, config, dominance_threshold=0.9, guard_seconds=80):
        if not 0 < dominance_threshold <= 1:
            raise ValueError("dominance_threshold must be in (0, 1]")
        self.base=GuardedDirectPolicy(model_path,guard_seconds)
        self.guard=self.base.guard
        self.config=config
        self.dominance_threshold=dominance_threshold

    def predict(self, info):
        state=info["raw_state"]
        model=self.base.predict(info)
        proxy=direct_phase_baseline("max_pressure_proxy",info,self.config)
        arrivals=state["recent_requests"]
        if arrivals is None or len(arrivals)!=3 or any(not isinstance(x,(int,float)) or x<0 for x in arrivals):
            raise ValueError("recent_requests must be three measured nonnegative values")
        total=sum(arrivals)
        concentration=max(arrivals)/total if total else 0
        source="rl" if total>=8 and concentration>=self.dominance_threshold else "proxy"
        selected=model.preferred if source=="rl" else proxy
        action,urgent=self.guard.choose(selected,list(info["action_mask"]),state["phase"],state["transitioning"],
                                        state["queues"],state["pending"])
        return AdaptiveDecision(action,selected,model.preferred,proxy,source,urgent,concentration)
