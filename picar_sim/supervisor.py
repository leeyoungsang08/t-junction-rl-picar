"""Independent intersection supervisor: the ONLY component allowed to issue GO.

The same class runs inside SUMO (picar_sim.env) and on the real central PC (picar_sim.real_bridge).
The RL policy only *requests* a phase; the supervisor enforces timing, occupancy and link health.

Stages: green -> yellow (STOP new entries) -> clear (wait all_red_min AND occupancy == 0) -> next green.
FAULT: occupancy unknown/stale, clear timeout or link loss -> every approach STOP until healthy and empty.
MAX-WAIT GUARD (optional, signal.max_wait_guard_s > 0): once the current green has met its minimum,
an approach whose oldest request (first APPROACH report of a car still waiting there) is >= the limit
must be served next; the policy's choice is restricted to that switch. Uses only car status reports.
"""
from dataclasses import dataclass, field

import numpy as np

APPROACHES = ("W", "E", "N")
STAGES = ("green", "yellow", "clear")


@dataclass
class SupervisorStatus:
    phase: int
    stage: str
    green_elapsed: float
    clear_elapsed: float
    fault: str | None
    permits: dict = field(default_factory=dict)


class IntersectionSupervisor:
    def __init__(self, cfg, n_approaches=3):
        self.cfg = cfg
        self.mode = cfg.get("control_mode", "cycle")
        if self.mode not in ("cycle", "direct"):
            raise ValueError("control_mode must be cycle or direct")
        if not 0 < cfg["min_green_s"] <= cfg["max_green_s"]:
            raise ValueError("invalid green limits")
        self.n = n_approaches
        self.reset()

    def reset(self):
        self.phase, self.next_phase = 0, None
        self.stage = "green"
        self.green_elapsed = self.clear_elapsed = 0.0
        self.fault = None
        self.switches = self.forced_switches = self.fault_events = self.guard_switches = 0
        self.waits = None
        self.guard_target = None

    @property
    def n_actions(self):
        return 2 if self.mode == "cycle" else self.n

    # ---- action interface (policy side) ---------------------------------------------------
    def observe_waits(self, waits):
        """Oldest request age per approach (s), from car APPROACH reports. Call before mask()/request()."""
        self.waits = list(waits)

    def _starving(self):
        limit = self.cfg.get("max_wait_guard_s") or 0
        if limit <= 0 or self.waits is None:
            return None
        cand = [(w, i) for i, w in enumerate(self.waits[:self.n]) if i != self.phase and w >= limit]
        return max(cand)[1] if cand else None

    def mask(self):
        locked = self.stage != "green" or self.fault is not None or self.green_elapsed < self.cfg["min_green_s"]
        self.guard_target = None if locked else self._starving()
        if self.mode == "cycle":
            if locked:
                return np.array([True, False])
            if self.guard_target is not None:
                return np.array([False, True])
            if self.green_elapsed >= self.cfg["max_green_s"]:
                return np.array([False, True])
            return np.array([True, True])
        m = np.zeros(self.n, dtype=bool)
        if locked:
            m[self.phase] = True
        elif self.guard_target is not None:
            m[self.guard_target] = True
        elif self.green_elapsed >= self.cfg["max_green_s"]:
            m[:] = True
            m[self.phase] = False
        else:
            m[:] = True
        return m

    def request(self, action):
        action = int(action)
        if not self.mask()[action]:
            raise ValueError(f"action {action} not allowed; mask={self.mask().tolist()}")
        target = (self.phase + 1) % self.n if (self.mode == "cycle" and action == 1) else (
            action if self.mode == "direct" else self.phase)
        if target != self.phase:
            if self.guard_target is not None:
                self.guard_switches += 1
            self._start_switch(target)

    def _start_switch(self, target):
        self.next_phase, self.stage, self.clear_elapsed = target, "yellow", 0.0
        self.switches += 1

    # ---- time interface (runs every control tick) -----------------------------------------
    def tick(self, dt, occupancy, occupancy_age_s, link_age_s):
        """Advance by dt seconds. occupancy=None means unknown. Returns per-approach permits."""
        c = self.cfg
        healthy = occupancy is not None and occupancy_age_s <= c["occupancy_stale_s"] and link_age_s <= c["link_timeout_s"]
        if not healthy:
            if self.fault is None:
                self.fault_events += 1
            self.fault = "sensor_or_link"
        elif self.fault == "sensor_or_link" and occupancy == 0:
            self.fault = None
        if self.stage == "green":
            if self.fault is None:
                self.green_elapsed += dt
            if self.green_elapsed >= c["max_green_s"] and self.fault is None:
                self._start_switch((self.phase + 1) % self.n)
                self.forced_switches += 1
        elif self.stage == "yellow":
            self.clear_elapsed += dt
            if self.clear_elapsed >= c["yellow_s"] - 1e-9:
                self.stage, self.clear_elapsed = "clear", 0.0
        else:  # clear: wait for minimum all-red AND a measured empty box
            self.clear_elapsed += dt
            if self.clear_elapsed >= c["clear_timeout_s"] and (occupancy or 0) > 0:
                if self.fault is None:
                    self.fault_events += 1
                self.fault = "clear_timeout"
            if self.clear_elapsed >= c["all_red_min_s"] - 1e-9 and healthy and occupancy == 0:
                self.phase, self.next_phase = self.next_phase, None
                self.stage, self.green_elapsed, self.clear_elapsed = "green", 0.0, 0.0
                if self.fault == "clear_timeout":
                    self.fault = None
        return self.permits()

    def permits(self):
        go = self.stage == "green" and self.fault is None
        return {a: ("GO" if go and i == self.phase else "STOP") for i, a in enumerate(APPROACHES[:self.n])}

    def status(self):
        return SupervisorStatus(self.phase, self.stage, self.green_elapsed, self.clear_elapsed, self.fault, self.permits())

    def clearance_remaining(self):
        if self.stage == "yellow":
            return self.cfg["yellow_s"] - self.clear_elapsed + self.cfg["all_red_min_s"]
        if self.stage == "clear":
            return max(0.0, self.cfg["all_red_min_s"] - self.clear_elapsed)
        return 0.0
