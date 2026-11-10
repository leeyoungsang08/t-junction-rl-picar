"""Observable state shared by SUMO and the real PiCar central PC (schema "picar_v1").

Inputs are only things the real system can know:
  * per-car status events over UDP (APPROACH marker, ARRIVE = stopped at the stop line by the car's IR,
    IN = crossed the stop line, EXIT = rear passed the exit beam sensor),
  * the supervisor's own phase/stage/timers.
Unknown values are never silently replaced by zero: if occupancy is unknown the supervisor faults.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from picar_sim.supervisor import APPROACHES, STAGES

CAR_STATES = ("CRUISE", "APPROACH", "ARRIVE", "IN", "EXIT")
FEATURES = ([f"arrived_{a}" for a in APPROACHES] + [f"wait_{a}" for a in APPROACHES]
            + [f"approach_n_{a}" for a in APPROACHES] + [f"recent_{a}" for a in APPROACHES]
            + ["occupancy"] + [f"phase_{a}" for a in APPROACHES] + [f"stage_{s}" for s in STAGES]
            + ["green_elapsed", "clear_elapsed", "fault"] + [f"exit_free_{a}" for a in APPROACHES])
FEATURES_V2 = FEATURES + [f"oldest_request_{a}" for a in APPROACHES]   # picar_v2 = v1 + request age
SCHEMAS = {"picar_v1": FEATURES, "picar_v2": FEATURES_V2}
RECENT_WINDOW_S = 20.0


class CarLedger:
    """Central ledger of car states, fed by events (real UDP or SUMO virtual sensors)."""

    def __init__(self):
        self.cars = {}             # car -> dict(state, approach, t)
        self.recent = {a: deque() for a in APPROACHES}
        self.occupancy_updated = None

    def event(self, car, state, approach, t):
        if state not in CAR_STATES:
            raise ValueError(state)
        prev = self.cars.get(car, {}).get("state")
        if prev == state:
            return False
        if state == "APPROACH":
            self.recent[approach].append(t)
        entry = {"state": state, "approach": approach, "t": t}
        if state in ("APPROACH", "ARRIVE"):                  # request age: first APPROACH of this visit
            entry["request_t"] = self.cars[car].get("request_t", t) if prev in ("APPROACH", "ARRIVE") else t
        if state == "ARRIVE":
            entry["arrive_t"] = t
        elif prev == "ARRIVE" and state == "APPROACH":     # IR flicker: keep ARRIVE time
            entry["arrive_t"] = self.cars[car].get("arrive_t", t)
        self.cars[car] = entry
        return True

    def heartbeat_occupancy(self, t):
        """Called whenever the exit/occupancy sensor chain reports healthy."""
        self.occupancy_updated = t

    def occupancy(self):
        return sum(1 for c in self.cars.values() if c["state"] == "IN")

    def snapshot(self, t):
        for a in APPROACHES:
            q = self.recent[a]
            while q and q[0] < t - RECENT_WINDOW_S:
                q.popleft()
        arrived, wait, n, recent, oldest = [], [], [], [], []
        for a in APPROACHES:
            here = [c for c in self.cars.values() if c["approach"] == a and c["state"] in ("APPROACH", "ARRIVE")]
            arr = [c["arrive_t"] for c in here if c["state"] == "ARRIVE"]
            arrived.append(1.0 if arr else 0.0)
            wait.append(t - min(arr) if arr else 0.0)
            n.append(float(len(here)))
            oldest.append(t - min(c["request_t"] for c in here) if here else 0.0)
            recent.append(float(len(self.recent[a])))
        age = float("inf") if self.occupancy_updated is None else t - self.occupancy_updated
        return {"arrived": arrived, "wait": wait, "approach_n": n, "recent": recent, "oldest_request": oldest,
                "occupancy": float(self.occupancy()), "occupancy_age": age}


class SegmentTracker:
    """Cars per road segment between the two T stop lines, from junction events only.

    REGISTER(edge) at start, IN(next=edge the car will take), EXIT(edge = exit-beam arm it passed).
    Used for "don't block the box": a car may enter only if its next segment has room.
    """

    def __init__(self, capacity):
        self.cap = dict(capacity)
        self.seg, self.target = {}, {}

    def event(self, e):
        car, st = e["car"], e["state"]
        if st == "REGISTER":
            self.seg[car], self.target[car] = e["edge"], None
        elif st == "IN":
            self.seg[car], self.target[car] = None, e["next"]
        elif st == "EXIT":
            self.seg[car], self.target[car] = e["edge"], None

    def free(self, edge):
        used = sum(1 for s in self.seg.values() if s == edge) + sum(1 for s in self.target.values() if s == edge)
        return self.cap[edge] - used


@dataclass
class Encoder:
    max_green_s: float
    clear_timeout_s: float
    schema: str = "picar_v1"

    @property
    def size(self):
        return len(SCHEMAS[self.schema])

    def encode(self, snap, status, exit_free):
        s = lambda x, k: np.asarray(x, float) / (np.asarray(x, float) + k)   # smooth, unbounded inputs
        if len(snap["wait"]) != 3 or min(snap["wait"]) < 0 or not np.isfinite(snap["wait"]).all():
            raise ValueError("invalid wait values")
        v = np.concatenate([
            snap["arrived"], s(snap["wait"], 10.0), s(snap["approach_n"], 2.0), s(snap["recent"], 3.0),
            [snap["occupancy"] / (snap["occupancy"] + 1.0)],
            np.eye(3)[status.phase], [1.0 if status.stage == st else 0.0 for st in STAGES],
            [min(status.green_elapsed / self.max_green_s, 1.0), min(status.clear_elapsed / self.clear_timeout_s, 1.0),
             1.0 if status.fault else 0.0],
            s(np.maximum(np.asarray(exit_free, float), 0.0), 2.0)])
        if self.schema == "picar_v2":
            v = np.concatenate([v, s(snap["oldest_request"], 10.0)])
        return v.astype(np.float32)
