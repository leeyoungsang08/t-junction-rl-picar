"""SUMO environment of the PiCar course: N recirculating cars, two T junctions.

J  (bottom) : GO/STOP requested by the RL policy, enforced by IntersectionSupervisor.
J2 (top)    : same supervisor class driven by a fixed queue rule ("나머지는 고정 규칙").
Both junctions apply "don't block the box": a car gets GO only if its next segment has room
(SegmentTracker, fed by the same junction events a real central PC receives).

The policy never sees SUMO ground truth: observations come from CarLedger/SegmentTracker fed by
virtual sensors at the real sensor positions (APPROACH marker, stop-line IR, exit beam per arm).
Ground truth is used only for reward and evaluation metrics.
"""
import os
import uuid
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from picar_sim.baselines import queue_rule
from picar_sim.map_builder import APPROACHES, PKG, build, load_config, segment_capacity
from picar_sim.observation import CarLedger, Encoder, SegmentTracker
from picar_sim.supervisor import IntersectionSupervisor

BIAS = {   # arm weights when choosing the next road at each T
    "uniform": {"J": {}, "J2": {}},
    "connector_heavy": {"J": {"N": 3}, "J2": {"N": 3}},
    "west_heavy": {"J": {"N": 2}, "J2": {"W": 4}},
    "east_heavy": {"J": {"N": 2}, "J2": {"E": 4}},
}
JUNCTIONS = ("J", "J2")


def bias_at(name, t, horizon):
    if name == "alternating":
        return BIAS[("west_heavy", "east_heavy", "connector_heavy")[min(2, int(3 * t / horizon))]]
    return BIAS[name]


class PicarEnv:
    def __init__(self, cfg=None, out_dir=None, randomize=None, backend=None, gui=False, log_decisions=False,
                 faults=None, control_mode=None):
        self.cfg = cfg or load_config()
        if control_mode:
            self.cfg["signal"]["control_mode"] = control_mode
        self.net = PKG / "maps/picar_course.net.xml"
        if not self.net.exists():
            build(self.cfg)
        self.out_dir = Path(out_dir) if out_dir else None
        self.randomize = self.cfg["randomization"]["enabled"] if randomize is None else randomize
        self.backend = backend or self.cfg["simulation"]["backend"]
        self.gui = gui
        self.log_decisions = log_decisions
        self.faults = faults or []        # [(t0, t1, "occupancy" | "link")] for fault-injection tests
        sc = self.cfg["signal"]
        self.sup = IntersectionSupervisor(sc)                                   # RL-controlled J
        self.sup2 = IntersectionSupervisor({**sc, "control_mode": "cycle"})    # fixed-rule J2
        self.encoder = Encoder(sc["max_green_s"], sc["clear_timeout_s"])
        self.obs_size, self.n_actions = self.encoder.size, self.sup.n_actions
        self.k = None
        self.auto_route = True        # closed-loop recirculation (calibration scripts switch it off)
        net = sumolib.net.readNet(str(self.net))
        self.lane_len = {f"{a}_{d}": net.getEdge(f"{a}_{d}").getLane(0).getLength()
                         for a in APPROACHES for d in ("in", "out")}
        self.capacity = segment_capacity(self.lane_len, self.cfg["vehicle"], self.cfg["course"])

    # ------------------------------------------------------------------ setup
    def _sample(self, key, rng):
        lo, hi = self.cfg["randomization"][key]
        return float(rng.uniform(lo, hi)) if self.randomize else None

    def _write_routes(self, path, rng):
        v = self.cfg["vehicle"]
        root = ET.Element("routes")
        self.car_speed, self.initial_edge = {}, {}
        slots = []          # (edge, pos) with >= 0.45 m spacing, clear of both junction boxes
        for e, L in self.lane_len.items():
            slots += [(e, p) for p in np.arange(0.35, L - 0.25, 0.45)][:max(0, self.capacity[e] - 1)]
        chosen = rng.choice(len(slots), size=self.n_cars, replace=False)
        # one routing stream per car -> identical turn choices for every controller (paired evaluation)
        self.route_rng = {f"car{i}": np.random.default_rng([self.seed, i]) for i in range(self.n_cars)}
        for i, idx in enumerate(sorted(chosen)):
            e, pos = slots[idx]
            speed = self._sample("speed_mps", rng) or v["speed_mps"]
            attrs = {"id": f"t{i}", "length": str(v["length_m"]), "width": str(v["width_m"]), "minGap": str(v["min_gap_m"]),
                     "maxSpeed": f"{speed:.4f}", "accel": f"{self._sample('accel_mps2', rng) or v['accel_mps2']:.4f}",
                     "decel": f"{self._sample('decel_mps2', rng) or v['decel_mps2']:.4f}", "emergencyDecel": "3",
                     "tau": str(v["tau_s"]), "sigma": "0", "speedDev": "0", "guiShape": "passenger",
                     "jmStoplineGap": "0.02",   # SUMO default 1 m would stop cars 1 m before the stop line
                     "startupDelay": f"{self._sample('startup_delay_s', rng) or v['startup_delay_s']:.3f}"}
            ET.SubElement(root, "vType", attrs)
            car = f"car{i}"
            nxt = self._next_edge(e, 0.0, self.route_rng[car])
            self.car_speed[car], self.initial_edge[car] = speed, e
            veh = ET.SubElement(root, "vehicle", {"id": car, "type": f"t{i}", "depart": "0", "departPos": f"{pos:.3f}",
                                                  "departSpeed": "0", "departLane": "0"})
            ET.SubElement(veh, "route", edges=f"{e} {nxt}")
        ET.indent(root)
        ET.ElementTree(root).write(path, encoding="utf-8")

    def _next_edge(self, edge, t, rng):
        arm, kind = edge.split("_")
        node = "J" if kind == "in" else "J2"
        options = [b for b in APPROACHES if b != arm]
        w = np.array([bias_at(self.bias, t, self.horizon)[node].get(b, 1) for b in options], float)
        b = options[rng.choice(len(options), p=w / w.sum())]
        return f"{b}_out" if kind == "in" else f"{b}_in"

    def reset(self, seed, cars, bias="uniform", horizon=None):
        self.close()
        self.seed, self.n_cars, self.bias = int(seed), int(cars), bias
        sim = self.cfg["simulation"]
        self.dt, self.horizon = sim["step_s"], float(horizon or sim["horizon_s"])
        self.substeps = int(round(sim["decision_s"] / self.dt))
        self.rng = np.random.default_rng(self.seed)
        run_dir = self.out_dir or Path(os.environ.get("TMPDIR", "/tmp")) / "picar_sim"
        run_dir.mkdir(parents=True, exist_ok=True)
        tag = uuid.uuid4().hex[:8]
        self.route_file = run_dir / f"routes_{self.seed}_{tag}.rou.xml"
        self._write_routes(self.route_file, self.rng)
        self.cmd_latency = self._sample("command_latency_s", self.rng) or 0.0
        self.obs_delay = self._sample("observation_delay_s", self.rng) or 0.0
        self.ir_prob = self._sample("ir_detect_prob", self.rng) or 1.0
        self.loss_prob = self._sample("status_loss_prob", self.rng) or 0.0
        binary = sumolib.checkBinary("sumo-gui" if self.gui else "sumo")
        cmd = [binary, "-n", str(self.net), "-r", str(self.route_file), "--step-length", str(self.dt),
               "--seed", str(self.seed), "--collision.check-junctions", "true", "--collision.action", "warn",
               "--time-to-teleport", "-1", "--no-step-log", "true", "--no-warnings", "true",
               "--duration-log.disable", "true", "--default.action-step-length", str(self.dt)]
        if self.gui:
            cmd += ["--start", "--delay", "50"]
        if self.backend == "libsumo" and not self.gui:
            import libsumo as k
            k.start(cmd)
        else:
            import traci as k
            k.start(cmd, label=tag)
        self.k = k
        self.links = {jn: [(lk[0][0].split("_")[0], lk[0][1].rsplit("_", 1)[0]) for lk in k.trafficlight.getControlledLinks(jn)]
                      for jn in JUNCTIONS}
        self.sup.reset()
        self.sup2.reset()
        self.ledgers = {jn: CarLedger() for jn in JUNCTIONS}
        self.ledger = self.ledgers["J"]
        self.tracker = SegmentTracker(self.capacity)
        self.t = 0.0
        self.truth = {}               # car -> (junction, state, arm)
        self.in_arm = {}              # car -> approach it entered its current junction from
        self.retry = set()            # cars whose last status packet was lost / IR missed
        self.event_queue = deque()    # (deliver_t, event)
        self.tls_queue = deque()      # (apply_t, {jn: state})
        self.applied = {jn: "r" * len(self.links[jn]) for jn in JUNCTIONS}
        self.requests, self.delays = {}, {a: [] for a in APPROACHES}
        self.n_requests = {a: 0 for a in APPROACHES}
        self.crossings = self.crossings_j2 = 0
        self.collisions = self.conflict_steps = self.stopline_violations = 0
        self.still_since = 0.0
        self.decisions, self.event_log, self.obs_log = [], [], []
        self.deadlock = False
        self.link_seen = 0.0
        for car, e in self.initial_edge.items():       # cars register their start segment with the central PC
            self._queue_event({"car": car, "junction": None, "state": "REGISTER", "lane": None, "edge": e}, force=True)
        self._deliver()
        for _ in range(2):            # let vehicles insert
            self._substep({"J": self.sup.permits(), "J2": self.sup2.permits()})
        return self._obs(), self._info()

    # ------------------------------------------------------------------ sensing (virtual sensors)
    def _fault(self, kind):
        return any(t0 <= self.t < t1 and k == kind for t0, t1, k in self.faults)

    def _truth_state(self, car, road, pos, spd):
        """-> (junction, state, arm). APPROACH/ARRIVE before a stop line, IN inside a box (until the rear
        passes that arm's exit beam), CRUISE on a segment."""
        c, length = self.cfg["course"], self.cfg["vehicle"]["length_m"]
        if road.startswith(":J2_"):
            return "J2", "IN", self.in_arm.get(car)
        if road.startswith(":J_"):
            return "J", "IN", self.in_arm.get(car)
        arm, kind = road.split("_")
        down = "J" if kind == "in" else "J2"     # junction this edge leads to
        up = "J2" if kind == "in" else "J"       # junction this edge comes from
        if pos - length < c["exit_line_m"] and self.truth.get(car, (None, None))[:2] == (up, "IN"):
            return up, "IN", self.in_arm.get(car)
        L = self.lane_len[road]
        if pos >= L - c["arrive_window_m"] and spd < 0.02:
            return down, "ARRIVE", arm
        if pos >= L - c["approach_zone_m"]:
            return down, "APPROACH", arm
        return None, "CRUISE", None

    def _sense(self):
        k = self.k
        moving = False
        waiting, front_wait = 0, {a: 0.0 for a in APPROACHES}
        in_arms = {"J": set(), "J2": set()}
        for car in k.vehicle.getIDList():
            road, pos, spd = k.vehicle.getRoadID(car), k.vehicle.getLanePosition(car), k.vehicle.getSpeed(car)
            moving |= spd > 0.01
            route = k.vehicle.getRoute(car)
            if self.auto_route and road == route[-1]:
                k.vehicle.setRoute(car, [road, self._next_edge(road, self.t, self.route_rng[car])])
                route = k.vehicle.getRoute(car)
            jn, state, arm = self._truth_state(car, road, pos, spd)
            pj, ps, parm = self.truth.get(car, (None, "CRUISE", None))
            if (jn, state) != (pj, ps):
                if ps == "IN":                                   # rear passed the exit beam of pj
                    self._queue_event({"car": car, "junction": pj, "state": "EXIT", "lane": None, "edge": road}, force=True)
                if state == "IN":
                    arm = parm if ps in ("APPROACH", "ARRIVE") else self.in_arm.get(car)
                    self.in_arm[car] = arm
                    self._entered(car, jn, arm, route[-1])
                elif state in ("APPROACH", "ARRIVE"):
                    self._queue_event({"car": car, "junction": jn, "state": state, "lane": arm, "next": route[-1]})
            if jn == "J" and state in ("APPROACH", "ARRIVE"):
                if car not in self.requests:
                    self.requests[car] = (arm, self.t)
                    self.n_requests[arm] += 1
                if spd < 0.02:
                    waiting += 1
                    front_wait[arm] = max(front_wait[arm], self.t - self.requests[car][1])
            if state == "IN":
                in_arms[jn].add(self.in_arm.get(car))
            self.truth[car] = (jn, state, arm)
        for jn in in_arms:
            in_arms[jn].discard(None)
            if len(in_arms[jn]) > 1:
                self.conflict_steps += 1
        self.still_since = self.t if moving else self.still_since
        return waiting, front_wait

    def _entered(self, car, jn, arm, target):
        self._queue_event({"car": car, "junction": jn, "state": "IN", "lane": arm, "next": target}, force=True)
        idx = [i for i, (a, _) in enumerate(self.links[jn]) if a == arm]
        if idx and all(self.applied[jn][i] not in "Gy" for i in idx):
            self.stopline_violations += 1
        if jn == "J2":
            self.crossings_j2 += 1
            return
        if car in self.requests:
            a, t0 = self.requests.pop(car)
            free = self.cfg["course"]["approach_zone_m"] / self.car_speed[car]
            self.delays[a].append(max(0.0, self.t - t0 - free))
            self.crossings += 1

    def _queue_event(self, e, force=False):
        """Status packet from a car / exit beam. ARRIVE may be missed by the IR and APPROACH/ARRIVE
        packets may be lost; they are re-sent on the next 0.1 s step while the state persists.
        IN/EXIT/REGISTER are treated as acknowledged (reliable) messages."""
        if not force:
            if (e["state"] == "ARRIVE" and self.rng.random() > self.ir_prob) or self.rng.random() < self.loss_prob:
                self.retry.add(e["car"])
                return
        self.retry.discard(e["car"])
        self.event_queue.append((self.t + self.obs_delay, e))

    def _resend_lost(self):
        for car in list(self.retry):
            jn, state, arm = self.truth.get(car, (None, "CRUISE", None))
            if state in ("APPROACH", "ARRIVE"):
                self._queue_event({"car": car, "junction": jn, "state": state, "lane": arm,
                                   "next": self.k.vehicle.getRoute(car)[-1]})
            else:
                self.retry.discard(car)

    def _deliver(self):
        while self.event_queue and self.event_queue[0][0] <= self.t + 1e-9:
            _, e = self.event_queue.popleft()
            e = {**e, "t": self.t}
            self.event_log.append(e)
            self.tracker.event(e)
            if e["junction"] in self.ledgers:
                self.ledgers[e["junction"]].event(e["car"], e["state"], e["lane"], self.t)
        if not self._fault("occupancy"):
            for led in self.ledgers.values():
                led.heartbeat_occupancy(self.t)
        if not self._fault("link"):
            self.link_seen = self.t

    # ------------------------------------------------------------------ stepping
    def _tls(self, jn, permits, sup):
        st = sup.status()
        chars = []
        for arm, target in self.links[jn]:
            if permits[arm] == "GO" and self.tracker.free(target) > 0:
                chars.append("G")
            elif arm == APPROACHES[st.phase] and (st.stage == "yellow" or st.fault or permits[arm] == "GO"):
                chars.append("y")       # stop if possible; a car already committed may finish crossing
            else:
                chars.append("r")
        return "".join(chars)

    def _substep(self, permits):
        k = self.k
        self.tls_queue.append((self.t + self.cmd_latency, {"J": self._tls("J", permits["J"], self.sup),
                                                           "J2": self._tls("J2", permits["J2"], self.sup2)}))
        while self.tls_queue and self.tls_queue[0][0] <= self.t + 1e-9:
            self.applied = self.tls_queue.popleft()[1]
        for jn in JUNCTIONS:
            k.trafficlight.setRedYellowGreenState(jn, self.applied[jn])
        k.simulationStep()
        self.t = round(self.t + self.dt, 6)
        self.collisions += k.simulation.getCollidingVehiclesNumber()
        out = self._sense()
        self._resend_lost()
        self._deliver()
        return out

    def exit_free(self):
        return [self.tracker.free(f"{a}_out") for a in APPROACHES]

    def _obs(self):
        return self.encoder.encode(self.ledger.snapshot(self.t), self.sup.status(), self.exit_free())

    def _info(self):
        return {"mask": self.sup.mask(), "t": self.t}

    def step(self, action):
        if self.k is None:
            raise RuntimeError("reset() first")
        r_cfg = self.cfg["reward"]
        mask_before = self.sup.mask().tolist()
        t_decision = self.t
        self.sup.request(action)
        self.sup2.request(queue_rule(self.ledgers["J2"].snapshot(self.t), self.sup2.status(), self.sup2.mask(), "cycle"))
        switches0 = self.sup.switches
        reward = 0.0
        for _ in range(self.substeps):
            link_age = self.t - self.link_seen
            permits = {}
            for jn, sup in (("J", self.sup), ("J2", self.sup2)):
                snap = self.ledgers[jn].snapshot(self.t)
                permits[jn] = sup.tick(self.dt, snap["occupancy"], snap["occupancy_age"], link_age)
            waiting, front = self._substep(permits)
            starve = sum(max(0.0, w - r_cfg["starvation_s"]) for w in front.values()) / r_cfg["starvation_s"]
            reward -= self.dt * (r_cfg["wait_weight"] * waiting / max(1, self.n_cars) + r_cfg["starvation_weight"] * starve)
            if self.t - self.still_since >= self.cfg["simulation"]["deadlock_s"]:
                self.deadlock = True
                break
            if self.t >= self.horizon - 1e-9:
                break
        reward -= r_cfg["switch_weight"] * (self.sup.switches - switches0)
        if self.deadlock:
            reward -= 50.0
        if self.log_decisions:
            st = self.sup.status()
            self.obs_log.append({"t_decision": t_decision, "action": int(action)})
            self.decisions.append({"t": self.t, "action": int(action), "mask": mask_before, "reward": reward,
                                   "phase": st.phase, "stage": st.stage, "fault": st.fault})
        terminated = self.deadlock
        truncated = self.t >= self.horizon - 1e-9
        info = self._info()
        if terminated or truncated:
            info["metrics"] = self.metrics()
            self.close()
        return self._obs(), reward, terminated, truncated, info

    def metrics(self):
        all_delays, per = [], {}
        for a in APPROACHES:
            open_ = [self.t - t0 for car, (arm, t0) in self.requests.items() if arm == a]   # censored at end
            d = self.delays[a] + open_
            all_delays += d
            per[f"delay_{a}_s"] = float(np.mean(d)) if d else 0.0
            per[f"max_wait_{a}_s"] = float(max(d)) if d else 0.0
            per[f"requests_{a}"] = self.n_requests[a]
        n_req = sum(self.n_requests.values())
        return {"seed": self.seed, "cars": self.n_cars, "bias": self.bias,
                "requests": n_req, "crossings": self.crossings, "crossings_J2": self.crossings_j2,
                "completion_rate": self.crossings / n_req if n_req else 1.0,
                "mean_delay_s": float(np.mean(all_delays)) if all_delays else 0.0,
                "p95_delay_s": float(np.percentile(all_delays, 95)) if all_delays else 0.0,
                "max_wait_s": float(max(all_delays)) if all_delays else 0.0,
                "throughput_per_min": self.crossings * 60 / max(self.t, 1e-9),
                "switches": self.sup.switches, "forced_switches": self.sup.forced_switches,
                "fault_events": self.sup.fault_events + self.sup2.fault_events, "collisions": self.collisions,
                "conflict_steps": self.conflict_steps, "stopline_violations": self.stopline_violations,
                "deadlock": self.deadlock, "sim_seconds": self.t,
                "cmd_latency_s": self.cmd_latency, "obs_delay_s": self.obs_delay, **per}

    def close(self):
        if self.k is not None:
            try:
                self.k.close()
            except Exception:
                pass
            self.k = None
        if getattr(self, "route_file", None) and self.out_dir is None:
            Path(self.route_file).unlink(missing_ok=True)
