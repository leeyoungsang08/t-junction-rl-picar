"""Central-PC bridge: PiCar UDP status -> CarLedger -> Encoder -> Policy -> IntersectionSupervisor -> PERMIT.

Uses only numpy + the standard library, and the SAME ledger/encoder/supervisor classes as the SUMO env.
Packets follow picar_lab/docs/CENTRAL_UDP_KO.md (lab_status with APPROACH/ARRIVE/IN/EXIT).

  # shadow: policy only logs what it would do; oldest_first (+ max-wait guard) actually drives the supervisor
  python -m picar_sim.real_bridge serve --policy runs/.../best_policy.npz --mode shadow --host 0.0.0.0
  # offline replay of a recorded status log (JSONL with t, car, lane, state[, occupancy_ok, link_ok])
  python -m picar_sim.real_bridge replay --policy runs/.../best_policy.npz --log lab_run/events.jsonl
NOT validated on hardware. Motors must stay gated by the Pi-local safety gate (TTL, line loss, obstacle).
"""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path

import numpy as np

from picar_sim.baselines import oldest_first
from picar_sim.dqn import Policy
from picar_sim.map_builder import load_config
from picar_sim.map_builder import segment_capacity
from picar_sim.observation import CarLedger, Encoder, SegmentTracker
from picar_sim.supervisor import APPROACHES, IntersectionSupervisor

LANE_MAP = {"W": "W", "E": "E", "N": "N", "S": "N"}   # legacy lab lanes W/E/S -> course arms W/E/N


class CentralController:
    """Controls J. Event dicts: {t, car, junction: "J"|"J2"|None, state: REGISTER|APPROACH|ARRIVE|IN|EXIT,
    lane (arm, for APPROACH/ARRIVE/IN), next (edge the car will take, from its registered route),
    edge (REGISTER start edge / EXIT edge = which exit beam fired)}. J2 events keep segment counts right."""

    def __init__(self, policy_path, mode="shadow", cfg=None, permit_ttl_s=0.5, lane_lengths=None):
        self.cfg = cfg or load_config()
        self.policy = Policy(policy_path) if policy_path else None
        sig = dict(self.cfg["signal"])
        if self.policy:
            if self.policy.meta.get("schema") not in ("picar_v1", "picar_v2"):
                raise ValueError("policy schema mismatch")
            for k in ("control_mode", "min_green_s", "max_green_s", "yellow_s", "all_red_min_s", "max_wait_guard_s"):
                if self.policy.meta["signal"].get(k, 0) != sig.get(k, 0):
                    raise ValueError(f"supervisor setting {k} differs from the trained policy")
        self.mode = mode
        self.sup = IntersectionSupervisor(sig)
        self.enc = Encoder(sig["max_green_s"], sig["clear_timeout_s"],
                           self.policy.meta.get("schema", "picar_v1") if self.policy else "picar_v1")
        self.ledger = CarLedger()
        if lane_lengths is None:
            import yaml
            from picar_sim.map_builder import PKG
            lane_lengths = yaml.safe_load((PKG / "maps/map_summary.yaml").read_text())["lanes"]
        self.tracker = SegmentTracker(segment_capacity(lane_lengths, self.cfg["vehicle"], self.cfg["course"]))
        self.next_edge = {}
        self.ttl = permit_ttl_s
        self.link_seen = None
        self.last_decision = None
        self.log = []

    def on_event(self, e):
        self.tracker.event(e)
        if e.get("next"):
            self.next_edge[e["car"]] = e["next"]
        if e.get("junction") == "J":
            lane = e.get("lane")
            self.ledger.event(e["car"], e["state"], LANE_MAP[lane] if lane else None, e["t"])
        self.link_seen = e["t"]

    def exit_free(self):
        return [self.tracker.free(f"{a}_out") for a in APPROACHES]

    def car_may_go(self, car, permits):
        """PERMIT for one car: its approach is GO and its next segment has room (don't block the box)."""
        c = self.ledger.cars.get(car, {})
        if c.get("state") not in ("APPROACH", "ARRIVE") or permits.get(c.get("approach")) != "GO":
            return False
        nxt = self.next_edge.get(car)
        return nxt is not None and self.tracker.free(nxt) > 0

    def on_occupancy_heartbeat(self, t):
        self.ledger.heartbeat_occupancy(t)

    def tick(self, t, dt):
        snap = self.ledger.snapshot(t)
        decision = None
        if self.last_decision is None or t - self.last_decision >= self.cfg["simulation"]["decision_s"] - 1e-9:
            self.last_decision = t
            self.sup.observe_waits(snap["oldest_request"])
            status, mask = self.sup.status(), self.sup.mask()
            obs = self.enc.encode(snap, status, self.exit_free())
            rec = int(self.policy.act(obs, mask)) if self.policy else None
            rule = int(oldest_first(snap, status, mask, self.sup.mode))   # best simple controller (4th experiment)
            action = rec if (self.mode == "active" and rec is not None) else rule
            self.sup.request(action)
            decision = {"t": t, "policy_action": rec, "rule_action": rule, "applied": action,
                        "mask": mask.tolist(), "snapshot": snap}
        link_age = float("inf") if self.link_seen is None else t - self.link_seen
        permits = self.sup.tick(dt, snap["occupancy"], snap["occupancy_age"], link_age)
        st = self.sup.status()
        entry = {"t": round(t, 3), "phase": APPROACHES[st.phase], "stage": st.stage, "fault": st.fault,
                 "permits": permits, "decision": decision}
        self.log.append(entry)
        return permits, entry


def replay(policy, log_path, out_path=None, dt=0.1, start_t=None, end_t=None, mode="shadow", events=None):
    """Replay status events. Sensors/link are assumed alive (periodic heartbeats) except inside
    {"fault": "occupancy"|"link", "t": t0, "until": t1} records of the log."""
    if events is None:
        events = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    faults = [(e["t"], e["until"], e["fault"]) for e in events if "fault" in e]
    events = sorted((e for e in events if "state" in e), key=lambda e: e["t"])  # stable: keeps same-time order
    cc = CentralController(policy, mode=mode)
    t = events[0]["t"] if start_t is None else start_t
    end = (events[-1]["t"] + 5) if end_t is None else end_t
    i = agree = total = 0
    while t <= end + 1e-9:
        while i < len(events) and events[i]["t"] <= t + 1e-9:
            e = events[i]
            cc.on_event(e)
            i += 1
        if not any(a <= t < b and k == "occupancy" for a, b, k in faults):
            cc.on_occupancy_heartbeat(t)
        if not any(a <= t < b and k == "link" for a, b, k in faults):
            cc.link_seen = t
        _, entry = cc.tick(t, dt)
        d = entry["decision"]
        if d and d["policy_action"] is not None:
            total += 1
            agree += d["policy_action"] == d["rule_action"]
        t = round(t + dt, 6)
    if out_path:
        Path(out_path).write_text("\n".join(json.dumps(x, default=float) for x in cc.log), encoding="utf-8")
    return {"decisions": total, "policy_rule_agreement": agree / total if total else None, "log": cc.log}


def serve(policy, host, port, mode, dt=0.1):
    cc = CentralController(policy, mode=mode)
    session = uuid.uuid4().hex
    peers = {}                     # car -> (addr, lane)
    grant = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind((host, port))
        s.settimeout(dt / 2)
        print(f"READY {host}:{port} session={session} mode={mode}", flush=True)
        t0 = time.monotonic()
        nxt = t0
        while True:
            try:
                raw, addr = s.recvfrom(2048)
                msg = json.loads(raw)
                now = time.monotonic() - t0
                if msg.get("type") == "lab_status":   # needs junction/next/edge fields (see CentralController doc)
                    peers[msg["car"]] = (addr, msg["lane"])
                    cc.on_event({"t": now, "car": msg["car"], "junction": msg.get("junction", "J"),
                                 "state": msg["state"], "lane": msg.get("lane"), "next": msg.get("next"),
                                 "edge": msg.get("edge")})
                elif msg.get("type") == "occupancy_heartbeat":     # exit-beam sensor node
                    cc.on_occupancy_heartbeat(now)
            except (socket.timeout, ValueError, KeyError):
                pass
            if time.monotonic() >= nxt:
                now = time.monotonic() - t0
                permits, _ = cc.tick(now, dt)
                for car, (addr, lane) in peers.items():
                    if cc.car_may_go(car, permits):
                        grant += 1
                        s.sendto(json.dumps({"v": 1, "type": "permit", "session": session, "car": car, "grant": grant,
                                             "cmd": "GO", "ttl_ms": int(cc.ttl * 1000)}).encode(), addr)
                nxt += dt


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--policy")
    r.add_argument("--log", required=True)
    r.add_argument("--out", default="replay_decisions.jsonl")
    v = sub.add_parser("serve")
    v.add_argument("--policy")
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=5007)
    v.add_argument("--mode", choices=["shadow", "active"], default="shadow")
    a = p.parse_args()
    if a.cmd == "replay":
        res = replay(a.policy, a.log, a.out)
        print({k: v for k, v in res.items() if k != "log"})
    else:
        serve(a.policy, a.host, a.port, a.mode)
