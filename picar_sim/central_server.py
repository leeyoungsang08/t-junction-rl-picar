"""Central PC server for the real PiCar course: both T junctions, protocol v2 (picar_lab/car_agent/protocol.py).

    python -m picar_sim.central_server --host 0.0.0.0 --port 5007 \\
        [--policy runs/picar_v9_photo_rt/seed42/best_policy.npz --mode shadow|active]

J  : CentralController with oldest_first (default) or the DQN (active) / DQN logged only (shadow)
J2 : CentralController with the fixed queue rule (as in SUMO)
Both: one shared SegmentTracker ("don't block the box"), PERMIT reservations, exit beams.

Reservation rules (CENTRAL_UDP_KO.md): a PERMIT reserves the car (RESERVED) and its next segment.
A reserved car counts as inside the box for the supervisor until it reports IN and the exit beam later
releases it, or until it reports a newer APPROACH/ARRIVE after the permit expired (proof it did not cross).
ACK loss or TTL expiry alone never frees a slot. The permit TTL must not exceed the all-red minimum.
NOT validated on hardware.
"""
import argparse
import json
import socket
import time
import uuid
from pathlib import Path

from picar_lab.car_agent.protocol import VERSION, decode, encode, valid_beam, valid_car_status
from picar_sim.map_builder import PKG, load_config, segment_capacity
from picar_sim.observation import SegmentTracker
from picar_sim.real_bridge import CentralController
from picar_sim.right_turn import arm_of, may_enter
from picar_sim.supervisor import APPROACHES

import yaml


class CentralServer:
    def __init__(self, host="127.0.0.1", port=5007, policy=None, mode="shadow", cfg=None, permit_ttl_s=0.5,
                 dt=0.1, log_path=None):
        self.cfg = cfg or load_config()
        if permit_ttl_s > self.cfg["signal"]["all_red_min_s"]:
            raise ValueError("permit TTL must not exceed all_red_min_s (a granted car must enter before switching)")
        lanes = yaml.safe_load((PKG / "maps/map_summary.yaml").read_text())["lanes"]
        self.tracker = SegmentTracker(segment_capacity(lanes, self.cfg["vehicle"], self.cfg["course"]))
        j_sig = None
        if policy:                                   # J runs in the action space the policy was trained for
            from picar_sim.dqn import Policy
            j_sig = {"control_mode": Policy(policy).meta["control_mode"]}
        self.ctrl = {
            "J": CentralController(policy, mode, self.cfg, permit_ttl_s, lanes, "J", "oldest_first", self.tracker,
                                   j_sig),
            "J2": CentralController(None, "shadow", self.cfg, permit_ttl_s, lanes, "J2", "queue_rule", self.tracker,
                                    {"control_mode": "cycle"}),
        }
        self.ttl, self.dt = permit_ttl_s, dt
        self.session = uuid.uuid4().hex
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.address = self.sock.getsockname()
        self.cars = {}           # car -> {addr, boot, last_seq, req, req_t, state, junction, lane, next}
        self.reserved = {}       # car -> {junction, next, grant, req, t}
        self.inside = {"J": [], "J2": []}   # FIFO of cars inside each box (for exit-beam assignment)
        self.grant_id = 0
        self.anomalies = []
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None
        self.stats = {"permits": 0, "right_turn_permits": 0, "permit_acks": 0, "exit_events": 0, "dup_status": 0,
                      "rejected": 0}

    # ------------------------------------------------------------------ inbound
    def _event(self, e):
        self.tracker.event(e)
        for jn, c in self.ctrl.items():
            if e.get("junction") in (jn, None):
                c.on_event(e)

    def _on_status(self, m, addr, now):
        car = m["car"]
        info = self.cars.get(car)
        if info is not None and m["boot"] != info["boot"]:
            if car in self.reserved or any(car in q for q in self.inside.values()):
                self.anomalies.append({"t": now, "car": car, "kind": "reboot_while_reserved"})
                for c in self.ctrl.values():                 # unknown position -> no new entries until resolved
                    c.ledger.occupancy_updated = None
            info = None
        if info is None:
            info = self.cars[car] = {"boot": m["boot"], "last_seq": 0, "addr": addr, "state": None}
        info["addr"] = addr
        self.sock.sendto(encode({"v": VERSION, "type": "status_ack", "session": self.session, "car": car,
                                 "boot": m["boot"], "ack": m["seq"]}), addr)
        if m["seq"] <= info["last_seq"]:
            self.stats["dup_status"] += 1
            return
        info["last_seq"] = m["seq"]
        st = m["state"]
        for c in self.ctrl.values():
            c.link_seen = now
        if st == "REGISTER":
            self._event({"t": now, "car": car, "junction": None, "state": "REGISTER", "lane": None, "edge": m["edge"]})
        elif st in ("APPROACH", "ARRIVE"):
            r = self.reserved.get(car)
            if r and now > r["t"] + self.ttl and m["seq"] > r["req"]:
                self.reserved.pop(car)                       # newer request after expiry: it did not cross
            info.update(req=m["seq"], req_t=now, junction=m["junction"], lane=m["lane"], next=m["next"])
            self._event({"t": now, "car": car, "junction": m["junction"], "state": st, "lane": m["lane"],
                         "next": m["next"]})
        elif st == "IN":
            r = self.reserved.pop(car, None)
            if r is None:
                self.anomalies.append({"t": now, "car": car, "kind": "entered_without_reservation"})
            self.inside[m["junction"]].append(car)
            info.update(req=None, move=(m["lane"], arm_of(m["next"])))   # movement fixed at entry
            self._event({"t": now, "car": car, "junction": m["junction"], "state": "IN", "lane": m["lane"],
                         "next": m["next"]})
        info["state"] = st

    def _on_beam(self, m, now):
        jn = m["junction"]
        if m["type"] == "beam_heartbeat":
            if m.get("ok"):
                self.ctrl[jn].on_occupancy_heartbeat(now)
            return
        self.stats["exit_events"] += 1
        q = self.inside[jn]
        nxt = self.ctrl[jn].next_edge
        match = next((c for c in q if nxt.get(c) == m["edge"]), None)
        if match is None:
            self.anomalies.append({"t": now, "kind": "unmatched_exit", "junction": jn, "edge": m["edge"]})
            self.ctrl[jn].ledger.occupancy_updated = None    # occupancy no longer trustworthy -> fault
            return
        q.remove(match)
        self._event({"t": now, "car": match, "junction": jn, "state": "EXIT", "lane": None, "edge": m["edge"]})

    def _recv_all(self, now):
        while True:
            try:
                raw, addr = self.sock.recvfrom(2048)
            except (BlockingIOError, InterruptedError):
                return
            m = decode(raw)
            if m is None:
                self.stats["rejected"] += 1
            elif m["type"] == "car_status" and valid_car_status(m):
                self._on_status(m, addr, now)
            elif m["type"] in ("exit_event", "beam_heartbeat") and valid_beam(m):
                self._on_beam(m, now)
            elif m["type"] == "permit_ack":
                self.stats["permit_acks"] += 1
            else:
                self.stats["rejected"] += 1

    # ------------------------------------------------------------------ control
    def _reserved_count(self, jn):
        return sum(1 for r in self.reserved.values() if r["junction"] == jn)

    def _free(self, edge):
        return self.tracker.free(edge) - sum(1 for r in self.reserved.values() if r["next"] == edge)

    def poll(self, now):
        self._recv_all(now)
        entries = {}
        for jn, c in self.ctrl.items():
            permits, entry = c.tick(now, self.dt, extra_occupancy=self._reserved_count(jn))
            entries[jn] = entry
            st = c.sup.status()
            if st.stage != "green" or st.fault is not None:
                continue                                     # every approach STOP
            led = c.ledger.cars
            green = APPROACHES[st.phase]
            active = [self.cars[x].get("move", (None, None)) for x in self.inside[jn]]
            active += [r["move"] for r in self.reserved.values() if r["junction"] == jn]
            lanes = {}
            for car, info in self.cars.items():
                if info.get("junction") == jn and info.get("state") in ("APPROACH", "ARRIVE") and info.get("req") is not None:
                    lanes.setdefault(info.get("lane"), []).append(car)
            for q in lanes.values():   # cars cannot overtake on one lane: arrival order, never skip a blocked front car
                q.sort(key=lambda car: led.get(car, {}).get("request_t", float("inf")))
            green_queue = [(green, arm_of(self.cars[x].get("next"))) for x in lanes.get(green, [])]
            enabled = bool(self.cfg["signal"].get("unprotected_right"))
            for lane, queue in lanes.items():
                for car in queue:
                    info = self.cars[car]
                    move = (lane, arm_of(info.get("next")))
                    r = self.reserved.get(car)
                    if r is None:
                        state = led.get(car, {}).get("state", info.get("state"))
                        if not may_enter(jn, move, st, enabled, active, green_queue, state) or self._free(info["next"]) <= 0:
                            break
                        self.grant_id += 1
                        r = self.reserved[car] = {"junction": jn, "next": info["next"], "grant": self.grant_id,
                                                  "req": info["req"], "t": now, "move": move}
                        active.append(move)
                        self.stats["permits"] += 1
                        self.stats["right_turn_permits"] += lane != green
                    elif r["req"] != info["req"]:
                        self.grant_id += 1                       # renewal for a newer request = new grant id
                        r.update(req=info["req"], t=now, grant=self.grant_id)
                    else:
                        if lane != green:
                            break
                        continue                                 # already answered this request
                    self.sock.sendto(encode({"v": VERSION, "type": "permit", "session": self.session, "car": car,
                                             "boot": info["boot"], "req": info["req"], "grant": r["grant"],
                                             "junction": jn, "ttl_ms": int(self.ttl * 1000)}), info["addr"])
                    if lane != green:
                        break                                    # unprotected right turn: head car only
        if self.log:
            self.log.write(json.dumps({"t": round(now, 3), "J": entries["J"], "J2": entries["J2"],
                                       "reserved": self.reserved}, default=float) + "\n")
        return entries

    def close(self):
        self.sock.close()
        if self.log:
            self.log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5007)
    ap.add_argument("--policy")
    ap.add_argument("--mode", choices=["shadow", "active"], default="shadow")
    ap.add_argument("--log", default="lab_run/central_v2.jsonl")
    a = ap.parse_args()
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    srv = CentralServer(a.host, a.port, a.policy, a.mode, log_path=a.log)
    print(f"READY {srv.address} session={srv.session} mode={a.mode}", flush=True)
    t0 = time.monotonic()
    nxt = t0
    try:
        while True:
            srv.poll(time.monotonic() - t0)
            nxt += srv.dt
            time.sleep(max(0.0, nxt - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()
