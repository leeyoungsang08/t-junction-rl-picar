"""Software-in-the-loop test: SUMO cars run the real Pi-side CarAgent and talk to the real CentralServer
over real UDP sockets (localhost), with virtual exit beams. Nothing in between is mocked.

    SUMO truth --virtual markers--> CarAgent (one UDP socket per car) <==UDP==> CentralServer
    SUMO truth --virtual exit beam--> beam sockets ==UDP==> CentralServer
    CarAgent.may_cross() --> front-car stop-line gate in SUMO (link state per front car)

python -m picar_sim.sil --cars 5 --seconds 180 [--policy ... --mode active] [--beam-fault 60 80]
"""
import argparse
import json
import socket

import numpy as np

from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.exit_beam import ExitBeamMonitor
from picar_lab.car_agent.protocol import decode, encode
from picar_sim.central_server import CentralServer
from picar_sim.env import PicarEnv
from picar_sim.map_builder import APPROACHES, load_config


class SILRun:
    def __init__(self, seed=1700, cars=5, bias="uniform", policy=None, mode="shadow", beam_fault=None,
                 link_fault=None):
        self.cfg = load_config()
        self.env = PicarEnv(self.cfg)
        self.env.reset(seed=seed, cars=cars, bias=bias, horizon=10_000)
        self.server = CentralServer("127.0.0.1", 0, policy, mode, self.cfg)
        self.addr = self.server.address
        self.beam_fault, self.link_fault = beam_fault, link_fault
        k = self.env.k
        self.agents, self.socks = {}, {}
        for car in k.vehicle.getIDList():
            road = k.vehicle.getRoadID(car)
            self.agents[car] = CarAgent(car, [road, k.vehicle.getRoute(car)[-1]])
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            s.setblocking(False)
            self.socks[car] = s
        self.beams = {"J": ExitBeamMonitor("beam-J", "J", [f"{a}_out" for a in APPROACHES]),
                      "J2": ExitBeamMonitor("beam-J2", "J2", [f"{a}_in" for a in APPROACHES])}
        self.beam_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.exits_pending = {"J": [], "J2": []}
        self.truth = {}
        self.applied = {jn: "r" * len(self.env.links[jn]) for jn in ("J", "J2")}
        self.m = {"crossings": 0, "illegal_crossings": 0, "collisions": 0, "conflict_steps": 0,
                  "crossings_during_beam_fault_J": 0, "deadlock": False}
        self.t = 0.0
        for car, ag in self.agents.items():
            ag.register(0.0)

    def _send(self, car, msgs):
        for m in msgs:
            if self.link_fault and self.link_fault[0] <= self.t < self.link_fault[1]:
                continue                                   # car radio silent
            self.socks[car].sendto(encode(m), self.addr)

    def _recv(self):
        for car, s in self.socks.items():
            while True:
                try:
                    raw, _ = s.recvfrom(2048)
                except BlockingIOError:
                    break
                msg = decode(raw)
                if msg:
                    self.agents[car].on_packet(msg, self.t)

    def _gate(self):
        """Per junction: only the front car of each approach may pass, and only with a valid permit."""
        k, env = self.env.k, self.env
        for jn in ("J", "J2"):
            front = {}
            for car, (tj, st, arm) in self.truth.items():
                if tj == jn and st in ("APPROACH", "ARRIVE"):
                    pos = k.vehicle.getLanePosition(car)
                    if arm not in front or pos > front[arm][1]:
                        front[arm] = (car, pos)
            chars = []
            for i, (arm, target) in enumerate(env.links[jn]):
                ok = False
                if arm in front:
                    car, pos = front[arm]
                    nxt = k.vehicle.getRoute(car)[-1]
                    L = env.lane_len[k.vehicle.getRoadID(car)] if not k.vehicle.getRoadID(car).startswith(":") else 0
                    ag = self.agents[car]
                    if nxt == target and pos >= L - 0.05 and k.vehicle.getSpeed(car) > 0.01:
                        ag.commit(self.t)                  # braking point passed while the permit was valid
                    ok = nxt == target and (ag.may_cross(self.t) or ag.committed)
                chars.append("G" if ok else "r")
            self.applied[jn] = "".join(chars)
            k.trafficlight.setRedYellowGreenState(jn, self.applied[jn])

    def _sense(self):
        k, env = self.env.k, self.env
        moving = False
        inside = {"J": set(), "J2": set()}
        for car in k.vehicle.getIDList():
            road, pos, spd = k.vehicle.getRoadID(car), k.vehicle.getLanePosition(car), k.vehicle.getSpeed(car)
            moving |= spd > 0.01
            route = k.vehicle.getRoute(car)
            ag = self.agents[car]
            if road == route[-1] and not road.startswith(":"):
                k.vehicle.setRoute(car, [road, env._next_edge(road, self.t, env.route_rng[car])])
                route = k.vehicle.getRoute(car)
            env.truth = self.truth
            jn, st, arm = env._truth_state(car, road, pos, spd)
            pj, ps, parm = self.truth.get(car, (None, "CRUISE", None))
            if ps == "IN" and (jn, st) != (pj, "IN"):
                self.exits_pending[pj].append(road)        # rear passed this exit beam
                ag.on_left_box(self.t)
            if not road.startswith(":") and ag.zone != "IN":
                ag.route, ag.idx = [road, route[-1]], 0     # keep the agent's plan = SUMO's plan
            if (jn, st) != (pj, ps):
                if st == "APPROACH":
                    ag.on_approach_marker(self.t)
                elif st == "ARRIVE":
                    ag.on_approach_marker(self.t)
                    ag.on_stopped_at_stopline(self.t)
                elif st == "IN" and ps in ("APPROACH", "ARRIVE"):
                    env.in_arm[car] = parm
                    arm = parm
                    legal = ag.on_crossed_stopline(self.t)
                    self.m["crossings"] += 1
                    self.m["illegal_crossings"] += 0 if legal else 1
                    # grace = stale limit before the supervisor faults + permit TTL already granted + one step
                    grace = self.cfg["signal"]["occupancy_stale_s"] + self.server.ttl + 0.2
                    if jn == "J" and self.beam_fault and self.beam_fault[0] + grace <= self.t < self.beam_fault[1]:
                        self.m["crossings_during_beam_fault_J"] += 1
            if st == "IN":
                inside[jn].add(env.in_arm.get(car))
            self.truth[car] = (jn, st, arm if st != "IN" else env.in_arm.get(car))
        for jn, arms in inside.items():
            arms.discard(None)
            if len(arms) > 1:
                self.m["conflict_steps"] += 1
        return moving

    def run(self, seconds, dt=0.1):
        k = self.env.k
        still = 0.0
        while self.t < seconds - 1e-9:
            self._gate()
            k.simulationStep()
            self.t = round(self.t + dt, 6)
            self.m["collisions"] += k.simulation.getCollidingVehiclesNumber()
            still = 0.0 if self._sense() else still + dt
            if still >= 60:
                self.m["deadlock"] = True
                break
            for jn, mon in self.beams.items():
                readings = {e: False for e in mon.edges}
                for e in self.exits_pending[jn]:         # emulate blocked -> clear on the beam of that edge
                    mon.blocked[e] = True
                faulty = self.beam_fault and jn == "J" and self.beam_fault[0] <= self.t < self.beam_fault[1]
                for msg in mon.update(readings):
                    if not (faulty and msg["type"] == "beam_heartbeat"):
                        self.beam_sock.sendto(encode(msg), self.addr)
                self.exits_pending[jn] = []
            for car, ag in self.agents.items():
                self._send(car, ag.tick(self.t))
            self.server.poll(self.t)
            self._recv()
        self.m.update(seconds=self.t, server=self.server.stats, anomalies=len(self.server.anomalies),
                      anomaly_kinds=sorted({a["kind"] for a in self.server.anomalies}))
        return self.m

    def close(self):
        self.env.close()
        self.server.close()
        for s in self.socks.values():
            s.close()
        self.beam_sock.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1700)
    ap.add_argument("--cars", type=int, default=5)
    ap.add_argument("--bias", default="uniform")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--policy")
    ap.add_argument("--mode", default="shadow")
    ap.add_argument("--beam-fault", type=float, nargs=2)
    ap.add_argument("--link-fault", type=float, nargs=2)
    a = ap.parse_args()
    run = SILRun(a.seed, a.cars, a.bias, a.policy, a.mode, a.beam_fault, a.link_fault)
    try:
        print(json.dumps(run.run(a.seconds), indent=1))
    finally:
        run.close()
