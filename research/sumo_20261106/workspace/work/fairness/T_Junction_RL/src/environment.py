from collections import deque
from dataclasses import asdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import uuid
import socket
import subprocess
import xml.etree.ElementTree as ET
import gymnasium as gym
import numpy as np
import pandas as pd
import traci
import traci.constants as tc
from src.utils import DIRECTIONS, ROOT, atomic_text, binary, load_yaml, sha256, write_json
from src.state_adapter import StateAdapter
from src.state_extractor import extract
from src.signal_controller import SignalController, DirectPhaseSignalController
from src.reward import reward_components
from src.metrics import episode_metrics


class TJunctionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, route_file=None, seed=1000, output_dir=None, capture=False, gui=False, delay_ms=0, *, network_file=None, service_observer=None):
        self.config = config or load_yaml("base.yaml")
        sim = self.config["simulation"]
        if sim["step_seconds"] != 1 or int(sim["decision_seconds"]) != sim["decision_seconds"] or sim["decision_seconds"] < 1:
            raise ValueError("Use 1-second simulation steps and positive integer decision_seconds")
        self.backend = sim["backend"]
        if self.backend not in ("traci", "libsumo"):
            raise ValueError("Unknown SUMO backend")
        if self.backend == "libsumo" and gui:
            raise ValueError("Use the TraCI backend for SUMO GUI")
        self.route_file = Path(route_file).resolve() if route_file else None
        self.network_file = Path(network_file).resolve() if network_file else ROOT / "maps/t_junction.net.xml"
        self.service_observer = service_observer
        self.sumo_seed = seed
        self.output_dir = Path(output_dir or ROOT / "outputs/tmp" / uuid.uuid4().hex)
        self.capture = capture
        self.gui = gui
        self.delay_ms = delay_ms
        self.adapter = StateAdapter(self.config["normalization"], self.config["signal"], self.config.get("observation_schema", 1))
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.adapter.features),), dtype=np.float32)
        self.direct_phase = self.config["signal"].get("control_mode", "cycle") == "direct"
        self.action_space = gym.spaces.Discrete(3 if self.direct_phase else 2)
        self.conn = None
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        options = options or {}
        route = Path(options.get("route_file", self.route_file)).resolve()
        self.sumo_seed = int(seed if seed is not None else self.sumo_seed)
        self.manifest = json.loads(route.with_suffix(".json").read_text(encoding="utf-8"))
        if "route_sha256" in self.manifest and sha256(route) != self.manifest["route_sha256"]:
            raise ValueError("Route file differs from the demand manifest")
        self.horizon = int(self.config["simulation"]["horizon_seconds"])
        if self.horizon != self.manifest["horizon_seconds"]:
            raise ValueError("Route demand duration and simulation horizon differ")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tripinfo = self.output_dir / "tripinfo.xml"
        command = [binary("sumo-gui" if self.gui else "sumo"), "-n", str(self.network_file),
                   "-r", str(route), "--seed", str(self.sumo_seed), "--step-length", "1",
                   "--no-step-log", "true", "--duration-log.disable", "true", "--no-warnings", "true",
                   "--tripinfo-output", str(self.tripinfo), "--tripinfo-output.write-unfinished", "true",
                   "--error-log", str(self.output_dir / "sumo_error.log")]
        if self.config["simulation"].get("audit_vehicle_safety", False):
            command += ["--time-to-teleport", "-1", "--collision.check-junctions", "true", "--collision.action", "warn"]
        if self.gui:
            command += ["--start", "--delay", str(self.delay_ms)]
        if self.backend == "libsumo":
            import libsumo

            with (self.output_dir / "traci_connect.log").open("w") as connection_log:
                connection_log.write("libsumo: in-process TraCI API\n")
            libsumo.start(command)
            self.conn = libsumo
        else:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            self.process = subprocess.Popen(command + ["--remote-port", str(port)], stdout=subprocess.DEVNULL)
            try:
                with (self.output_dir / "traci_connect.log").open("w") as connection_log, redirect_stdout(connection_log):
                    self.conn = traci.connect(port, host="127.0.0.1", proc=self.process, waitBetweenRetries=0.02, numRetries=500)
                self.conn.getVersion()
            except BaseException:
                self.process.terminate()
                self.process.wait(timeout=10)
                raise
        controller_type = DirectPhaseSignalController if self.direct_phase else SignalController
        self.controller = controller_type(self.conn.trafficlight.getControlledLinks("J"), self.config["signal"])
        self.applied_signal = self.controller.before_second()
        self.conn.trafficlight.setRedYellowGreenState("J", self.applied_signal)
        self.timeseries, self.decisions, self.frames = [], [], []
        self.departed, self.arrived = set(), set()
        self.collisions = self.teleports = 0
        self.time = 0
        self.finished = False
        self.vehicle_data = {}
        self.requests = deque(sorted(
            (v["requested_depart"], vid, DIRECTIONS.index(v["origin"]))
            for vid, v in self.manifest["vehicles"].items()
        ))
        self.pending = [{} for _ in DIRECTIONS]
        self.recent = [deque() for _ in DIRECTIONS]
        self.accumulated_wait = {}
        self.conn.simulation.subscribe((tc.VAR_DEPARTED_VEHICLES_IDS, tc.VAR_ARRIVED_VEHICLES_IDS,
                                        tc.VAR_TELEPORT_STARTING_VEHICLES_IDS, tc.VAR_COLLIDING_VEHICLES_IDS))
        if self.service_observer is not None:
            self.service_observer.reset()
        self._observe()
        return self.adapter.encode(self.state), self._info()

    def _observe(self, just_departed=()):
        self.state, self.traffic = extract(self.conn, self.controller, self.vehicle_data)
        while self.requests and self.requests[0][0] < self.time:
            requested, vid, direction = self.requests.popleft()
            self.recent[direction].append(requested)
            if vid not in self.departed:
                self.pending[direction][vid] = requested
        for vid in just_departed:
            direction = DIRECTIONS.index(self.manifest["vehicles"][vid]["origin"])
            self.pending[direction].pop(vid, None)
        waits = [0.0] * 3
        accumulated = {}
        for vid, data in self.vehicle_data.items():
            road = data["road"]
            if road.endswith("_in") and road[0] in DIRECTIONS:
                direction = DIRECTIONS.index(road[0])
                accumulated[vid] = self.accumulated_wait.get(vid, 0) + int(data["speed"] < 0.1)
                waits[direction] = max(waits[direction], accumulated[vid])
        self.accumulated_wait = accumulated
        for direction, recent in enumerate(self.recent):
            while recent and recent[0] < self.time - 30:
                recent.popleft()
            oldest = next(iter(self.pending[direction].values()), self.time)
            waits[direction] = max(waits[direction], self.time - oldest)
        c = self.controller
        clearance = 0
        if c.stage == "yellow":
            clearance = c.config["yellow"] - c.clearance_elapsed + c.config["all_red"]
        elif c.stage == "all_red":
            clearance = c.config["all_red"] - c.clearance_elapsed
        self.state.queues = self.traffic["queues"]
        self.state.pending = [len(group) for group in self.pending]
        self.state.longest_wait = waits
        self.state.recent_requests = [len(group) for group in self.recent]
        self.state.clearance_remaining = clearance

    def _info(self):
        return {"action_mask": self.controller.mask(), "raw_state": asdict(self.state),
                "queues": self.traffic["queues"], "max_direction_wait": self.traffic["max_direction_wait"], "time": self.time}

    def _read_vehicles(self, just_departed):
        for vid in just_departed:
            self.departed.add(vid)
            variables = [tc.VAR_SPEED, tc.VAR_ROAD_ID, tc.VAR_WAITING_TIME]
            if self.capture:
                variables.append(tc.VAR_POSITION)
            self.conn.vehicle.subscribe(vid, variables)
        data = {}
        for vid, values in self.conn.vehicle.getAllSubscriptionResults().items():
            if tc.VAR_SPEED not in values:
                continue
            data[vid] = {"speed": values[tc.VAR_SPEED], "road": values[tc.VAR_ROAD_ID], "wait": values[tc.VAR_WAITING_TIME]}
            if self.capture:
                data[vid]["position"] = list(values[tc.VAR_POSITION])
        return data

    def step(self, action):
        if self.conn is None or self.finished:
            raise RuntimeError("Call reset() before step()")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"Expected discrete action in 0..{self.action_space.n-1}, got {action!r}")
        action = int(action)
        started_at = self.time
        forced_before = self.controller.forced_switches
        raw_before = asdict(self.state)
        mask_before = self.controller.mask().tolist()
        switches_before = self.controller.switches
        self.controller.request(action)
        mask_required_switch = self.controller.forced_switches > forced_before
        requested_target = self.controller.target_phase if self.controller.switches > switches_before else None
        switch_target = requested_target
        reward, component_sum = 0.0, {}
        while self.time < self.horizon:
            signal = self.controller.before_second()
            phase, stage = self.controller.phase, self.controller.stage
            if signal != self.applied_signal:
                self.conn.trafficlight.setRedYellowGreenState("J", signal)
                self.applied_signal = signal
            self.conn.simulationStep()
            self.time += 1
            events = self.conn.simulation.getSubscriptionResults()
            just_departed = events[tc.VAR_DEPARTED_VEHICLES_IDS]
            self.vehicle_data = self._read_vehicles(just_departed)
            self.arrived.update(events[tc.VAR_ARRIVED_VEHICLES_IDS])
            if events[tc.VAR_COLLIDING_VEHICLES_IDS]:
                self.collisions += len(self.conn.simulation.getCollisions())
            self.teleports += len(events[tc.VAR_TELEPORT_STARTING_VEHICLES_IDS])
            self.controller.after_second()
            self._observe(just_departed)
            if self.service_observer is not None:
                self.service_observer.observe_second(phase, stage, self.traffic["queues"], self.state.pending)
            stopped = sum(v["speed"] < 0.1 for v in self.vehicle_data.values())
            new_switches = self.controller.switches - switches_before
            if new_switches:
                switch_target = self.controller.target_phase
            switches_before = self.controller.switches
            r, terms = reward_components(self.traffic["queues"], stopped, self.traffic["internal_stopped"],
                                        self.state.longest_wait if self.adapter.schema == 2 else self.traffic["max_direction_wait"],
                                        new_switches, self.config["reward"],
                                        pending=sum(self.state.pending) if self.adapter.schema == 2 else 0,
                                        internal_count=self.state.internal_vehicles if self.adapter.schema == 2 else 0)
            reward += r
            for k, v in terms.items():
                component_sum[k] = component_sum.get(k, 0) + v
            self.timeseries.append({"time": self.time, "phase": phase, "stage": stage, "tls_state": signal,
                                    "queue_total": sum(self.traffic["queues"]), "pending_total": sum(self.state.pending),
                                    "longest_wait_s": max(self.state.longest_wait),
                                    "switch_events": new_switches,
                                    **{f"queue_{d}": q for d, q in zip(DIRECTIONS, self.traffic["queues"])},
                                    **({f"pending_{d}": n for d, n in zip(DIRECTIONS, self.state.pending)}
                                       if self.service_observer is not None else {}),
                                    "max_consecutive_wait_s": max((v["wait"] for v in self.vehicle_data.values()), default=0),
                                    "mean_current_wait_s": float(np.mean([v["wait"] for v in self.vehicle_data.values()])) if self.vehicle_data else 0,
                                    "active_vehicles": len(self.vehicle_data), "cumulative_departed": len(self.departed),
                                    "cumulative_arrived": len(self.arrived), "reward": r, **terms})
            if self.capture:
                self.frames.append({"time": self.time, "phase": phase, "stage": stage, "signal": signal,
                                    "queues": self.traffic["queues"], "arrived": len(self.arrived),
                                    "mean_current_wait_s": self.timeseries[-1]["mean_current_wait_s"],
                                    "vehicles": [{"id": vid, "x": v["position"][0], "y": v["position"][1],
                                                  "speed": v["speed"], "origin": self.manifest["vehicles"][vid]["origin"]}
                                                 for vid, v in self.vehicle_data.items()]})
            if self.time % self.config["simulation"]["decision_seconds"] == 0:
                break
        self.decisions.append({"time": self.time, "decision_time": started_at, "state": json.dumps(raw_before), "action": action,
                               "requested_action": action, "executed_switch_target": switch_target,
                               "requested_switch_target": requested_target, "result_phase": self.controller.phase,
                               "result_stage": self.controller.stage,
                               "max_green_mask_required": mask_required_switch,
                               "automatic_switches": self.controller.forced_switches - forced_before - int(mask_required_switch),
                               "forced_switches": self.controller.forced_switches - forced_before,
                               "action_mask": json.dumps(mask_before), "reward": reward, **component_sum})
        truncated = self.time >= self.horizon
        info = self._info()
        info["executed_action"] = action  # Validated action applied at the decision boundary.
        info["executed_switch_target"] = switch_target
        info["duration_seconds"] = self.time - started_at
        info["forced_switches"] = self.controller.forced_switches - forced_before
        info["reward_components"] = component_sum
        if truncated:
            self.finished = True
            self.close()
            self.result = episode_metrics(self.tripinfo, self.manifest, self.timeseries, self.departed, self.arrived,
                                          self.controller.switches, self.controller.forced_switches, self.collisions, self.teleports)
            info["metrics"] = self.result
            self.save_logs()
        return self.adapter.encode(self.state), reward, False, truncated, info

    def save_logs(self):
        trip_xml = self.tripinfo.read_text(encoding="utf-8")
        ET.fromstring(trip_xml)
        atomic_text(self.output_dir / "timeseries.csv", pd.DataFrame(self.timeseries).to_csv(index=False))
        atomic_text(self.output_dir / "decisions.csv", pd.DataFrame(self.decisions).to_csv(index=False))
        write_json(self.output_dir / "metrics.json", self.result)
        if self.capture:
            atomic_text(self.output_dir / "trajectory.jsonl", "".join(json.dumps(f) + "\n" for f in self.frames))

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
