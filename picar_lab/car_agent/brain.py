"""CarBrain: line following + markers + intersection agent + junction maneuvers, without ROS.

ros2_drive_node.py wraps it for Gazebo and the Pi; picar_gazebo/kinematic_check.py drives it with a 2D
kinematic model that reads the generated course texture as the IR sensor, so the same decisions are tested
before Gazebo.
"""
from picar_lab.car_agent.agent import CarAgent, MarkerDetector
from picar_lab.car_agent.drive import JunctionManeuver, LineFollower, Odometer, movement


class CarBrain:
    def __init__(self, agent: CarAgent, plans, speed_mps=0.25, dark=1, turn_rate_dps=90.0,
                 approach_max_m=0.03, stop_min_m=0.04, dt=0.05, obstacle_stop_m=0.20, obstacle_clear_m=0.25,
                 obstacle_slow_m=0.40,
                 kp=1.0, kd=0.0, trim_per_m=0.0, turn_scale=1.0, latency_s=0.05, accel_mps2=1.0,
                 min_approach_to_stop_m=0.3):
        self.agent, self.plans = agent, plans
        self.trim_per_m, self.turn_scale = trim_per_m, turn_scale
        self.odo = Odometer(latency_s, accel_mps2)          # distance since the stop line, incl. roll-out
        self.s_stopline = None
        self.s_handoff, self.recover_left = None, 0
        self.s_approach, self.min_approach_to_stop = None, min_approach_to_stop_m      # line re-acquisition right after a junction
        self.speed, self.dark, self.turn_rate = speed_mps, dark, turn_rate_dps
        self.follower = LineFollower(speed_mps, kp=kp, kd=kd, dark=dark)
        self.markers = MarkerDetector(dt=dt, approach_max_m=approach_max_m, stop_min_m=stop_min_m)
        self.mode = "follow"            # follow | hold | maneuver | recover | fault
        self.maneuver = None
        self.cmd = (0.0, 0.0)
        self.events = []                # (t, text) for logs/tests
        self.obstacle_stop, self.obstacle_clear, self.obstacle_slow = obstacle_stop_m, obstacle_clear_m, obstacle_slow_m
        self.blocked = False

    def _start_maneuver(self, now):
        jn, arm_in = self.agent.junction, self.agent.lane
        arm_out = self.agent.next_edge.split("_")[0]
        plan = self.plans[f"{jn}:{arm_in}:{arm_out}"]
        if plan["turn"] != movement(jn, arm_in, arm_out):
            raise RuntimeError("maneuver table does not match the course")
        legal = self.agent.on_crossed_stopline(now)
        self.events.append((now, f"enter {jn} {arm_in}->{arm_out} legal={legal}"))
        self.maneuver = JunctionManeuver(plan, speed_mps=min(self.speed, 0.2), turn_rate_dps=self.turn_rate,
                                         dark=self.dark, trim_per_m=self.trim_per_m, turn_scale=self.turn_scale,
                                         odometer=self.odo, s_ref=self.s_stopline)
        self.mode = "maneuver"

    def _markers(self, values, now):
        """Tape marks are measured with the odometer speed, so a mark crossed while braking for the car
        ahead (or while blocked) is still measured correctly."""
        self.odo.update(now)
        kind = self.markers.update(values, self.odo.v)
        if kind == "approach" and self.mode == "follow" and self.agent.zone == "CRUISE":
            self.agent.on_approach_marker(now)
            self.s_approach = self.odo.s
            self.events.append((now, f"approach {self.agent.junction}:{self.agent.lane}"))
        elif kind == "stopline" and self.mode == "follow":
            if (self.agent.zone == "APPROACH" and self.s_approach is not None
                    and self.odo.s - self.s_approach < self.min_approach_to_stop):
                self.events.append((now, "ignored implausible stop line (too close to the approach marker)"))
                return
            self.mode = "hold"
            self.s_stopline = self.odo.s
            self.events.append((now, f"stop line {self.agent.junction}:{self.agent.lane}"))

    def step(self, values, now, sensors_fresh=True, range_m=None):
        """values: 4 ints left->right; range_m: forward ultrasonic distance (None = no reading).
        Returns (v, w). A car ahead closer than obstacle_stop_m stops the car (hysteresis to obstacle_clear_m);
        during a junction maneuver the dead-reckoning pauses while stopped."""
        v = w = 0.0
        if range_m is not None:
            self.blocked = range_m < (self.obstacle_clear if self.blocked else self.obstacle_stop)
        if self.blocked and self.mode != "fault":
            if self.maneuver is not None:
                self.maneuver.last_t = now                  # do not count stopped time as distance
            self.agent.set_health(True, True)
            if sensors_fresh and self.mode in ("follow", "hold"):
                self._markers(values, now)
            self.cmd = (0.0, 0.0)
            self.odo.command(0.0, now)
            return 0.0, 0.0
        self.agent.set_health(True, False)
        if not sensors_fresh or self.mode == "fault":
            pass
        elif self.mode == "recover":
            v, w = self.maneuver.update(values, now)
            if self.maneuver.done:
                self.events.append((now, "line re-acquired"))
                self.mode, self.maneuver = "follow", None
            elif self.maneuver.failed:
                self.mode = "fault"
                self.events.append((now, "FAULT line not found after junction (search)"))
        elif self.mode == "maneuver":
            v, w = self.maneuver.update(values, now)
            if self.maneuver.done:
                self.agent.on_left_box(now)
                self.events.append((now, f"left box onto {self.agent.edge}"))
                self.mode, self.maneuver = "follow", None
                self.s_handoff, self.recover_left = self.odo.s, 2
            elif self.maneuver.failed:
                self.mode = "fault"
                self.events.append((now, "FAULT line not found after junction"))
        else:
            fv, fw, line_ok = self.follower.update(values, now)
            self.agent.set_health(line_ok, False)
            self._markers(values, now)
            if self.mode == "hold":
                if self.agent.zone == "CRUISE":            # missed the approach marker: still ask first
                    self.agent.on_approach_marker(now)
                if self.cmd[0] < 0.01 and self.agent.zone == "APPROACH":
                    self.agent.on_stopped_at_stopline(now)
                if self.agent.may_cross(now) and self.agent.commit(now):
                    self._start_maneuver(now)
                    v, w = self.maneuver.update(values, now)
            else:
                v, w = (fv, fw) if line_ok else (0.0, 0.0)
                if range_m is not None and range_m < self.obstacle_slow:     # close in slowly (curves hide cars)
                    f = (range_m - self.obstacle_stop) / max(self.obstacle_slow - self.obstacle_stop, 1e-6)
                    v *= min(1.0, max(0.4, f))
                just_left = self.s_handoff is not None and self.odo.s - self.s_handoff < 0.30
                if not line_ok and just_left and self.recover_left > 0:
                    # the "line" that ended the maneuver was the edge of a cross tape: search again
                    self.recover_left -= 1
                    self.maneuver = JunctionManeuver({"turn": "straight", "d1": 0.0, "angle_deg": 0.0,
                                                      "d2_min": 0.0, "d2_max": 0.0}, dark=self.dark,
                                                     turn_scale=self.turn_scale, odometer=self.odo)
                    self.maneuver.phase = "search"
                    self.mode = "recover"
                    self.events.append((now, "line lost right after junction: search"))
                elif not line_ok and self.agent.zone == "APPROACH":
                    self.mode = "fault"                     # line ends after the approach marker: stop
                    self.events.append((now, "FAULT missed stop line (line ended in APPROACH)"))
                elif not line_ok:
                    self.events.append((now, "line lost"))
        self.cmd = (v, w)
        if self.mode not in ("maneuver", "recover"):                        # the maneuver feeds the odometer itself
            self.odo.command(v, now)
        return v, w
