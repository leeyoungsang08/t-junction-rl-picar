"""CameraBrain: the real PiCar's driving (UROP2): front camera lane following, IR stop-line detection,
ultrasonic stop, plus the intersection agent. No ROS: ros2_camera_drive_node.py wraps it for Gazebo / the Pi and
picar_gazebo.kinematic_check drives it in 2D with a camera rendered from the course texture.

* follow  : LaneFollower on the right lane line (left line if the right one is missing).
* APPROACH: no tape on the real course, so the request is sent after driving approach_after[edge] metres
            (odometry) since leaving the previous box; a car that reaches the stop line first asks there.
* stop    : >= 3 of 4 IR channels dark = stop line (lane lines are 10 cm to the sides); debounced by distance.
* box     : JunctionManeuver (odometry turn, no lines inside the box), ended when the camera sees the new lane.
Events (t, text) keep the CarBrain wording, so logs and checkers are shared.
"""
import math

from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.drive import JunctionManeuver, Odometer, movement
from picar_lab.car_agent.lane_camera import CameraModel, LaneDetector, LaneFollower


class CameraBrain:
    def __init__(self, agent: CarAgent, plans, approach_after, speed_mps=0.25, lane_width=0.21, camera=None,
                 dark=1, turn_rate_dps=90.0, obstacle_stop_m=0.20, obstacle_clear_m=0.25, obstacle_slow_m=0.40,
                 trim_per_m=0.0, turn_scale=1.0, latency_s=0.05, accel_mps2=1.0, approach_in_m=None,
                 stop_debounce_m=0.40, lost_fault_s=3.0, **_ignored):
        self.agent, self.plans, self.approach_after = agent, plans, approach_after
        self.speed, self.dark, self.turn_rate = speed_mps, dark, turn_rate_dps
        self.trim_per_m, self.turn_scale = trim_per_m, turn_scale
        self.detector = LaneDetector(CameraModel(camera), lane_width)
        self.follower = LaneFollower(speed_mps, lane_width)
        self.odo = Odometer(latency_s, accel_mps2)
        self.mode = "follow"            # follow | hold | maneuver | fault
        self.maneuver = None
        self.cmd = (0.0, 0.0)
        self.events = []
        self.obstacle_stop, self.obstacle_clear, self.obstacle_slow = obstacle_stop_m, obstacle_clear_m, obstacle_slow_m
        self.blocked = False
        self.s_stopline = None
        self.s_last_stop = -math.inf
        self.debounce = stop_debounce_m
        self.s_approach_at = approach_in_m        # odometer reading at which APPROACH is sent (None: at the line)
        self.lost_since, self.lost_fault_s = None, lost_fault_s
        self.lanes = {"right": None, "left": None}

    # ------------------------------------------------------------------ helpers
    def _stop_line(self, ir):
        return sum(1 for v in ir if v == self.dark) >= 3

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
                                         odometer=self.odo, s_ref=self.s_stopline, search_deg=40.0)
        self.mode = "maneuver"

    # ------------------------------------------------------------------ main step (20 Hz)
    def step(self, image, ir, now, sensors_fresh=True, range_m=None):
        """image: gray front-camera frame (H x W, 0..1 or 0..255) or None; ir: 4 ints left->right;
        range_m: forward ultrasonic distance (None = no reading). Returns (v, w)."""
        self.odo.update(now)
        if image is not None:
            self.lanes = self.detector.detect(image)
        v = w = 0.0
        if range_m is not None:
            self.blocked = range_m < (self.obstacle_clear if self.blocked else self.obstacle_stop)
            if self.maneuver is not None and self.maneuver.phase in ("turn", "search"):
                self.blocked = False      # turning on the spot does not move forward; the sweep of the cone over a
                #                           car waiting in the next lane would otherwise freeze the box (deadlock)
        if self.blocked and self.mode != "fault":
            if self.maneuver is not None:
                self.maneuver.last_t = now                   # stopped time is not distance
            self.agent.set_health(True, True)
            if sensors_fresh and self.mode == "follow" and self._stop_line(ir):
                self._on_stop_line(now)                      # rolled onto the line while braking
            self.cmd = (0.0, 0.0)
            self.odo.command(0.0, now)
            return 0.0, 0.0
        self.agent.set_health(True, False)
        if not sensors_fresh or self.mode == "fault":
            pass
        elif self.mode == "maneuver":
            seen = self.follower.aligned(self.lanes)
            v, w = self.maneuver.update(ir, now, seen=seen)
            if self.maneuver.done:
                self.agent.on_left_box(now)
                self.events.append((now, f"left box onto {self.agent.edge}"))
                self.mode, self.maneuver = "follow", None
                self.s_last_stop = self.odo.s
                d = self.approach_after.get(self.agent.edge)
                self.s_approach_at = None if d is None else self.odo.s + d
                self.follower.last_ok = now
            elif self.maneuver.failed:
                self.mode = "fault"
                self.events.append((now, "FAULT line not found after junction"))
        else:
            fv, fw, line_ok = self.follower.update(self.lanes, now)
            self.agent.set_health(line_ok, False)
            if self.mode == "follow":
                if (self.agent.zone == "CRUISE" and self.s_approach_at is not None
                        and self.odo.s >= self.s_approach_at):
                    self.agent.on_approach_marker(now)
                    self.events.append((now, f"approach {self.agent.junction}:{self.agent.lane}"))
                if self._stop_line(ir):
                    self._on_stop_line(now)
            if self.mode == "hold":
                if self.agent.zone == "CRUISE":            # reached the line before the odometry trigger
                    self.agent.on_approach_marker(now)
                    self.events.append((now, f"approach {self.agent.junction}:{self.agent.lane}"))
                if self.cmd[0] < 0.01 and self.agent.zone == "APPROACH":
                    self.agent.on_stopped_at_stopline(now)
                if self.agent.may_cross(now) and self.agent.commit(now):
                    self._start_maneuver(now)
                    v, w = self.maneuver.update(ir, now, seen=False)
            else:
                v, w = (fv, fw) if line_ok else (0.0, 0.0)
                if range_m is not None and range_m < self.obstacle_slow:     # close in slowly (curves hide cars)
                    f = (range_m - self.obstacle_stop) / max(self.obstacle_slow - self.obstacle_stop, 1e-6)
                    v *= min(1.0, max(0.4, f))
                if not line_ok:
                    if self.lost_since is None:
                        self.lost_since = now
                        self.events.append((now, "line lost"))
                    elif now - self.lost_since > self.lost_fault_s:
                        self.mode = "fault"
                        self.events.append((now, "FAULT lane lost"))
                else:
                    self.lost_since = None
        self.cmd = (v, w)
        if self.mode != "maneuver":                       # the maneuver feeds the odometer itself
            self.odo.command(v, now)
        return v, w

    def _on_stop_line(self, now):
        if self.mode != "follow" or self.odo.s - self.s_last_stop < self.debounce:
            return
        self.mode = "hold"
        self.s_stopline = self.s_last_stop = self.odo.s
        self.events.append((now, f"stop line {self.agent.junction}:{self.agent.lane}"))
