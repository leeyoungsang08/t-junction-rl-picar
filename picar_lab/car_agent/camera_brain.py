"""CameraBrain: the real PiCar's driving (UROP2): front camera lane following, IR stop-line detection,
ultrasonic stop, plus the intersection agent. No ROS: ros2_camera_drive_node.py wraps it for Gazebo / the Pi and
picar_gazebo.kinematic_check drives it in 2D with a camera rendered from the course texture.

* follow  : LaneFollower on the right lane line (left line if the right one is missing).
* APPROACH: no tape on the real course, so the request is sent after driving approach_after[edge] metres
            (odometry) since leaving the previous box; a car that reaches the stop line first asks there.
* stop    : >= 3 of 4 IR channels dark = stop line (lane lines are 10 cm to the sides); debounced by distance.
* box     : JunctionManeuver (odometry turn, no lines inside the box), ended when the camera sees the new lane;
            if the lane drops out again within coast_after_box_m the car keeps going straight slowly.
Events (t, text) keep the CarBrain wording, so logs and checkers are shared.
"""
import math

from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.drive import JunctionManeuver, Odometer, movement
from picar_lab.car_agent.lane_camera import CameraModel, LaneDetector, LaneFollower


class CameraBrain:
    def __init__(self, agent: CarAgent, plans, approach_after, speed_mps=0.25, lane_width=0.21, camera=None,
                 dark=1, turn_rate_dps=90.0, obstacle_stop_m=0.20, obstacle_clear_m=0.25, obstacle_slow_m=0.40,
                 trim_per_m=0.0, turn_scale=1.0, latency_s=0.05, accel_mps2=1.0, speed_scale=1.0, approach_in_m=None,
                 stop_debounce_m=0.40, lost_fault_s=3.0, coast_after_box_m=0.30, obstacle_box_m=0.09,
                 clear_box_m=0.40, stop_after=None, **_ignored):
        self.agent, self.plans, self.approach_after = agent, plans, approach_after
        self.speed, self.dark, self.turn_rate = speed_mps, dark, turn_rate_dps
        self.trim_per_m, self.turn_scale = trim_per_m, turn_scale
        self.detector = LaneDetector(CameraModel(camera), lane_width)
        self.follower = LaneFollower(speed_mps, lane_width)
        self.odo = Odometer(latency_s, accel_mps2, speed_scale)
        self.mode = "follow"            # follow | hold | maneuver | fault
        self.maneuver = None
        self.cmd = (0.0, 0.0)
        self.events = []
        self.obstacle_stop, self.obstacle_clear, self.obstacle_slow = obstacle_stop_m, obstacle_clear_m, obstacle_slow_m
        self.blocked = False
        self.ir_prev, self.stop_after, self.stop_after_m = 0, stop_after or {}, None
        self.align_left = math.radians(25.0)
        self.recover_left = math.radians(60.0)
        self.obstacle_box, self.clear_box = obstacle_box_m, clear_box_m
        self.s_stopline = None
        self.s_last_stop = -math.inf
        self.debounce = stop_debounce_m
        self.s_approach_at = approach_in_m        # odometer reading at which APPROACH is sent (None: at the line)
        self.lost_since, self.lost_fault_s = None, lost_fault_s
        self.s_left_box, self.coast_after_box = None, coast_after_box_m
        self.lanes = {"right": None, "left": None}

    # ------------------------------------------------------------------ helpers
    def _stop_line(self, ir):
        """All four channels dark, or three in two readings in a row. A single reading with three is not enough:
        a car that has drifted onto a lane line has two channels dark and one noise flip makes three (rehearsal:
        false stop line 0.75 m early, the car turned in the middle of the road). Lines this early after a box are
        ignored as well: the map says how far the next stop line is."""
        n = sum(1 for v in ir if v == self.dark)
        hit = n == 4 or (n >= 3 and self.ir_prev >= 3)
        self.ir_prev = n
        if hit and self.s_left_box is not None and self.stop_after_m is not None \
                and self.odo.s - self.s_left_box < 0.6 * self.stop_after_m:
            return False
        return hit

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
        range_m: forward ultrasonic distance (None = no reading). Returns (v, w) to command; v is raised by
        1/speed_scale so that a car with weak motors still drives the planned speed and distances."""
        v, w = self._step(image, ir, now, sensors_fresh, range_m)
        return self.odo.cmd(v), w

    def _step(self, image, ir, now, sensors_fresh, range_m):
        self.odo.update(now)
        if image is not None:
            if self.maneuver is not None and self.maneuver.phase in ("turn", "search"):
                # while the car spins, lines sweep across the view; "the line I tracked last frame" would carry the
                # divider into the right-line slot and the new lane is then never recognised (rehearsal: car left
                # the box 25 deg skewed and followed the opposite lane). Classify by side only.
                self.detector.prev = {"right": None, "left": None}
            self.lanes = self.detector.detect(image)
        v = w = 0.0
        clearing = False
        if range_m is not None:
            # Clearing a junction: inside the box and for clear_box_m after it the car must not wait for something
            # 20 cm away, because that is usually the car at the stop line of the opposite lane (4 cm beside this
            # lane, inside the +-15 deg ultrasonic cone as soon as the turn ends a few degrees off). Waiting there
            # keeps the box occupied and nobody gets a permit (deadlock seen in Gazebo and in the 6-car rehearsal).
            # The central PC only lets a car in when the exit lane has room, so only a real obstacle right in
            # front (obstacle_box_m) stops the car here.
            # Not on the way in (d1) or straight across: there a close echo is the car ahead from the same approach,
            # still turning in the box.
            leaving = self.maneuver is not None and self.maneuver.phase in ("d2", "search") \
                and self.maneuver.plan["turn"] != "straight"
            clearing = leaving or (self.maneuver is None and self.s_left_box is not None
                                   and self.odo.s - self.s_left_box < self.clear_box)
            stop_m = self.obstacle_box if clearing else self.obstacle_stop
            self.blocked = range_m < (stop_m + 0.05 if self.blocked else stop_m)
            if self.maneuver is not None and self.maneuver.phase in ("turn", "search"):
                self.blocked = False      # turning on the spot does not move forward; the sweep of the cone over a
                #                           car waiting in the next lane would otherwise freeze the box (deadlock)
        if self.blocked and self.mode != "fault":
            if self.maneuver is not None:
                self.maneuver.last_t = now                   # stopped time is not distance
            self.agent.set_health(True, True)
            if sensors_fresh and self.mode == "follow" and self._stop_line(ir):
                self._on_stop_line(now)                      # rolled onto the line while braking
            w = 0.0
            pose = self.follower.lane_pose(self.lanes)
            line = self.lanes.get("right") or self.lanes.get("left")
            if self.mode == "follow" and clearing and pose is not None and line["f1"] - line["f0"] > 0.25 \
                    and 6.0 < abs(pose[1]) < 30.0 and abs(pose[0]) < 0.16 and self.align_left > 0.0:
                # at most 25 deg per stop: with a car in the picture the "lane direction" can be wrong, and an
                # unlimited turn sent a car round and back through the junction (Gazebo fleet run)
                self.align_left -= JunctionManeuver.spin_min * 0.05
                # standing skewed in the lane (typically just out of a box, looking at the car waiting in the
                # opposite lane): line up with the lane on the spot; no forward motion while something is ahead
                w = math.copysign(JunctionManeuver.spin_min, pose[1])
            self.cmd = (0.0, w)
            self.odo.command(0.0, now)
            return 0.0, w
        self.agent.set_health(True, False)
        self.align_left = math.radians(25.0)
        if not sensors_fresh or self.mode == "fault":
            pass
        elif self.mode == "maneuver":
            v, w = self.maneuver.update(ir, now, seen=False, lane=self.follower.lane_pose(self.lanes) or (9.9, 99.0))
            if self.maneuver.done:
                self.agent.on_left_box(now)
                self.events.append((now, f"left box onto {self.agent.edge}"))
                self.mode, self.maneuver = "follow", None
                self.s_last_stop = self.s_left_box = self.odo.s
                self.stop_after_m = self.stop_after.get(self.agent.edge)
                d = self.approach_after.get(self.agent.edge)
                self.s_approach_at = None if d is None else self.odo.s + d
                self.follower.last_ok = now
            elif self.maneuver.failed:
                self.mode = "fault"
                self.events.append((now, "FAULT line not found after junction"))
        else:
            fv, fw, line_ok = self.follower.update(self.lanes, now, moving=self.mode == "follow" and self.cmd[0] > 0.05,
                                                   oncoming=self.detector.left_obstacle)
            # waiting at the stop line the lane ends 17 cm ahead (junction box): seeing no line there is normal and
            # must not block the crossing, which starts open loop anyway
            self.agent.set_health(line_ok or self.mode == "hold", False)
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
                if line_ok:
                    v, w = fv, fw + self.trim_per_m * fv      # measured motor mismatch as feed-forward
                elif self.s_left_box is not None and self.odo.s - self.s_left_box < self.coast_after_box:
                    # just out of a box: the car can still be over the unpainted box with the new lane's lines
                    # at the edge of the view; stopping here would never bring them back (Gazebo 2026-11-11:
                    # car stopped 0.1 m inside the box after a left turn -> FAULT lane lost -> both T's jammed)
                    v = min(self.speed, 0.15)
                    w = self.trim_per_m * v
                elif self.recover_left > 0.0 and abs(self.follower.last_k) > 1.0:
                    # lane lost in a curve (typically: stopped for the oncoming car at the apex, which then hides
                    # the divider while it passes): the lane went the way the car was steering, so turn that way
                    # on the spot, at most 60 deg, instead of standing there until FAULT
                    self.recover_left -= JunctionManeuver.spin_min * 0.05
                    self.detector.prev = {"right": None, "left": None}
                    v, w = 0.0, math.copysign(JunctionManeuver.spin_min, self.follower.last_k)
                else:
                    v, w = 0.0, 0.0
                if range_m is not None and range_m < self.obstacle_slow:     # close in slowly (curves hide cars)
                    f = (range_m - self.obstacle_stop) / max(self.obstacle_slow - self.obstacle_stop, 1e-6)
                    v, w = v * min(1.0, max(0.4, f)), w * min(1.0, max(0.4, f))   # same path, lower speed
                if not line_ok:
                    if self.lost_since is None:
                        self.lost_since = now
                        self.events.append((now, "line lost"))
                    elif now - self.lost_since > self.lost_fault_s:
                        self.mode = "fault"
                        self.events.append((now, "FAULT lane lost"))
                else:
                    self.lost_since = None
                    self.recover_left = math.radians(60.0)
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
