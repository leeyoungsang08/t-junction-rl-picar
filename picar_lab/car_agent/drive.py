"""Driving logic shared by Gazebo and the real PiCar (standard library only).

LineFollower    : 4-channel line sensor -> (v, w) in m/s and rad/s, keeps going straight over cross tapes
JunctionManeuver: what to do inside a T junction box, where there is no tape (assumption; see docs).
                  Plans come from maneuvers.yaml produced from the course geometry
                  (picar_gazebo/world_gen.py writes it for the same course.yaml the SUMO map uses).
                  Execution is dead reckoning: drive d1, turn in place, drive until the line is found again
                  (Odometer models latency + braking; trim/turn_scale/brake come from a per-car calibration).
wheel_pwm       : (v, w) -> left/right PWM duty for a differential-drive PiCar, with deadband.
The legacy Yahboom motor_node can only drive straight or spin; motor_diff_node.py uses wheel_pwm instead.
"""
import math


class LineFollower:
    """PD on the line position across the 4 sensors (weights = sensor positions in units of the spacing)."""

    def __init__(self, speed_mps=0.25, kp=1.0, kd=0.0, w_max=3.0, lost_timeout_s=0.5, dark=1,
                 weights=(-1.5, -0.5, 0.5, 1.5)):
        self.speed, self.kp, self.kd, self.w_max = speed_mps, kp, kd, w_max
        self.lost_timeout, self.dark, self.weights = lost_timeout_s, dark, weights
        self.last_err, self.lost_since = 0.0, None
        self.prev_err, self.prev_t = None, None

    def update(self, values, now):
        """values: 4 ints left->right. Returns (v, w, line_ok)."""
        on = [1 if v == self.dark else 0 for v in values]
        n = sum(on)
        if n >= 3:                                   # cross tape (marker/stop line): hold heading
            self.lost_since = None
            return self.speed, 0.0, True
        if n == 0:
            if self.lost_since is None:
                self.lost_since = now
            if now - self.lost_since > self.lost_timeout:
                return 0.0, 0.0, False                # line lost: stop
            return self.speed * 0.5, max(-self.w_max, min(self.w_max, self.kp * 1.5 * self.last_err)), True
        self.lost_since = None
        err = sum(w for w, o in zip(self.weights, on) if o) / n      # >0: line is to the right
        derr = 0.0
        if self.prev_t is not None and now > self.prev_t:
            derr = (err - self.prev_err) / (now - self.prev_t)
        self.prev_err, self.prev_t = err, now
        w = max(-self.w_max, min(self.w_max, -(self.kp * err + self.kd * derr)))
        self.last_err = -err
        v = self.speed * max(0.4, 1.0 - abs(err) / 2.0)                # slow down in sharp corrections
        return v, w, True


class Odometer:
    """Distance estimate from the commanded speed, without wheel encoders.

    Model: a command takes effect after latency_s, then the speed ramps at accel_mps2. Both come from the
    stop-roll calibration (calibrate_brake). stop_distance() is how far the car still rolls if it is told
    to stop now, so a maneuver can issue the stop early and come to rest on its target point.
    """

    def __init__(self, latency_s=0.05, accel_mps2=1.0, speed_scale=1.0):
        self.latency, self.acc = latency_s, accel_mps2
        self.scale = speed_scale      # metres really driven per commanded metre (calibration straight run);
        #                               the brain commands cmd(v), so the odometer itself counts the wanted speed
        self.queue, self.target, self.v, self.s, self.t = [], 0.0, 0.0, 0.0, None

    def command(self, v, now):
        self.update(now)
        self.queue.append((now + self.latency, v))

    def update(self, now):
        if self.t is None:
            self.t = now
        while self.t < now - 1e-9:
            dt = min(0.01, now - self.t)
            while self.queue and self.queue[0][0] <= self.t + 1e-9:
                self.target = self.queue.pop(0)[1]
            dv = max(-self.acc * dt, min(self.acc * dt, self.target - self.v))
            self.s += (self.v + dv / 2.0) * dt
            self.v += dv
            self.t += dt
        return self.s

    def stop_distance(self):
        pending = self.queue[-1][1] if self.queue else self.target
        v = max(self.v, 0.0)
        return v * self.latency + v * v / (2.0 * self.acc) if pending > 0 or v > 0 else 0.0

    def cmd(self, v):
        """Speed to command so that the car really drives v (weak battery / slow motors)."""
        return v / self.scale


class JunctionManeuver:
    """Executes one plan: {"turn": "straight"|"left"|"right", "d1": m, "angle_deg": deg, "d2_min": m, "d2_max": m}

    d1 is measured from the point where the stop line was detected (odometer reading s_ref), so the
    roll-out after a stop at the line is accounted for. Open-loop driving needs per-car calibration
    (docs: Gazebo/real-car calibration):
      trim_per_m : rad of yaw correction per metre of straight travel (cancels left/right motor mismatch)
      turn_scale : actual angle / commanded angle of an in-place spin (latency, slip, rate error)
      odometer   : latency + braking model, so the spin happens on the turn point (calibrate_brake)
    If the lane line is not under the sensors after d2_max, a slow search sweep (+-search_deg) looks
    for it before declaring failure. Three or four sensors dark is a cross tape, not the lane line;
    the line must be seen in confirm_frames consecutive frames (IR noise).
    """

    def __init__(self, plan, speed_mps=0.2, turn_rate_dps=90.0, dark=1, trim_per_m=0.0, turn_scale=1.0,
                 odometer=None, s_ref=None, search_deg=60.0, search_rate_dps=60.0, confirm_frames=2):
        self.plan, self.v, self.rate = plan, speed_mps, math.radians(turn_rate_dps)
        self.dark, self.trim, self.turn_scale = dark, trim_per_m, max(0.3, turn_scale)
        self.odo = odometer if odometer is not None else Odometer(latency_s=0.0, accel_mps2=1e9)
        self.search_max, self.search_rate = math.radians(search_deg), math.radians(search_rate_dps)
        self.phase, self.done, self.failed = "d1", False, False
        self.seen_frames, self.confirm = 0, confirm_frames
        self.s_ref = s_ref                 # odometer reading at the stop line (None: start of maneuver)
        self.s_d2 = None
        self.angle = 0.0
        self.sweep, self.sweep_leg = 0.0, 0
        self.last_t = None

    @property
    def dist(self):
        """Distance driven in the current phase (for logs)."""
        if self.phase == "d1":
            return self.odo.s - (self.s_ref or 0.0)
        return self.odo.s - self.s_d2 if self.s_d2 is not None else 0.0

    def _out(self, v, w, now):
        self.odo.command(v, now)
        return v, w

    ALIGN_DEG, ALIGN_M = 10.0, 0.08          # camera: the maneuver ends only this well lined up with the new lane
    spin_min = 0.7                           # rad/s: slowest on-the-spot rotation that surely beats the deadband

    def update(self, values, now, seen=None, lane=None):
        """Returns (v, w). Sets .done when the line is found after the box, .failed when the search gives up.
        seen: lane re-acquired by another sensor (front camera); None = use the IR values.
        lane: (offset_m, heading_deg) of the new lane from the front camera, None = not visible. With it the last
        part is closed loop: drive/rotate towards the lane and finish only when lined up. (Open loop the car left
        the box up to 25 deg skewed and then followed the wrong pair of lines.)"""
        dt = 0.0 if self.last_t is None else max(0.0, now - self.last_t)
        self.last_t = now
        self.odo.update(now)
        if self.s_ref is None:
            self.s_ref = self.odo.s
        p = self.plan
        if seen is None:
            n_dark = sum(1 for v in values if v == self.dark)
            seen = 0 < n_dark < 3
        if lane is not None:
            seen = abs(lane[1]) < self.ALIGN_DEG and abs(lane[0]) < self.ALIGN_M
        self.seen_frames = self.seen_frames + 1 if seen else 0
        line_seen = self.seen_frames >= self.confirm        # debounced; >= 3 dark is a cross tape
        steer = None                                        # lane visible but not lined up yet: turn towards it
        if lane is not None and abs(lane[0]) < 0.16 and abs(lane[1]) < 50.0:
            steer = max(-0.9, min(0.9, 1.5 * math.radians(lane[1]) + 4.0 * lane[0]))
        if self.phase == "d1":
            if p["turn"] == "straight":
                self.phase, self.s_d2 = "d2", self.s_ref        # straight: d2 counts from the stop line
            elif self.odo.s - self.s_ref + self.odo.stop_distance() >= p["d1"] - 1e-9:
                self.phase = "turn"
            else:
                return self._out(self.v, self.trim * self.v, now)
        if self.phase == "turn":
            target = math.radians(abs(p["angle_deg"])) / self.turn_scale
            self.angle += self.rate * dt
            if self.angle >= target - 1e-9:
                self.phase, self.s_d2 = "d2", self.odo.s
            else:
                return self._out(0.0, math.copysign(self.rate, p["angle_deg"]), now)
        if self.phase == "d2":
            d = self.odo.s - self.s_d2
            if d >= p["d2_min"] and line_seen:
                self.done = True
                return self._out(0.0, 0.0, now)
            if d >= p["d2_max"]:
                self.phase, self.sweep, self.sweep_leg = "search", 0.0, 0
                return self._out(0.0, 0.0, now)
            if steer is not None and d >= p["d2_min"]:
                return self._out(self.v * 0.6, steer, now)
            return self._out(self.v, self.trim * self.v, now)
        if self.phase == "search":
            if line_seen:
                self.done = True
                return self._out(0.0, 0.0, now)
            if steer is not None:                            # rotate on the spot until lined up
                # full spin rate or nothing: a gentle command (0.15 rad/s = 1 cm/s at the wheels) is below the
                # motors' deadband and the car just stands there (rehearsal: stuck in the box for good)
                if abs(lane[1]) >= self.ALIGN_DEG / 2 or abs(lane[0]) >= self.ALIGN_M:
                    return self._out(0.0, math.copysign(self.spin_min, steer), now)
                return self._out(self.v * 0.6, 0.0, now)     # lined up but the frame count is not there yet
            goal = (self.search_max, -self.search_max, 0.0)[self.sweep_leg]     # +max, -max, back to 0
            step = self.search_rate * dt / self.turn_scale
            if abs(goal - self.sweep) <= step:
                self.sweep = goal
                self.sweep_leg += 1
                if self.sweep_leg > 2:
                    self.failed = True
                return self._out(0.0, 0.0, now)
            sign = 1.0 if goal > self.sweep else -1.0
            self.sweep += sign * step
            return self._out(0.0, sign * self.search_rate, now)
        return self._out(0.0, 0.0, now)


def calibrate_trim(yaw_drift_rad, distance_m):
    """Straight open-loop run with trim 0: yaw changed by yaw_drift_rad (CCW +) over distance_m."""
    return -yaw_drift_rad / max(distance_m, 1e-6)


def calibrate_brake(roll_m, speed_mps, latency_s=0.05):
    """Car cruising at speed_mps is told to stop and rolls roll_m further -> effective decel (m/s^2)."""
    braking = max(roll_m - speed_mps * latency_s, 1e-4)
    return speed_mps ** 2 / (2.0 * braking)


def calibrate_brake2(roll1_m, v1_mps, roll2_m, v2_mps):
    """Roll-out measured at two cruise speeds -> (latency_s, accel_mps2) of  roll = v*L + v^2/(2a)."""
    # [v1, v1^2/2; v2, v2^2/2] [L, 1/a]^T = [roll1, roll2]
    det = v1_mps * v2_mps ** 2 / 2 - v2_mps * v1_mps ** 2 / 2
    L = (roll1_m * v2_mps ** 2 / 2 - roll2_m * v1_mps ** 2 / 2) / det
    inv_a = (v1_mps * roll2_m - v2_mps * roll1_m) / det
    L = min(max(L, 0.0), 0.3)
    if inv_a <= 1e-6:
        return L, calibrate_brake(roll2_m, v2_mps, L)
    return L, 1.0 / inv_a


def calibrate_turn_scale(measured_deg, commanded_deg=90.0):
    """In-place spin commanded for commanded_deg with turn_scale 1: measured_deg actually turned."""
    return abs(measured_deg) / abs(commanded_deg)


def movement(junction, arm_in, arm_out):
    """Turn direction for right-hand traffic on this course (J: stem points north, J2: stem points south)."""
    if arm_in == arm_out:
        raise ValueError("U-turns are not allowed")
    if {arm_in, arm_out} == {"W", "E"}:
        return "straight"
    left = {("J", "W", "N"), ("J", "N", "E"), ("J2", "E", "N"), ("J2", "N", "W")}
    return "left" if (junction, arm_in, arm_out) in left else "right"


def wheel_pwm(v, w, track_m=0.15, mps_per_pwm=0.004, deadband_pwm=25, max_pwm=100):
    """(v, w) -> (left, right) duty in [-max, max]. mps_per_pwm and deadband must be measured on the car."""
    out = []
    for side in (-1, 1):
        s = v + side * w * track_m / 2.0
        if abs(s) < 1e-4:
            out.append(0)
            continue
        duty = deadband_pwm + abs(s) / mps_per_pwm * (max_pwm - deadband_pwm) / max_pwm
        out.append(int(math.copysign(min(max_pwm, duty), s)))
    return tuple(out)
