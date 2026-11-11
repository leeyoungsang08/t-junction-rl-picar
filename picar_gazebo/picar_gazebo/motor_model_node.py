"""Motor imperfections between the car brain and Gazebo's DiffDrive (the car's hidden drivetrain).

cmd_vel (Twist, m/s rad/s) -> gz_cmd_vel (Twist) with: command latency, deadband (minimum wheel speed), left/right
gain mismatch, battery (speed scale that sags over the run), in-place spin efficiency that varies from spin to
spin, slow speed ripple. Same model and parameter names as picar_gazebo.virtual_fleet.VirtualCar.physics (2D
rehearsal); course.launch.py fills them per car from fleet:=fleet_truth.yaml. Defaults: a mildly imperfect car.
"""
import math
import random
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from rclpy.node import Node


class MotorModel(Node):
    def __init__(self):
        super().__init__("motor_model")
        p = lambda n, d: float(self.declare_parameter(n, d).value)
        self.latency = p("latency_s", 0.05)
        self.deadband = p("deadband_mps", 0.03)
        self.gain_l, self.gain_r = p("gain_left", 1.0), p("gain_right", 0.97)
        self.scale, self.sag = p("speed_scale", 1.0), p("sag_per_min", 0.0)
        self.spin_eff, self.spin_sd, self.spin_stick = p("spin_eff", 1.0), p("spin_sd", 0.0), p("spin_stick_prob", 0.0)
        self.ripple = p("ripple", 0.0)
        self.track = p("track_m", 0.17)
        self.rng = random.Random(int(p("seed", 0.0)))
        self.spinning, self.spin_e, self.rip, self.t0, self.last = False, 1.0, 0.0, None, (0.0, 0.0)
        self.q = deque()
        self.pub = self.create_publisher(Twist, "gz_cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(0.01, self.on_timer)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_cmd(self, msg):
        self.q.append((self._now() + self.latency, msg.linear.x, msg.angular.z))

    def _wheel(self, s, gain, scale):
        return 0.0 if abs(s) < self.deadband else s * gain * scale

    def on_timer(self):
        now = self._now()
        if self.t0 is None:
            self.t0 = now
        fresh = False
        while self.q and self.q[0][0] <= now:
            _, v, w = self.q.popleft()
            spin = abs(w) > 0.3 and abs(v) < 0.02
            if spin and not self.spinning:                    # every spin scrubs a little differently
                e = self.rng.gauss(self.spin_eff, self.spin_sd) if self.spin_sd > 0 else self.spin_eff
                if self.rng.random() < self.spin_stick:
                    e *= 0.92
                self.spin_e = min(1.05, max(0.5, e))
            self.spinning = spin
            scale = max(0.5, self.scale - self.sag * (now - self.t0) / 60.0)
            half = w * self.track / 2 * (self.spin_e if spin else 1.0)
            vl = self._wheel(v - half, self.gain_l, scale)
            vr = self._wheel(v + half, self.gain_r, scale)
            self.last, fresh = ((vl + vr) / 2, (vr - vl) / self.track), True
        if self.ripple > 0:                                   # correlation time ~0.5 s, as in the 2D model
            self.rip += (-self.rip / 0.5) * 0.01 + self.ripple * math.sqrt(2 * 0.01 / 0.5) * self.rng.gauss(0, 1)
        if fresh or (self.ripple > 0 and abs(self.last[0]) > 1e-6):
            out = Twist()
            out.linear.x, out.angular.z = self.last[0] * (1.0 + self.rip), self.last[1]
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MotorModel()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass                                  # launch sends SIGINT at the end of a run: normal shutdown
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
