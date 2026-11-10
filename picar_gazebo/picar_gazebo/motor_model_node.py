"""Motor imperfections between the car brain and Gazebo's DiffDrive (sim-to-real knobs).

cmd_vel (Twist, m/s rad/s) -> gz_cmd_vel (Twist) with: command latency, deadband (minimum wheel speed),
left/right gain mismatch, speed scale (battery). Defaults are mild guesses; replace with measurements.
"""
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from rclpy.node import Node


class MotorModel(Node):
    def __init__(self):
        super().__init__("motor_model")
        p = lambda n, d: self.declare_parameter(n, d).value
        self.latency = float(p("latency_s", 0.05))
        self.deadband = float(p("deadband_mps", 0.03))
        self.gain_l, self.gain_r = float(p("gain_left", 1.0)), float(p("gain_right", 0.97))
        self.scale = float(p("speed_scale", 1.0))
        self.track = float(p("track_m", 0.17))
        self.q = deque()
        self.pub = self.create_publisher(Twist, "gz_cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(0.01, self.on_timer)

    def on_cmd(self, msg):
        self.q.append((self.get_clock().now().nanoseconds * 1e-9 + self.latency, msg.linear.x, msg.angular.z))

    def _wheel(self, s, gain):
        s *= gain * self.scale
        return 0.0 if abs(s) < self.deadband else s

    def on_timer(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        while self.q and self.q[0][0] <= now:
            _, v, w = self.q.popleft()
            vl = self._wheel(v - w * self.track / 2, self.gain_l)
            vr = self._wheel(v + w * self.track / 2, self.gain_r)
            out = Twist()
            out.linear.x, out.angular.z = (vl + vr) / 2, (vr - vl) / self.track
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MotorModel()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass                                  # launch sends SIGINT at the end of a run: normal shutdown
    finally:
        pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
