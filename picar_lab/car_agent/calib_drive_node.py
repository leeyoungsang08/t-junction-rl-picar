"""Real PiCar: run ONE open-loop calibration test, then stop (pair with motor_diff_node.py).

    ros2 run ... or:  python3 -m picar_lab.car_agent.calib_drive_node --ros-args -p test:=straight
    tests: straight (1 m at 0.2 m/s) | brake_slow (0.15 m/s, then stop) | brake_fast (0.25 m/s) | spin (90 deg)

Mark the start with tape. After `straight` measure the distance and the heading change (drift-deg, + = left);
after brake_* measure how far the car rolled past the mark it was on when "STOP" was logged (or film it);
after `spin` measure the angle. Then:  python3 -m picar_lab.car_agent.calibration --help
Safety: lift the car or clear 2 m in front; Ctrl-C publishes zero velocity.
"""
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from picar_lab.car_agent.calibration import TESTS


class CalibDrive(Node):
    def __init__(self):
        super().__init__("picar_calib_drive")
        self.test = self.declare_parameter("test", "straight").value
        if self.test not in TESTS:
            raise SystemExit(f"test must be one of {list(TESTS)}")
        (self.v, self.w), self.dur = TESTS[self.test]
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.t0 = time.monotonic() + 2.0          # 2 s to step back
        self.stopped_at = None
        self.create_timer(0.05, self.on_timer)
        self.get_logger().info(f"{self.test}: v={self.v} w={self.w:.3f} for {self.dur:.2f} s, starting in 2 s")

    def on_timer(self):
        now = time.monotonic()
        t = Twist()
        if self.t0 <= now < self.t0 + self.dur:
            t.linear.x, t.angular.z = float(self.v), float(self.w)
        elif now >= self.t0 + self.dur and self.stopped_at is None:
            self.stopped_at = now
            self.get_logger().info("STOP")
        self.pub.publish(t)
        if self.stopped_at is not None and now - self.stopped_at > 2.0:
            raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = CalibDrive()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.pub.publish(Twist())
        rclpy.shutdown()


if __name__ == "__main__":
    main()
