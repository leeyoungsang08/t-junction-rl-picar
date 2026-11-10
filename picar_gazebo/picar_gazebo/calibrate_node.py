"""Gazebo: measure car_calibration.yaml for one simulated car from its ground-truth pose.

Runs the same open-loop tests as the real car (picar_lab/car_agent/calibration.py TESTS) through the motor
model, reads /model/<car>/pose, and writes the file the drive nodes load (calibration_file parameter).
Sequence: straight, spin, brake_slow, spin, brake_fast, spin (spins are averaged).
"""
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node

from picar_gazebo.core import yaw_from_quat
from picar_lab.car_agent.calibration import TESTS, from_measurements, save


class Calibrate(Node):
    SEQ = ["straight", "spin", "brake_slow", "spin", "brake_fast", "spin"]

    def __init__(self):
        super().__init__("picar_calibrate")
        car = self.declare_parameter("car", "car0").value
        self.out = self.declare_parameter("out_file", "car_calibration.yaml").value
        self.settle = float(self.declare_parameter("settle_s", 1.5).value)
        self.pub = self.create_publisher(Twist, f"/{car}/cmd_vel", 10)
        self.create_subscription(PoseStamped, f"/model/{car}/pose", self.on_pose, 10)
        self.pose, self.yaw_unwrapped, self.last_yaw = None, 0.0, None
        self.i, self.phase, self.t0, self.mark = 0, "wait", time.monotonic() + 3.0, None
        self.res = {"spin": []}
        self.create_timer(0.05, self.on_timer)

    def on_pose(self, msg):
        q = msg.pose.orientation
        yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if self.last_yaw is not None:
            self.yaw_unwrapped += math.atan2(math.sin(yaw - self.last_yaw), math.cos(yaw - self.last_yaw))
        self.last_yaw = yaw
        self.pose = (msg.pose.position.x, msg.pose.position.y, self.yaw_unwrapped)

    def snap(self):
        return self.pose

    def on_timer(self):
        now = time.monotonic()
        cmd = Twist()
        if self.pose is None:
            self.t0 = now + 3.0
        elif self.phase == "wait" and now >= self.t0:
            self.phase, self.t0, self.mark = "drive", now, {"start": self.snap()}
        elif self.phase == "drive":
            name = self.SEQ[self.i]
            (v, w), dur = TESTS[name]
            if now - self.t0 < dur:
                cmd.linear.x, cmd.angular.z = float(v), float(w)
            else:
                self.mark["stop"] = self.snap()
                self.phase, self.t0 = "settle", now
        elif self.phase == "settle" and now - self.t0 >= self.settle:
            self.record(self.SEQ[self.i], self.mark["start"], self.mark["stop"], self.snap())
            self.i += 1
            if self.i == len(self.SEQ):
                self.finish()
                return
            self.phase, self.t0 = "wait", now + 0.5
        self.pub.publish(cmd)

    def record(self, name, start, stop, end):
        dist = math.hypot(end[0] - start[0], end[1] - start[1])
        if name == "straight":
            self.res["straight_m"], self.res["drift_deg"] = dist, math.degrees(end[2] - start[2])
        elif name.startswith("brake"):
            self.res[f"roll_{name.split('_')[1]}_m"] = math.hypot(end[0] - stop[0], end[1] - stop[1])
        else:
            self.res["spin"].append(math.degrees(end[2] - start[2]))
        self.get_logger().info(f"{name}: {self.res}")

    def finish(self):
        r = self.res
        cal = from_measurements(r["straight_m"], r["drift_deg"], r["roll_slow_m"], r["roll_fast_m"], r["spin"])
        path = save(self.out, cal, "gazebo calibrate_node")
        self.get_logger().info(f"calibration {cal} -> {path}")
        self.pub.publish(Twist())
        raise SystemExit(0)


def main(args=None):
    rclpy.init(args=args)
    node = Calibrate()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.pub.publish(Twist())
        rclpy.shutdown()


if __name__ == "__main__":
    main()
