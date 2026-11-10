#!/usr/bin/env python3
"""Log Gazebo ground-truth poses of every car to <run_dir>/poses.csv (for the deadlock check).

python3 pose_logger.py <run_dir>      # waits for <run_dir>/course_meta.yaml written by course.launch.py
Columns: wall_s (monotonic, same clock as course_monitor), sim_s (pose header stamp), car, x, y
"""
import sys
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


class PoseLogger(Node):
    def __init__(self, run_dir, cars, period=0.5):
        super().__init__("picar_pose_logger")
        self.f = open(run_dir / "poses.csv", "w", buffering=1)
        self.f.write("wall_s,sim_s,car,x,y\n")
        self.last = {}
        self.period = period
        for c in cars:
            self.create_subscription(PoseStamped, f"/model/{c}/pose", lambda m, c=c: self.on_pose(c, m), 10)

    def on_pose(self, car, msg):
        now = time.monotonic()
        if now - self.last.get(car, -1e9) < self.period:
            return
        self.last[car] = now
        sim = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.position
        self.f.write(f"{now:.3f},{sim:.3f},{car},{p.x:.4f},{p.y:.4f}\n")


def main():
    run_dir = Path(sys.argv[1]).expanduser()
    meta = run_dir / "course_meta.yaml"
    for _ in range(600):
        if meta.exists():
            break
        time.sleep(0.1)
    cars = [c["name"] for c in yaml.safe_load(meta.read_text())["cars"]]
    rclpy.init()
    node = PoseLogger(run_dir, cars)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.f.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
