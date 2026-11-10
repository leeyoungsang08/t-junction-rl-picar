"""Exit beams + measurements from Gazebo ground-truth poses.

* Exit beams: when a car's REAR passes an exit line (box edge + exit_line_m) leaving the box, send the same
  UDP exit_event a real beam sensor node would send (no car id), plus beam heartbeats for both junctions.
  Optional beam_fault_start/stop (s) silences the J heartbeat to test the central supervisor.
* Measurements (written to <out_dir>/metrics.json every 5 s and on exit):
  collisions (footprint overlap onsets), lane departures (> lane_tolerance_m from the lane centre outside boxes),
  stop errors at stop lines (front bumper vs. stop-line far edge, + = stopped before the line), distance driven.
"""
import json
import math
import socket
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Range

from picar_gazebo.core import Course, crossed, footprint, forward_range, overlap, yaw_from_quat
from picar_lab.car_agent.protocol import VERSION, encode


class CourseMonitor(Node):
    def __init__(self):
        super().__init__("course_monitor")
        p = lambda n, d: self.declare_parameter(n, d).value
        self.meta = yaml.safe_load(Path(p("meta_file", "course_meta.yaml")).read_text())
        self.out = Path(p("out_dir", "."))
        self.addr = (p("central_host", "127.0.0.1"), int(p("central_port", 5007)))
        self.fault = (float(p("beam_fault_start", -1.0)), float(p("beam_fault_stop", -1.0)))
        self.lane_tol = float(p("lane_tolerance_m", 0.045))   # world_gen: car body covers a whole lane line
        self.course = Course()
        car = self.meta["car"]
        self.L, self.W = car["length"], car["width"]
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.t0 = time.monotonic()
        self.cars = [c["name"] for c in self.meta["cars"]]
        self.pose, self.prev_rear, self.prev_t = {}, {}, {}
        self.m = {"collisions": 0, "lane_departures": 0, "exit_events": 0, "stop_errors_m": [], "distance_m": {},
                  "min_gap_m": None}
        self.in_contact, self.departed, self.stopped_since, self.stop_logged = set(), set(), {}, set()
        self.range_pub = {}
        for c in self.cars:
            self.create_subscription(PoseStamped, f"/model/{c}/pose", lambda msg, c=c: self.on_pose(c, msg), 10)
            self.range_pub[c] = self.create_publisher(Range, f"/{c}/range", 10)   # virtual ultrasonic
            self.m["distance_m"][c] = 0.0
        self.create_timer(0.05, self.on_timer)
        self.create_timer(5.0, self.dump)

    def _send(self, kind, junction, **extra):
        self.seq += 1
        self.sock.sendto(encode({"v": VERSION, "type": kind, "node": f"gz-beam-{junction}", "seq": self.seq,
                                 "junction": junction, **extra}), self.addr)

    def on_pose(self, car, msg):
        q = msg.pose.orientation
        x, y, yaw = msg.pose.position.x, msg.pose.position.y, yaw_from_quat(q.x, q.y, q.z, q.w)
        t = time.monotonic()
        if car in self.pose:
            px, py, _ = self.pose[car]
            step = math.hypot(x - px, y - py)
            self.m["distance_m"][car] += step
            speed = step / max(t - self.prev_t[car], 1e-3)
            self._stop_check(car, x, y, yaw, speed, t)
        self.pose[car], self.prev_t[car] = (x, y, yaw), t
        rear = np.array([x - math.cos(yaw) * self.L / 2, y - math.sin(yaw) * self.L / 2])
        for b in self.meta["beams"]:
            if crossed(self.prev_rear.get(car), rear, np.array(b["point"]), np.array(b["normal"]), self.course.lw / 2):
                self._send("exit_event", b["junction"], edge=b["edge"])
                self.m["exit_events"] += 1
        self.prev_rear[car] = rear
        centre = np.array([x, y])
        if not self.course.in_box(centre, margin=0.05):
            off = self.course.lane_distance(centre) > self.lane_tol
            if off and car not in self.departed:
                self.m["lane_departures"] += 1
                self.departed.add(car)
            elif not off:
                self.departed.discard(car)

    def _stop_check(self, car, x, y, yaw, speed, t):
        if speed > 0.01:
            self.stopped_since.pop(car, None)
            self.stop_logged.discard(car)
            return
        self.stopped_since.setdefault(car, t)
        if t - self.stopped_since[car] < 0.5 or car in self.stop_logged:
            return
        front = np.array([x + math.cos(yaw) * self.L / 2, y + math.sin(yaw) * self.L / 2])
        heading = np.array([math.cos(yaw), math.sin(yaw)])
        for s in self.meta["stoplines"]:
            n = np.array(s["normal"])
            d = float(np.dot(np.array(s["far"]) - front, n))         # + : front is before the line
            lateral = abs(float(np.dot(front - np.array(s["far"]), np.array([n[1], -n[0]]))))
            if -0.15 < d < 0.4 and lateral < self.course.lw / 2 and float(np.dot(heading, n)) > 0.8:
                self.m["stop_errors_m"].append(round(d, 4))
                self.stop_logged.add(car)
                break

    def on_timer(self):
        now = time.monotonic() - self.t0
        for jn in ("J", "J2"):
            if not (jn == "J" and self.fault[0] <= now < self.fault[1]):
                self._send("beam_heartbeat", jn, ok=True)
        cars = [c for c in self.cars if c in self.pose]
        for c in cars:
            r = Range()
            r.header.stamp = self.get_clock().now().to_msg()
            r.radiation_type, r.field_of_view, r.min_range, r.max_range = Range.ULTRASOUND, 0.52, 0.02, 1.0
            r.range = float(forward_range(self.pose[c], [self.pose[o] for o in cars if o != c], self.L, self.W))
            self.range_pub[c].publish(r)
        gaps = []
        for i, a in enumerate(cars):
            for b in cars[i + 1:]:
                fa, fb = footprint(*self.pose[a], self.L, self.W), footprint(*self.pose[b], self.L, self.W)
                gaps.append(math.dist(self.pose[a][:2], self.pose[b][:2]))
                key = (a, b)
                if overlap(fa, fb):
                    if key not in self.in_contact:
                        self.m["collisions"] += 1
                        self.in_contact.add(key)
                else:
                    self.in_contact.discard(key)
        if gaps:
            g = min(gaps)
            self.m["min_gap_m"] = g if self.m["min_gap_m"] is None else min(self.m["min_gap_m"], g)

    def dump(self):
        errs = self.m["stop_errors_m"]
        summary = {**self.m, "elapsed_s": round(time.monotonic() - self.t0, 1),
                   "stop_error_mean_m": round(float(np.mean(errs)), 4) if errs else None,
                   "stop_error_p95_abs_m": round(float(np.percentile(np.abs(errs), 95)), 4) if errs else None}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "metrics.json").write_text(json.dumps(summary, indent=1))


def main(args=None):
    rclpy.init(args=args)
    node = CourseMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass                                  # launch sends SIGINT at the end of a run: normal shutdown
    finally:
        node.dump()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
