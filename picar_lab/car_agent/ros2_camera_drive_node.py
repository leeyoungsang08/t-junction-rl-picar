"""ROS2 driving node of one PiCar as on the real car (UROP2): front-camera lane following, IR stop line,
ultrasonic stop, intersection agent (CameraBrain). Runs unchanged in Gazebo (picar_gazebo) and on the Pi.

    image (sensor_msgs/Image, front camera)        --\\
    line_values (Int32MultiArray, IR 4 ch l->r)    ----> [this node] --> cmd_vel (Twist), drive_state (String)
    range (sensor_msgs/Range, forward ultrasonic)  --/            <==UDP v2==> picar_sim.central_server

Parameters: car, route ("W_in,E_out,..." cyclic), start_index, central_host, central_port, maneuvers_file
            (maneuvers.yaml from world_gen: crossing plans + approach_after_exit_m + lane_width_m), approach_in_m
            (distance from the start position to the first APPROACH point; <0 = ask at the stop line),
            speed_mps, dark_value, turn_rate_dps, calibration_file, camera_x/camera_z/camera_pitch_deg/camera_hfov_deg
            (mount of the front camera, see lane_camera.CAMERA).
Safety: default STOP; stops on lane loss (FAULT after a few seconds), maneuver failure or stale sensor data.
"""
import socket
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image, Range
from std_msgs.msg import Int32MultiArray, String

from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.calibration import load as load_calibration
from picar_lab.car_agent.camera_brain import CameraBrain
from picar_lab.car_agent.lane_camera import CAMERA, CameraModel, LaneDetector
from picar_lab.car_agent.protocol import decode, encode


def image_to_gray(msg):
    ch = 3 if msg.encoding in ("rgb8", "bgr8") else (4 if msg.encoding in ("rgba8", "bgra8") else 1)
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch)
    return (img[..., :3].mean(axis=2) if ch > 1 else img[..., 0]).astype(np.float32) / 255.0


class CameraDriveNode(Node):
    def __init__(self):
        super().__init__("picar_camera_drive")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.agent = CarAgent(p("car", "car0"), p("route", "W_in,E_out,N_in,W_out").split(","), int(p("start_index", 0)))
        self.addr = (p("central_host", "127.0.0.1"), int(p("central_port", 5007)))
        self.sensor_timeout = float(p("sensor_timeout_s", 0.3))
        with open(p("maneuvers_file", "maneuvers.yaml"), encoding="utf-8") as f:
            man = yaml.safe_load(f)
        self.cam = {**CAMERA, "x": float(p("camera_x", CAMERA["x"])), "z": float(p("camera_z", CAMERA["z"])),
                    "pitch_deg": float(p("camera_pitch_deg", CAMERA["pitch_deg"])),
                    "hfov_deg": float(p("camera_hfov_deg", CAMERA["hfov_deg"]))}
        cal_file = p("calibration_file", "")
        cal = load_calibration(cal_file)
        if not cal_file:
            self.get_logger().warn("no calibration_file: junction turns use trim 0 / turn_scale 1 (expect misses)")
        appr = float(p("approach_in_m", -1.0))
        self.lane_width = float(man.get("lane_width_m", 0.21))
        self.brain = CameraBrain(self.agent, man["maneuvers"], man.get("approach_after_exit_m", {}),
                                 speed_mps=float(p("speed_mps", 0.25)), lane_width=self.lane_width, camera=self.cam,
                                 dark=int(p("dark_value", 1)), turn_rate_dps=float(p("turn_rate_dps", 90.0)),
                                 approach_in_m=None if appr < 0 else appr, **cal)
        self.get_logger().info(f"calibration {cal}  camera {self.cam}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.values, self.values_t = [0, 0, 0, 0], 0.0
        self.image, self.image_t = None, 0.0
        self.range_m, self.range_t = None, 0.0
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "drive_state", 10)
        self.create_subscription(Int32MultiArray, "line_values", self.on_line, 10)
        self.create_subscription(Image, "image", self.on_image, 2)
        self.create_subscription(Range, "range", self.on_range, 10)
        self.create_timer(0.05, self.on_timer)
        self.agent.register(time.monotonic())
        self.get_logger().info(f"camera drive node {self.agent.car} route={self.agent.route} boot={self.agent.boot}")

    def on_line(self, msg):
        self.values, self.values_t = list(msg.data)[:4], time.monotonic()

    def on_image(self, msg):
        gray = image_to_gray(msg)
        if gray.shape != (self.brain.detector.model.h, self.brain.detector.model.w):   # real camera resolution
            self.brain.detector = LaneDetector(CameraModel({**self.cam, "w": gray.shape[1], "h": gray.shape[0]}),
                                               self.lane_width)
        self.image, self.image_t = gray, time.monotonic()

    def on_range(self, msg):
        self.range_m, self.range_t = float(msg.range), time.monotonic()

    def on_timer(self):
        now = time.monotonic()
        while True:
            try:
                raw, _ = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            msg = decode(raw)
            if msg:
                self.agent.on_packet(msg, now)
        fresh = now - self.values_t <= self.sensor_timeout and now - self.image_t <= self.sensor_timeout
        img = self.image if fresh else None
        self.image = None                                  # each camera frame is processed once
        rng = self.range_m if now - self.range_t <= self.sensor_timeout else None
        n_events = len(self.brain.events)
        v, w = self.brain.step(img, self.values, now, sensors_fresh=fresh, range_m=rng)
        for _, text in self.brain.events[n_events:]:
            self.get_logger().info(text)
        for m in self.agent.tick(now):
            self.sock.sendto(encode(m), self.addr)
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)
        s = String()
        s.data = f"{self.brain.mode}:{self.agent.zone}:{self.agent.edge}"
        self.state_pub.publish(s)


def main(args=None):
    rclpy.init(args=args)
    node = CameraDriveNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass                                  # launch sends SIGINT at the end of a run: normal shutdown
    finally:
        if rclpy.ok():
            node.pub.publish(Twist())
            rclpy.shutdown()


if __name__ == "__main__":
    main()
