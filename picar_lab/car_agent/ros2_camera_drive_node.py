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
Time: the header stamp of the latest front-camera frame (simulated time in Gazebo, so odometry, turn timing and
timeouts stay right when the simulation runs slower than real time; capture time on the Pi). A frame without a stamp
falls back to the ROS clock. Not use_sim_time: /clock at 1 kHz into every car's Python nodes cost ~30 % RTF at 6 cars.
"""
import os
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
from picar_lab.car_agent.calibration import load as load_calibration, load_camera
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
        self.cam = load_camera(cal_file, self.cam)         # mount measured with camera_calib.py, if present
        # simulation only (Gazebo's camera is perfect): image noise sd, exposure gain, delay, dropped frames
        self.sim_cam = {k: float(p(f"sim_cam_{k}", d)) for k, d in
                        (("noise", 0.0), ("gain", 1.0), ("delay_s", 0.0), ("drop_prob", 0.0))}
        self.sim_frames = []
        if not cal_file:
            self.get_logger().warn("no calibration_file: junction turns use trim 0 / turn_scale 1 (expect misses)")
        appr = float(p("approach_in_m", -1.0))
        self.lane_width = float(man.get("lane_width_m", 0.21))
        self.brain = CameraBrain(self.agent, man["maneuvers"], man.get("approach_after_exit_m", {}),
                                 speed_mps=float(p("speed_mps", 0.25)), lane_width=self.lane_width, camera=self.cam,
                                 dark=int(p("dark_value", 1)), turn_rate_dps=float(p("turn_rate_dps", 90.0)),
                                 approach_in_m=None if appr < 0 else appr,
                                 stop_after=man.get("stop_after_exit_m"), **cal)
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
        self.registered = False                            # REGISTER on the first tick with a valid clock
        self.cam_stamp, self.cam_wall = 0.0, None          # latest frame time (s) and wall time it arrived
        # debugging: PICAR_DUMP_DIR=<dir> keeps 4 camera frames a second (raw uint8) and what the brain did with
        # them, so a run can be replayed offline (frames_<car>.u8 + frames_<car>.csv)
        self.dump, self.dump_n = None, 0
        if os.environ.get("PICAR_DUMP_DIR"):
            base = os.path.join(os.environ["PICAR_DUMP_DIR"], f"frames_{self.agent.car}")
            self.dump = (open(base + ".u8", "wb"), open(base + ".csv", "w", encoding="utf-8"))
            self.dump[1].write("t,h,w,mode,phase,zone,edge,v,w_cmd,range,blocked,ir,odo_s,right_l0,left_l0\n")
        self.create_timer(0.05, self.on_timer)
        self.get_logger().info(f"camera drive node {self.agent.car} route={self.agent.route} boot={self.agent.boot}")

    def _now(self):
        return self.cam_stamp

    def on_line(self, msg):
        self.values, self.values_t = list(msg.data)[:4], self._now()

    def on_image(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = self.get_clock().now().nanoseconds * 1e-9
        self.cam_stamp, self.cam_wall = max(self.cam_stamp, stamp), time.monotonic()
        gray = image_to_gray(msg)
        if gray.shape != (self.brain.detector.model.h, self.brain.detector.model.w):   # real camera resolution
            self.brain.detector = LaneDetector(CameraModel({**self.cam, "w": gray.shape[1], "h": gray.shape[0]}),
                                               self.lane_width)
        sc = self.sim_cam
        if sc["noise"] > 0 or sc["gain"] != 1.0:
            gray = np.clip(gray * sc["gain"] + np.random.normal(0, sc["noise"], gray.shape), 0.0, 1.0).astype(np.float32)
        if sc["drop_prob"] > 0 and np.random.random() < sc["drop_prob"]:
            return
        if sc["delay_s"] > 0:
            self.sim_frames.append((stamp, gray))
            ready = [f for f in self.sim_frames if f[0] <= stamp - sc["delay_s"] + 1e-6]
            self.sim_frames = [f for f in self.sim_frames if f[0] > stamp - sc["delay_s"] + 1e-6]
            if not ready:
                return
            gray = ready[-1][1]
        self.image, self.image_t = gray, self._now()

    def on_range(self, msg):
        self.range_m, self.range_t = float(msg.range), self._now()

    def on_timer(self):
        now = self._now()
        if now <= 0.0:
            return                                         # no camera frame yet
        if not self.registered:
            self.agent.register(now)
            self.registered = True
        while True:
            try:
                raw, _ = self.sock.recvfrom(2048)
            except BlockingIOError:
                break
            msg = decode(raw)
            if msg:
                self.agent.on_packet(msg, now)
        fresh = (now - self.values_t <= self.sensor_timeout and now - self.image_t <= self.sensor_timeout
                 and time.monotonic() - self.cam_wall <= 1.0)   # frames stopped: frame time stands still
        img = self.image if fresh else None
        self.image = None                                  # each camera frame is processed once
        rng = self.range_m if now - self.range_t <= self.sensor_timeout else None
        n_events = len(self.brain.events)
        v, w = self.brain.step(img, self.values, now, sensors_fresh=fresh, range_m=rng)
        for _, text in self.brain.events[n_events:]:
            self.get_logger().info(text)
        if self.dump and img is not None:
            self.dump_n += 1
            if self.dump_n % 5 == 0:
                b = self.brain
                l0 = [("" if b.lanes.get(k) is None else f"{b.lanes[k]['l0']:.3f}") for k in ("right", "left")]
                self.dump[0].write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
                self.dump[1].write(f"{now:.2f},{img.shape[0]},{img.shape[1]},{b.mode},"
                                   f"{b.maneuver.phase if b.maneuver else ''},{self.agent.zone},{self.agent.edge},"
                                   f"{v:.3f},{w:.3f},{'' if rng is None else round(rng, 3)},{int(b.blocked)},"
                                   f"{''.join(map(str, self.values))},{b.odo.s:.3f},{l0[0]},{l0[1]}\n")
                self.dump[0].flush(); self.dump[1].flush()
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
