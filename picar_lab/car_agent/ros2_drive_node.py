"""ROS2 'car brain' for one PiCar: line following + markers + intersection agent + junction maneuvers.

Runs unchanged in Gazebo (picar_gazebo) and on the Pi. Publishes cmd_vel in physical units (m/s, rad/s);
on the Pi pair it with motor_diff_node.py (the legacy Yahboom motor_node cannot steer while driving).

    line_values (Int32MultiArray, 4 ch left->right) --> [this node] --> cmd_vel (Twist)
    range (sensor_msgs/Range, forward ultrasonic)  --/
                                                          <==UDP v2==> picar_sim.central_server

Parameters: car, route ("W_in,E_out,..." cyclic), start_index, central_host, central_port, maneuvers_file,
            speed_mps, dark_value (sensor value on black tape), turn_rate_dps, approach_mark_max_m, stop_line_min_m,
            calibration_file (car_calibration.yaml: trim/turn_scale/latency/braking; see calibration.py), kp, kd
Safety: default STOP; stops on line loss, maneuver timeout, or stale sensor data. Hardware E-STOP stays separate.
"""
import socket
import time

import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Int32MultiArray, String

from picar_lab.car_agent.agent import CarAgent
from picar_lab.car_agent.brain import CarBrain
from picar_lab.car_agent.calibration import load as load_calibration
from picar_lab.car_agent.protocol import decode, encode


class DriveNode(Node):
    def __init__(self):
        super().__init__("picar_drive")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.agent = CarAgent(p("car", "car0"), p("route", "W_in,E_out,N_in,W_out").split(","), int(p("start_index", 0)))
        self.addr = (p("central_host", "127.0.0.1"), int(p("central_port", 5007)))
        self.speed = float(p("speed_mps", 0.25))
        self.dark = int(p("dark_value", 1))
        self.turn_rate = float(p("turn_rate_dps", 90.0))
        self.sensor_timeout = float(p("sensor_timeout_s", 0.3))
        with open(p("maneuvers_file", "maneuvers.yaml"), encoding="utf-8") as f:
            plans = yaml.safe_load(f)["maneuvers"]
        cal_file = p("calibration_file", "")
        cal = load_calibration(cal_file)
        if not cal_file:
            self.get_logger().warn("no calibration_file: junction turns use trim 0 / turn_scale 1 (expect misses)")
        self.brain = CarBrain(self.agent, plans, self.speed, self.dark, self.turn_rate,
                              float(p("approach_mark_max_m", 0.03)), float(p("stop_line_min_m", 0.04)), dt=0.05,
                              kp=float(p("kp", 1.0)), kd=float(p("kd", 0.0)), **cal)
        self.get_logger().info(f"calibration {cal}")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.values, self.values_t = [0, 0, 0, 0], 0.0
        self.range_m, self.range_t = None, 0.0
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.state_pub = self.create_publisher(String, "drive_state", 10)
        self.create_subscription(Int32MultiArray, "line_values", self.on_line, 10)
        self.create_subscription(Range, "range", self.on_range, 10)        # forward ultrasonic
        self.create_timer(0.05, self.on_timer)
        self.agent.register(time.monotonic())
        self.get_logger().info(f"drive node {self.agent.car} route={self.agent.route} boot={self.agent.boot}")

    def on_line(self, msg):
        self.values, self.values_t = list(msg.data)[:4], time.monotonic()

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
        fresh = now - self.values_t <= self.sensor_timeout
        n_events = len(self.brain.events)
        rng = self.range_m if now - self.range_t <= self.sensor_timeout else None
        v, w = self.brain.step(self.values, now, sensors_fresh=fresh, range_m=rng)
        for t, text in self.brain.events[n_events:]:
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
    node = DriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        rclpy.shutdown()


if __name__ == "__main__":
    main()
