"""ROS2 intersection gate node for the Yahboom PiCar (NOT tested on hardware; no ROS2 in the dev workspace).

Wiring (replaces the direct line_follower -> motor_controller link):
    line_tracker --line_values--> [this node] <--cmd_vel_raw-- line_follower (remap cmd_vel -> cmd_vel_raw)
                                      |--cmd_vel--> motor_controller
                                      |<==UDP==> central PC (picar_sim.central_server)

    ros2 run line_follower line_follower_node --ros-args -r cmd_vel:=cmd_vel_raw
    python3 -m picar_lab.car_agent.ros2_gate_node --ros-args -p car:=W1 -p central_host:=192.168.50.10 \\
        -p route:="W_in,E_out,N_in,W_out" -p start_index:=0

Safety: default STOP. Motion passes only if CarAgent.motion_allowed(); a stale cmd_vel_raw (watchdog)
or a lost line also stops the car. The hardware E-STOP stays independent of this software.
"""
import socket
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

from picar_lab.car_agent.agent import CarAgent, MarkerDetector
from picar_lab.car_agent.protocol import decode, encode


class IntersectionGate(Node):
    def __init__(self):
        super().__init__("intersection_gate")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.agent = CarAgent(p("car", "W1"), p("route", "W_in,E_out,N_in,W_out").split(","), int(p("start_index", 0)))
        self.addr = (p("central_host", "192.168.50.10"), int(p("central_port", 5007)))
        self.speed_mps = float(p("nominal_speed_mps", 0.25))     # measured value for marker length estimation
        self.watchdog_s = float(p("cmd_watchdog_s", 0.3))
        self.markers = MarkerDetector(dt=0.1, approach_max_m=float(p("approach_mark_max_m", 0.03)),
                                      stop_min_m=float(p("stop_line_min_m", 0.04)))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.raw_cmd, self.raw_cmd_t = Twist(), 0.0
        self.at_stopline = False
        self.lost_line_frames = 0
        self.pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.create_subscription(Twist, "cmd_vel_raw", self.on_cmd, 10)
        self.create_subscription(Int32MultiArray, "line_values", self.on_line, 10)
        self.create_timer(0.05, self.on_timer)
        self.agent.register(time.monotonic())
        self.get_logger().info(f"intersection gate {self.agent.car} boot={self.agent.boot}")

    def on_cmd(self, msg):
        self.raw_cmd, self.raw_cmd_t = msg, time.monotonic()

    def on_line(self, msg):
        now = time.monotonic()
        values = list(msg.data)
        self.lost_line_frames = self.lost_line_frames + 1 if not any(values) else 0
        self.agent.set_health(line_ok=self.lost_line_frames < 5, obstacle=False)
        moving = self.raw_cmd.linear.x > 0 and now - self.raw_cmd_t < self.watchdog_s
        kind = self.markers.update(values, self.speed_mps if moving else 0.0)
        if kind == "approach":
            self.agent.on_approach_marker(now)
        elif kind == "stopline":
            if self.agent.zone == "IN":
                pass
            elif self.agent.may_cross(now):
                self.agent.on_crossed_stopline(now)         # roll through with a valid permit
            else:
                self.at_stopline = True                     # hold here until a permit arrives
        if self.agent.zone == "IN" and kind == "approach":
            self.agent.on_left_box(now)                     # optional exit tape after the box

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
        if self.at_stopline:
            if self.agent.zone == "APPROACH":
                self.agent.on_stopped_at_stopline(now)
            if self.agent.may_cross(now):
                self.agent.on_crossed_stopline(now)
                self.at_stopline = False
        for m in self.agent.tick(now):
            self.sock.sendto(encode(m), self.addr)
        fresh = now - self.raw_cmd_t < self.watchdog_s
        out = Twist()
        if fresh and self.agent.motion_allowed(now, self.at_stopline):
            out = self.raw_cmd
        self.pub.publish(out)                               # zero Twist = stop


def main(args=None):
    rclpy.init(args=args)
    node = IntersectionGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        rclpy.shutdown()


if __name__ == "__main__":
    main()
