"""Emulates the 4-channel IR line sensor from the downward Gazebo camera.

Subscribes image (sensor_msgs/Image, rgb8) and publishes line_values (Int32MultiArray, 4 ch left->right),
the same topic and format as the real line_tracker node. Optional noise: per-channel flip probability.
"""
import random

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray

from picar_gazebo.core import ir_from_gray


class IrCamera(Node):
    def __init__(self):
        super().__init__("ir_camera")
        p = lambda n, d: self.declare_parameter(n, d).value
        self.cols = [int(c) for c in p("sensor_columns", [7, 13, 18, 24])]
        self.threshold = float(p("threshold", 0.35))
        self.dark = int(p("dark_value", 1))
        self.flip_prob = float(p("flip_prob", 0.0))
        self.pub = self.create_publisher(Int32MultiArray, "line_values", 10)
        self.create_subscription(Image, "image", self.on_image, 10)

    def on_image(self, msg):
        ch = 3 if msg.encoding in ("rgb8", "bgr8") else (4 if msg.encoding in ("rgba8", "bgra8") else 1)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
        gray = img.reshape(msg.height, msg.width, ch)[..., :3].mean(axis=2) / 255.0 if ch > 1 else img / 255.0
        values = ir_from_gray(gray, self.cols, self.threshold, self.dark)
        if self.flip_prob > 0:
            values = [1 - v if random.random() < self.flip_prob else v for v in values]
        out = Int32MultiArray()
        out.data = values
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = IrCamera()
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
