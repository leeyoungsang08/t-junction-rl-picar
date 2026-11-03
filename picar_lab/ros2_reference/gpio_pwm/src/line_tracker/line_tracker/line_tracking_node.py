import rclpy
from rclpy.node import Node
import RPi.GPIO as GPIO
from std_msgs.msg import Int32MultiArray

class LineTracker(Node):
    def __init__(self):
        super().__init__('line_tracker')

        # Yahboom Raspbot 라인트래킹 센서 핀
        self.pins = [11, 13, 15, 16]  # BOARD 기준 (좌→우)
        GPIO.setmode(GPIO.BOARD)
        for p in self.pins:
            GPIO.setup(p, GPIO.IN)

        self.publisher_ = self.create_publisher(Int32MultiArray, 'line_values', 10)
        self.timer = self.create_timer(0.1, self.read_sensors)
        self.get_logger().info("✅ Yahboom Raspbot Line Tracker Node started")

    def read_sensors(self):
        values = [GPIO.input(p) for p in self.pins]
        msg = Int32MultiArray()
        msg.data = values
        self.publisher_.publish(msg)
        self.get_logger().info(f"Sensor Values: {values}")

def main(args=None):
    rclpy.init(args=args)
    node = LineTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        rclpy.shutdown()
