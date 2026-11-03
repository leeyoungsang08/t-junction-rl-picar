import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist

class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')
        self.subscription = self.create_subscription(Int32MultiArray, 'line_values', self.callback, 10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("✅ Yahboom Raspbot Line Follower Node started")

    def callback(self, msg):
        left1, left2, right1, right2 = msg.data
        cmd = Twist()

        if left1 == 1 and right1 == 1:
            cmd.linear.x = 1.0
        elif left1 == 1:
            cmd.angular.z = 1.0
        elif right1 == 1:
            cmd.angular.z = -1.0
        else:
            cmd.linear.x = 0.0

        self.publisher_.publish(cmd)
