import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from motor_controller.YB_Pcb_Car import YB_Pcb_Car  # ✅ Yahboom 드라이버 import

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        # ✅ Yahboom Raspbot용 모터 제어 클래스 초기화
        self.car = YB_Pcb_Car()
        self.subscription = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.get_logger().info("✅ Yahboom Raspbot Motor Controller Node started (I2C mode)")

    def cmd_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        if linear > 0:          # 직진
            self.car.Car_Run(70, 70)
        elif linear < 0:        # 후진
            self.car.Car_Back(70, 70)
        elif angular > 0:       # 좌회전
            self.car.Car_Left(50, 70)
        elif angular < 0:       # 우회전
            self.car.Car_Right(70, 50)
        else:                   # 정지
            self.car.Car_Stop()

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.car.Car_Stop()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
