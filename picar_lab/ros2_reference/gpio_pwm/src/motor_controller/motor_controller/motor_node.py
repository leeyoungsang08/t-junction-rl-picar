import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO

class MotorController(Node):
    def __init__(self):
        super().__init__('motor_controller')

        # Yahboom Raspbot Motor GPIO (BOARD 기준)
        self.LEFT_IN1 = 29
        self.LEFT_IN2 = 31
        self.RIGHT_IN1 = 33
        self.RIGHT_IN2 = 35
        self.LEFT_PWM = 32
        self.RIGHT_PWM = 12

        GPIO.setmode(GPIO.BOARD)
        for pin in [self.LEFT_IN1, self.LEFT_IN2, self.RIGHT_IN1, self.RIGHT_IN2, self.LEFT_PWM, self.RIGHT_PWM]:
            GPIO.setup(pin, GPIO.OUT)

        # PWM 초기화
        self.left_pwm = GPIO.PWM(self.LEFT_PWM, 100)
        self.right_pwm = GPIO.PWM(self.RIGHT_PWM, 100)
        self.left_pwm.start(0)
        self.right_pwm.start(0)

        self.subscription = self.create_subscription(Twist, 'cmd_vel', self.cmd_callback, 10)
        self.get_logger().info("✅ Yahboom Raspbot Motor Controller Node started")

    def cmd_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        left_speed = right_speed = int(abs(linear * 80))

        if linear > 0:
            self.forward(left_speed, right_speed)
        elif linear < 0:
            self.backward(left_speed, right_speed)
        elif angular > 0:
            self.turn_left(60)
        elif angular < 0:
            self.turn_right(60)
        else:
            self.stop()

    def forward(self, ls, rs):
        GPIO.output(self.LEFT_IN1, True)
        GPIO.output(self.LEFT_IN2, False)
        GPIO.output(self.RIGHT_IN1, True)
        GPIO.output(self.RIGHT_IN2, False)
        self.left_pwm.ChangeDutyCycle(ls)
        self.right_pwm.ChangeDutyCycle(rs)

    def backward(self, ls, rs):
        GPIO.output(self.LEFT_IN1, False)
        GPIO.output(self.LEFT_IN2, True)
        GPIO.output(self.RIGHT_IN1, False)
        GPIO.output(self.RIGHT_IN2, True)
        self.left_pwm.ChangeDutyCycle(ls)
        self.right_pwm.ChangeDutyCycle(rs)

    def turn_left(self, speed):
        GPIO.output(self.LEFT_IN1, False)
        GPIO.output(self.LEFT_IN2, True)
        GPIO.output(self.RIGHT_IN1, True)
        GPIO.output(self.RIGHT_IN2, False)
        self.left_pwm.ChangeDutyCycle(speed)
        self.right_pwm.ChangeDutyCycle(speed)

    def turn_right(self, speed):
        GPIO.output(self.LEFT_IN1, True)
        GPIO.output(self.LEFT_IN2, False)
        GPIO.output(self.RIGHT_IN1, False)
        GPIO.output(self.RIGHT_IN2, True)
        self.left_pwm.ChangeDutyCycle(speed)
        self.right_pwm.ChangeDutyCycle(speed)

    def stop(self):
        self.left_pwm.ChangeDutyCycle(0)
        self.right_pwm.ChangeDutyCycle(0)
        for p in [self.LEFT_IN1, self.LEFT_IN2, self.RIGHT_IN1, self.RIGHT_IN2]:
            GPIO.output(p, False)

def main(args=None):
    rclpy.init(args=args)
    node = MotorController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.cleanup()
        rclpy.shutdown()
