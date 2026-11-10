"""Real PiCar motor node that can steer while driving: cmd_vel (m/s, rad/s) -> left/right PWM.

Same GPIO pins as picar_lab/ros2_reference/.../motor_node.py (Yahboom Raspbot, BOARD numbering).
Calibrate mps_per_pwm, deadband_pwm and track_m on the car before use. NOT tested on hardware.
A cmd_vel older than watchdog_s stops the motors.
"""
import time

import rclpy
import RPi.GPIO as GPIO
from geometry_msgs.msg import Twist
from rclpy.node import Node

from picar_lab.car_agent.drive import wheel_pwm

PINS = {"L_IN1": 29, "L_IN2": 31, "R_IN1": 33, "R_IN2": 35, "L_PWM": 32, "R_PWM": 12}


class MotorDiff(Node):
    def __init__(self):
        super().__init__("motor_diff")
        p = lambda name, default: self.declare_parameter(name, default).value
        self.track, self.k, self.dead = float(p("track_m", 0.15)), float(p("mps_per_pwm", 0.004)), int(p("deadband_pwm", 25))
        self.watchdog = float(p("watchdog_s", 0.3))
        GPIO.setmode(GPIO.BOARD)
        for pin in PINS.values():
            GPIO.setup(pin, GPIO.OUT)
        self.pl, self.pr = GPIO.PWM(PINS["L_PWM"], 100), GPIO.PWM(PINS["R_PWM"], 100)
        self.pl.start(0)
        self.pr.start(0)
        self.last = 0.0
        self.create_subscription(Twist, "cmd_vel", self.on_cmd, 10)
        self.create_timer(0.05, self.on_timer)

    def _side(self, in1, in2, pwm, duty):
        GPIO.output(PINS[in1], duty > 0)
        GPIO.output(PINS[in2], duty < 0)
        pwm.ChangeDutyCycle(abs(duty))

    def on_cmd(self, msg):
        left, right = wheel_pwm(msg.linear.x, msg.angular.z, self.track, self.k, self.dead)
        self._side("L_IN1", "L_IN2", self.pl, left)
        self._side("R_IN1", "R_IN2", self.pr, right)
        self.last = time.monotonic()

    def on_timer(self):
        if time.monotonic() - self.last > self.watchdog:
            self._side("L_IN1", "L_IN2", self.pl, 0)
            self._side("R_IN1", "R_IN2", self.pr, 0)

    def stop(self):
        self.pl.stop()
        self.pr.stop()
        GPIO.cleanup()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDiff()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
