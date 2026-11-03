from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='line_tracker', executable='line_tracking_node', output='screen'),
        Node(package='line_follower', executable='line_follower_node', output='screen'),
        Node(package='motor_controller', executable='motor_node', output='screen'),
    ])
