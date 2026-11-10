"""Gazebo: measure the drive calibration of one simulated car (run once before course.launch.py).

ros2 launch picar_gazebo calibrate.launch.py                 # writes ~/picar_gz_runs/car_calibration.yaml
ros2 launch picar_gazebo calibrate.launch.py motor_latency_s:=0.08 gain_right:=0.92 gui:=false

The car drives straight / brakes / spins open loop through motor_model (same imperfections as course.launch.py)
and the pose from Gazebo replaces the tape measure. Takes about 25 s, then the node exits.
"""
import os
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    cfg = {k: LaunchConfiguration(k).perform(context) for k in
           ("gui", "out_file", "motor_latency_s", "gain_right", "seed")}
    repo = Path(os.environ.get("PICAR_REPO", Path.home() / "t-junction-rl-picar")).expanduser()
    if not (repo / "picar_sim").exists():
        raise RuntimeError(f"PICAR_REPO={repo} does not contain picar_sim; export PICAR_REPO=<repo path>")
    import sys
    sys.path.insert(0, str(repo))
    from picar_gazebo.world_gen import generate
    out = Path.home() / "picar_gz_runs" / ("calib_" + time.strftime("%Y%m%d_%H%M%S"))
    generate(out, 1, int(cfg["seed"]))
    env = {**os.environ, "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "PICAR_REPO": str(repo)}
    gz_cmd = ["ign", "gazebo", "-r", str(out / "picar_course.sdf")]
    if cfg["gui"].lower() in ("false", "0", "no"):
        gz_cmd[2:2] = ["-s", "--headless-rendering"]
    n = "car0"
    bridge = ["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
              f"/model/{n}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
              f"/model/{n}/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose"]
    out_file = cfg["out_file"] or str(Path.home() / "picar_gz_runs" / "car_calibration.yaml")
    later = [Node(package="picar_gazebo", executable="motor_model", namespace=n, additional_env=env,
                  parameters=[{"latency_s": float(cfg["motor_latency_s"]), "gain_right": float(cfg["gain_right"])}],
                  remappings=[("gz_cmd_vel", f"/model/{n}/cmd_vel")]),
             Node(package="picar_gazebo", executable="calibrate", output="screen", additional_env=env,
                  parameters=[{"car": n, "out_file": out_file}])]
    print(f"[picar_gazebo] calibration -> {out_file}")
    return [ExecuteProcess(cmd=gz_cmd, output="screen", additional_env=env),
            Node(package="ros_gz_bridge", executable="parameter_bridge", arguments=bridge, output="screen"),
            TimerAction(period=6.0, actions=later)]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("out_file", default_value=""),
        DeclareLaunchArgument("motor_latency_s", default_value="0.05"),
        DeclareLaunchArgument("gain_right", default_value="0.97"),
        DeclareLaunchArgument("seed", default_value="1"),
        OpaqueFunction(function=_setup)])
