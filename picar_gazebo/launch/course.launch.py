"""Gazebo Fortress + ROS 2 Humble: PiCar course, N cars, central server, monitor.
Cars drive like the real PiCar: front camera lane following + IR stop line + ultrasonic (ros2_camera_drive_node).

ros2 launch picar_gazebo calibrate.launch.py      # once: writes ~/picar_gz_runs/car_calibration.yaml
ros2 launch picar_gazebo course.launch.py cars:=3
ros2 launch picar_gazebo course.launch.py cars:=3 policy:=$PICAR_REPO/runs/picar_v9_photo_rt/seed42/best_policy.npz mode:=active
ros2 launch picar_gazebo course.launch.py cars:=3 gui:=false beam_fault_start:=60 beam_fault_stop:=80
"""
import os
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    cfg = {k: LaunchConfiguration(k).perform(context) for k in
           ("cars", "seed", "policy", "mode", "gui", "out_dir", "beam_fault_start", "beam_fault_stop", "speed_mps",
            "motor_latency_s", "gain_right", "ir_flip_prob", "port", "calibration")}
    repo = Path(os.environ.get("PICAR_REPO", Path.home() / "t-junction-rl-picar")).expanduser()
    if not (repo / "picar_sim").exists():
        raise RuntimeError(f"PICAR_REPO={repo} does not contain picar_sim; export PICAR_REPO=<repo path>")
    os.environ["PICAR_REPO"] = str(repo)
    import sys
    sys.path.insert(0, str(repo))
    from picar_gazebo.world_gen import generate
    import yaml
    out = Path(cfg["out_dir"] or Path.home() / "picar_gz_runs" / time.strftime("%Y%m%d_%H%M%S")).expanduser()
    generate(out, int(cfg["cars"]), int(cfg["seed"]))
    meta = yaml.safe_load((out / "course_meta.yaml").read_text())
    env = {**os.environ, "PYTHONPATH": f"{repo}{os.pathsep}{os.environ.get('PYTHONPATH', '')}", "PICAR_REPO": str(repo)}
    port = cfg["port"]
    calib = cfg["calibration"] or str(Path.home() / "picar_gz_runs" / "car_calibration.yaml")
    if not Path(calib).expanduser().exists():
        print(f"[picar_gazebo] WARNING {calib} missing: run calibrate.launch.py first (turns will miss the line)")
        calib = ""
    actions = []
    gz_cmd = ["ign", "gazebo", "-r", str(out / "picar_course.sdf")]
    if cfg["gui"].lower() in ("false", "0", "no"):
        gz_cmd[2:2] = ["-s", "--headless-rendering"]
    actions.append(ExecuteProcess(cmd=gz_cmd, output="screen", additional_env=env))
    bridge = ["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"]
    for c in meta["cars"]:
        n = c["name"]
        bridge += [f"/model/{n}/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
                   f"/model/{n}/line_cam@sensor_msgs/msg/Image[ignition.msgs.Image",
                   f"/model/{n}/front_cam@sensor_msgs/msg/Image[ignition.msgs.Image",
                   f"/model/{n}/pose@geometry_msgs/msg/PoseStamped[ignition.msgs.Pose"]
    actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", arguments=bridge, output="screen"))
    central = ["python3", "-m", "picar_sim.central_server", "--host", "127.0.0.1", "--port", port,
               "--log", str(out / "central.jsonl")]
    if cfg["policy"]:
        central += ["--policy", cfg["policy"], "--mode", cfg["mode"]]
    actions.append(ExecuteProcess(cmd=central, output="screen", additional_env=env))
    actions.append(Node(package="picar_gazebo", executable="course_monitor", output="screen", additional_env=env,
                        parameters=[{"meta_file": str(out / "course_meta.yaml"), "out_dir": str(out),
                                     "central_port": int(port), "lane_tolerance_m": float(meta["lane_tolerance_m"]),
                                     "beam_fault_start": float(cfg["beam_fault_start"]),
                                     "beam_fault_stop": float(cfg["beam_fault_stop"])}]))
    cars = []
    for c in meta["cars"]:
        n = c["name"]
        cars.append(Node(package="picar_gazebo", executable="ir_camera", namespace=n, additional_env=env,
                         parameters=[{"sensor_columns": meta["sensor_columns"], "flip_prob": float(cfg["ir_flip_prob"])}],
                         remappings=[("image", f"/model/{n}/line_cam")]))
        cars.append(Node(package="picar_gazebo", executable="motor_model", namespace=n, additional_env=env,
                         parameters=[{"latency_s": float(cfg["motor_latency_s"]), "track_m": meta["car"]["wheel_sep"],
                                      "gain_right": float(cfg["gain_right"])}],
                         remappings=[("gz_cmd_vel", f"/model/{n}/cmd_vel")]))
        fc = meta["car"]["front_cam"]
        cars.append(ExecuteProcess(output="screen", additional_env=env, cmd=[
            "python3", "-m", "picar_lab.car_agent.ros2_camera_drive_node", "--ros-args", "-r", f"__ns:=/{n}",
            "-r", f"__node:={n}_drive", "-r", f"image:=/model/{n}/front_cam",
            "-p", f"car:={n}", "-p", f"route:={','.join(c['route'])}",
            "-p", "start_index:=0", "-p", f"maneuvers_file:={out / 'maneuvers.yaml'}",
            "-p", f"approach_in_m:={float(c.get('approach_in_m', -1.0))}",
            "-p", "central_host:=127.0.0.1", "-p", f"central_port:={port}", "-p", f"speed_mps:={cfg['speed_mps']}",
            "-p", f"camera_x:={float(fc['x'])}", "-p", f"camera_z:={float(fc['z'])}",
            "-p", f"camera_pitch_deg:={float(fc['pitch_deg'])}", "-p", f"camera_hfov_deg:={float(fc['hfov_deg'])}"]
            + (["-p", f"calibration_file:={calib}"] if calib else [])))
    actions.append(TimerAction(period=6.0, actions=cars))           # let Gazebo and the bridge come up first
    print(f"[picar_gazebo] run directory: {out}")
    return actions


def generate_launch_description():
    args = [DeclareLaunchArgument("cars", default_value="3"),
            DeclareLaunchArgument("seed", default_value="1"),
            DeclareLaunchArgument("policy", default_value="", description="DQN .npz; empty = oldest_first rule"),
            DeclareLaunchArgument("mode", default_value="shadow", description="shadow | active"),
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("out_dir", default_value=""),
            DeclareLaunchArgument("beam_fault_start", default_value="-1"),
            DeclareLaunchArgument("beam_fault_stop", default_value="-1"),
            DeclareLaunchArgument("speed_mps", default_value="0.25"),
            DeclareLaunchArgument("motor_latency_s", default_value="0.05"),
            DeclareLaunchArgument("gain_right", default_value="0.97", description="right motor gain (mismatch)"),
            DeclareLaunchArgument("calibration", default_value="", description="car_calibration.yaml"),
            DeclareLaunchArgument("ir_flip_prob", default_value="0.0"),
            DeclareLaunchArgument("port", default_value="5007")]
    return LaunchDescription(args + [SetEnvironmentVariable("RCUTILS_COLORIZED_OUTPUT", "1"), OpaqueFunction(function=_setup)])
