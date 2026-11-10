from glob import glob

from setuptools import setup

package_name = "picar_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="t-junction-rl-picar",
    maintainer_email="noreply@example.com",
    description="Gazebo Fortress world of the PiCar course",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ir_camera = picar_gazebo.ir_camera_node:main",
            "motor_model = picar_gazebo.motor_model_node:main",
            "course_monitor = picar_gazebo.course_monitor_node:main",
            "calibrate = picar_gazebo.calibrate_node:main",
            "world_gen = picar_gazebo.world_gen:main_cli",
        ],
    },
)
