from setuptools import setup

package_name = 'motor_controller'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Motor controller for PiCar',
    license='MIT',
    entry_points={
        'console_scripts': [
            'motor_node = motor_controller.motor_node:main',
        ],
    },
)
