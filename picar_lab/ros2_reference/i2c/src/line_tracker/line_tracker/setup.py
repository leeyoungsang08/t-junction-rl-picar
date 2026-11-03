from setuptools import setup

package_name = 'line_tracker'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Line tracker sensor node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'line_tracking_node = line_tracker.line_tracking_node:main',
        ],
    },
)
