from setuptools import setup

package_name = 'line_follower'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='you@example.com',
    description='Line follower logic node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'line_follower_node = line_follower.line_follower_node:main',
        ],
    },
)
