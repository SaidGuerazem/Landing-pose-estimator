from setuptools import setup

package_name = 'landing_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    py_modules=[
        'landing_detection.aruco_landing_pose',
        'landing_detection.detect_bulleye',
        'landing_detection.Landing_spot_pose_estimator',
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Said',
    maintainer_email='said.guerazem@city.ac.uk',
    description='ROS 2 nodes for detecting and estimating landing zones for sapience comp 2.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'aruco_landing_pose = landing_detection.aruco_landing_pose:main',
            'detect_bulleye = landing_detection.detect_bulleye:main',
            'landing_spot_pose_estimator = landing_detection.Landing_spot_pose_estimator:main',
        ],
    },
)

