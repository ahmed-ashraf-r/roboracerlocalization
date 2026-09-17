import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'f1tenth_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sama-ahmed',
    maintainer_email='sama-ahmed@todo.todo',
    description='F1TENTH Control Architecture Package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pps_icra_2026= f1tenth_control.pps_icra_2026:main'
        ],
    },
)