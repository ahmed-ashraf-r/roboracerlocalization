import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='rf2o_laser_odometry',
            executable='rf2o_laser_odometry_node',
            name='rf2o_laser_odometry',
            output='screen',
            parameters=[{
                'laser_scan_topic': '/autodrive/roboracer_1/lidar',
                'odom_topic': '/rf2o/odom',
                'publish_tf': False,            # DO NOT broadcast TF (EKF will handles TF)
                'base_frame_id': 'roboracer_1',  # Robot base link
                'odom_frame_id': 'odom',        # Odometry frame
                'init_pose_from_topic': '',
                'freq': 50.0                    # Matches LiDAR scan rate (Hz)
            }],
        ),
    ])