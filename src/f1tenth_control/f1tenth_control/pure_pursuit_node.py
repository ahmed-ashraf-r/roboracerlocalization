#!/usr/bin/env python3
"""
Simple Pure Pursuit Controller following the 4-step canonical logic:
1. Find closest waypoint from filtered odom.
2. Find target waypoint where distance >= fixed lookahead distance.
3. Compute curvature (kappa) using fixed lookahead distance.
4. Compute steering angle from curvature and wheelbase.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


# =============================================================================
# Helper Functions
# =============================================================================

def clip(value, low, high):
    return max(low, min(high, value))


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def global_to_vehicle_frame(car_x, car_y, car_yaw, point_x, point_y):
    dx = point_x - car_x
    dy = point_y - car_y

    c = math.cos(car_yaw)
    s = math.sin(car_yaw)

    # Local x (forward), local y (lateral to the left)
    x_local = c * dx + s * dy
    y_local = -s * dx + c * dy
    return x_local, y_local


# =============================================================================
# Pure Pursuit Node
# =============================================================================

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Real-time state from sensors
        self.car_x = 0.81
        self.car_y = 3.16
        self.car_yaw = 4.71
        self.have_odom = False

        # Vehicle and control parameters
        self.wheelbase = 0.324             # meters (L)
        self.max_steer_rad = 0.5236        # max steering angle (~30 deg)
        self.lookahead_distance = 0.95     # fixed lookahead distance (Ld)
        self.constant_throttle = 0.155       # fixed forward throttle

        # Load raceline CSV (Columns: 0=X, 1=Y, 2=Speed)
        csv_path = '/home/autodrive_devkit/src/f1tenth_control/practice_iros_2026.csv'
        data = np.genfromtxt(csv_path, delimiter=',', comments='#')
        data = data[~np.isnan(data).any(axis=1)]

        self.path_x = data[:, 0]
        self.path_y = data[:, 1]
        self.n_points = len(self.path_x)

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/pf/pose/odom',
            self.odom_callback,
            qos_profile_sensor_data
        )

        # Publishers
        self.steer_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/steering_command',
            10
        )
        self.throttle_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/throttle_command',
            10
        )

        # 40 Hz control loop
        self.timer = self.create_timer(1.0 / 40.0, self.control_loop)
        self.get_logger().info('Pure Pursuit controller started cleanly.')

    def odom_callback(self, msg: Odometry):
        # Extract position
        self.car_x = msg.pose.pose.position.x
        self.car_y = msg.pose.pose.position.y
        
        # Extract orientation and convert quaternion to yaw
        self.car_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        
        self.have_odom = True

    def control_loop(self):
        if not self.have_odom:
            return

        # Step 1: closest waypoint
        distances = (self.path_x - self.car_x)**2 + (self.path_y - self.car_y)**2
        closest_idx = int(np.argmin(distances))

        # Step 2: first point at least lookahead_distance away and ahead of car
        target_idx = closest_idx
        for _ in range(self.n_points):
            dx = self.path_x[target_idx] - self.car_x
            dy = self.path_y[target_idx] - self.car_y
            dist = math.hypot(dx, dy)

            if dist >= self.lookahead_distance:
                x_loc, y_loc = global_to_vehicle_frame(
                    self.car_x, self.car_y, self.car_yaw,
                    self.path_x[target_idx], self.path_y[target_idx]
                )
                if x_loc > 0.0:
                    break

            target_idx = (target_idx + 1) % self.n_points

        target_x = self.path_x[target_idx]
        target_y = self.path_y[target_idx]

        # Step 3: curvature
        x_local, y_local = global_to_vehicle_frame(
            self.car_x, self.car_y, self.car_yaw, target_x, target_y
        )
        kappa = 2.0 * y_local / (self.lookahead_distance ** 2)

        # Step 4: steering
        steering = math.atan(self.wheelbase * kappa)
        steering = clip(steering, -self.max_steer_rad, self.max_steer_rad)

        self.steer_pub.publish(Float32(data=float(steering)))
        self.throttle_pub.publish(Float32(data=float(self.constant_throttle)))

    def stop_robot(self):
        self.steer_pub.publish(Float32(data=0.0))
        self.throttle_pub.publish(Float32(data=0.0))


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()