#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from tf_transformations import quaternion_from_euler


def clamp(value, low, high):
    """Clamp a value between a lower and upper bound."""
    return max(low, min(value, high))


class AckermannOdom(Node):

    def __init__(self):
        super().__init__("ackermann_odom")

        # ============================================================
        # ROBOT & VEHICLE PARAMETERS
        # ============================================================
        self.declare_parameter("wheel_base", 0.3240)
        self.declare_parameter("wheel_radius", 0.0590)
        self.declare_parameter("max_steering", 0.5236)
        self.declare_parameter("com_x", 0.15532)        # Center of Mass offset
        self.declare_parameter("init_x", -0.1582)       # Starting X position
        self.declare_parameter("init_y", 0.0)           # Starting Y position
        self.declare_parameter("init_theta", 0.0)       # Starting heading

        self.l = self.get_parameter("wheel_base").value
        self.r = self.get_parameter("wheel_radius").value
        self.max_steer = self.get_parameter("max_steering").value
        self.com_x = self.get_parameter("com_x").value

        # ============================================================
        # VELOCITY SMOOTHING (Low-Pass Filter)
        # alpha = 1.0 (no smoothing), alpha = 0.1 (heavy smoothing)
        # ============================================================
        self.vel_alpha = 0.15  

        self.left_velocity = 0.0
        self.right_velocity = 0.0

        self.prev_left_angle = None
        self.prev_right_angle = None
        self.prev_left_time = None
        self.prev_right_time = None

        # ============================================================
        # STEERING
        # ============================================================
        self.delta_angle = 0.0

        # ============================================================
        # ODOMETRY STATE (Rear Axle Tracking)
        # ============================================================
        self.x = self.get_parameter("init_x").value
        self.y = self.get_parameter("init_y").value
        self.theta = self.get_parameter("init_theta").value
        self.odom_prev_time = self.get_clock().now()

        # ============================================================
        # SUBSCRIBERS
        # ============================================================
        self.right_encoder_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/right_encoder",
            self.right_encoder_callback,
            10
        )
        self.left_encoder_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/left_encoder",
            self.left_encoder_callback,
            10
        )
        self.steer_sub = self.create_subscription(
            Float32,
            "/autodrive/roboracer_1/steering",
            self.steer_callback,
            10
        )

        # ============================================================
        # ODOMETRY PUBLISHER
        # ============================================================
        self.odom_pub = self.create_publisher(
            Odometry,
            "/roboracer/odom",
            10
        )

        # ============================================================
        # ODOM TIMER (50 Hz)
        # ============================================================
        self.timer = self.create_timer(0.02, self.update_odom)

        # ============================================================
        # ODOM MESSAGE PRE-ALLOCATION
        # ============================================================
        self.odom_msg = Odometry()
        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "roboracer_1"

        self.get_logger().info("Ackermann Odom started with Low-Pass Filter")

    # ================================================================
    # RIGHT ENCODER CALLBACK
    # ================================================================
    def right_encoder_callback(self, msg: JointState):
        if len(msg.position) == 0:
            return

        current_angle = msg.position[0]
        current_time = Time.from_msg(msg.header.stamp)

        if self.prev_right_angle is not None:
            delta_phi = current_angle - self.prev_right_angle
            dt = (current_time - self.prev_right_time).nanoseconds * 1e-9
            
            if dt > 0.0:
                raw_velocity = self.r * (delta_phi / dt)
                # Apply Exponential Moving Average (Low-Pass Filter)
                self.right_velocity = (self.vel_alpha * raw_velocity) + ((1.0 - self.vel_alpha) * self.right_velocity)

        self.prev_right_angle = current_angle
        self.prev_right_time = current_time

    # ================================================================
    # LEFT ENCODER CALLBACK
    # ================================================================
    def left_encoder_callback(self, msg: JointState):
        if len(msg.position) == 0:
            return

        current_angle = msg.position[0]
        current_time = Time.from_msg(msg.header.stamp)

        if self.prev_left_angle is not None:
            delta_phi = current_angle - self.prev_left_angle
            dt = (current_time - self.prev_left_time).nanoseconds * 1e-9
            
            if dt > 0.0:
                raw_velocity = self.r * (delta_phi / dt)
                # Apply Exponential Moving Average (Low-Pass Filter)
                self.left_velocity = (self.vel_alpha * raw_velocity) + ((1.0 - self.vel_alpha) * self.left_velocity)

        self.prev_left_angle = current_angle
        self.prev_left_time = current_time

    # ================================================================
    # STEERING CALLBACK
    # ================================================================
    def steer_callback(self, msg: Float32):
        self.delta_angle = clamp(msg.data, -self.max_steer, self.max_steer)

    # ================================================================
    # UPDATE ODOMETRY (timer callback)
    # ================================================================
    def update_odom(self):
        # Raw wheel linear velocity (average of left and right filtered speeds)
        vx_raw = (self.left_velocity + self.right_velocity) / 2.0

        # Apply Dynamic Slip Scaling Factor (Fitted from your 10%-40% throttle data)
        # V_actual = V_calc * (0.9948 - (0.0033 * V_calc))
        slip_multiplier = 0.9948 - (0.0033 * abs(vx_raw))
        vx = vx_raw * clamp(slip_multiplier, 0.85, 1.0)  # clamped for safety

        # Vehicle angular velocity (bicycle model using corrected vx)
        wz = (vx / self.l) * math.tan(self.delta_angle)

        now = self.get_clock().now()
        dt = (now - self.odom_prev_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.odom_prev_time = now

        # Midpoint Integration for Rear Axle
        mid_theta = self.theta + (wz * dt / 2.0)
        self.x += vx * math.cos(mid_theta) * dt
        self.y += vx * math.sin(mid_theta) * dt
        self.theta += wz * dt
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))

        # Transform Rear Axle position to Center of Mass (CoM)
        x_com = self.x + self.com_x * math.cos(self.theta)
        y_com = self.y + self.com_x * math.sin(self.theta)

        q = quaternion_from_euler(0.0, 0.0, self.theta)

        # Fill Odometry Message
        self.odom_msg.header.stamp = now.to_msg()
        self.odom_msg.pose.pose.position.x = x_com
        self.odom_msg.pose.pose.position.y = y_com
        self.odom_msg.pose.pose.position.z = 0.0

        self.odom_msg.pose.pose.orientation.x = q[0]
        self.odom_msg.pose.pose.orientation.y = q[1]
        self.odom_msg.pose.pose.orientation.z = q[2]
        self.odom_msg.pose.pose.orientation.w = q[3]

        self.odom_msg.twist.twist.linear.x = vx
        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0
        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0
        self.odom_msg.twist.twist.angular.z = wz

        # Publish
        self.odom_pub.publish(self.odom_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()