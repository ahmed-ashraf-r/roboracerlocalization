#!/usr/bin/env python3

import math

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32
from tf_transformations import quaternion_from_euler


def clamp(value, low, high):
    return max(low, min(value, high))


def wrap_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


class AckermannOdom(Node):
    """
    High-speed RoboRacer odometry.

    Core idea
    ---------
    Wheel encoders are NOT trusted during strong positive acceleration because
    the latest rosbag shows substantial driven-wheel spin.

    Instead:

      strong positive acceleration:
          integrate IMU longitudinal acceleration
          wheel speed correction -> near zero

      coast / steady speed / braking:
          wheel speed becomes reliable again
          strongly correct the integrated IMU speed toward wheel speed

    Pose is then integrated from this HYBRID vehicle speed.

    This avoids both previous failure modes:

      old node:
          heavy wheel-speed LPF -> large deceleration lag

      direct encoder-distance node:
          wheel spin -> odometry runs metres ahead during acceleration
    """

    def __init__(self):
        super().__init__("ackermann_odom")

        # ==================================================================
        # VEHICLE
        # ==================================================================

        self.declare_parameter("wheel_base", 0.3240)
        self.declare_parameter("wheel_radius", 0.0590)
        self.declare_parameter("max_steering", 0.5236)

        self.declare_parameter("com_x", 0.15532)

        self.declare_parameter("init_x", -0.1582)
        self.declare_parameter("init_y", 0.0)
        self.declare_parameter("init_theta", 0.0)

        self.l = float(
            self.get_parameter("wheel_base").value
        )

        self.r = float(
            self.get_parameter("wheel_radius").value
        )

        self.max_steer = float(
            self.get_parameter("max_steering").value
        )

        self.com_x = float(
            self.get_parameter("com_x").value
        )

        # ==================================================================
        # WHEEL SPEED FILTER
        # ==================================================================
        #
        # Wheel velocity itself is still filtered lightly.
        #
        # The hybrid estimator decides how much to TRUST it.
        # ==================================================================

        self.declare_parameter(
            "wheel_velocity_alpha",
            0.75
        )

        self.declare_parameter(
            "wheel_decel_alpha",
            0.95
        )

        self.wheel_alpha = float(
            self.get_parameter(
                "wheel_velocity_alpha"
            ).value
        )

        self.wheel_decel_alpha = float(
            self.get_parameter(
                "wheel_decel_alpha"
            ).value
        )

        # ==================================================================
        # IMU / WHEEL HYBRID SPEED FUSION
        # ==================================================================
        #
        # From the uploaded high-speed rosbag:
        #
        #   positive acceleration:
        #       wheel speed can be much too high due to wheel spin
        #
        #   braking / coast:
        #       wheel speed tracks true chassis speed closely
        #
        # Therefore wheel trust is asymmetric.
        #
        # wheel trust:
        #   ax <= +0.30 m/s²  -> full trust
        #   ax >= +1.50 m/s²  -> zero trust
        #   between           -> linear blend
        #
        # WHEEL_CORRECTION_GAIN controls how strongly each 50-Hz update
        # pulls the IMU-predicted speed toward wheel speed.
        # ==================================================================

        self.declare_parameter(
            "wheel_trust_accel_low",
            0.30
        )

        self.declare_parameter(
            "wheel_trust_accel_high",
            1.50
        )

        self.declare_parameter(
            "wheel_correction_gain",
            0.80
        )

        # ------------------------------------------------------------------
        # V2: bounded wheel lower-bound during strong acceleration
        #
        # The wheel-only rosbag showed:
        #   - wheel speed is excellent during steady/coast/braking
        #   - during hard acceleration, wheel speed commonly leads true
        #     chassis speed by roughly 2 m/s or more
        #
        # V1 completely ignored wheels at high positive acceleration.  That
        # made lap 1 excellent, but lap 2 could accumulate ~0.8 m of
        # under-distance from pure IMU integration.
        #
        # V2 never trusts the spinning wheel directly.  Instead it forms a
        # conservative lower-bound:
        #
        #     wheel_floor = wheel_speed - accel_slip_margin
        #
        # and only nudges the hybrid speed upward if it falls below that
        # lower-bound.  The correction is deliberately very small and capped.
        # ------------------------------------------------------------------

        self.declare_parameter(
            "accel_floor_activation",
            1.50
        )

        self.declare_parameter(
            "accel_slip_margin",
            2.50
        )

        self.declare_parameter(
            "accel_floor_alpha",
            0.05
        )

        self.declare_parameter(
            "accel_floor_max_step",
            0.025
        )

        self.declare_parameter(
            "imu_accel_deadband",
            0.03
        )

        self.declare_parameter(
            "imu_accel_scale",
            1.00
        )

        self.accel_low = float(
            self.get_parameter(
                "wheel_trust_accel_low"
            ).value
        )

        self.accel_high = float(
            self.get_parameter(
                "wheel_trust_accel_high"
            ).value
        )

        self.wheel_correction_gain = float(
            self.get_parameter(
                "wheel_correction_gain"
            ).value
        )

        self.accel_floor_activation = float(
            self.get_parameter(
                "accel_floor_activation"
            ).value
        )

        self.accel_slip_margin = float(
            self.get_parameter(
                "accel_slip_margin"
            ).value
        )

        self.accel_floor_alpha = float(
            self.get_parameter(
                "accel_floor_alpha"
            ).value
        )

        self.accel_floor_max_step = float(
            self.get_parameter(
                "accel_floor_max_step"
            ).value
        )

        self.imu_accel_deadband = float(
            self.get_parameter(
                "imu_accel_deadband"
            ).value
        )

        self.imu_accel_scale = float(
            self.get_parameter(
                "imu_accel_scale"
            ).value
        )

        # ==================================================================
        # EMPIRICAL WHEEL SCALE
        # ==================================================================

        self.declare_parameter(
            "slip_base",
            0.9948
        )

        self.declare_parameter(
            "slip_speed_coeff",
            0.0033
        )

        self.declare_parameter(
            "slip_min",
            0.85
        )

        self.slip_base = float(
            self.get_parameter(
                "slip_base"
            ).value
        )

        self.slip_speed_coeff = float(
            self.get_parameter(
                "slip_speed_coeff"
            ).value
        )

        self.slip_min = float(
            self.get_parameter(
                "slip_min"
            ).value
        )

        # ==================================================================
        # SAFETY
        # ==================================================================

        self.declare_parameter(
            "max_valid_wheel_speed",
            15.0
        )

        self.declare_parameter(
            "encoder_stale_timeout",
            0.15
        )

        self.declare_parameter(
            "imu_stale_timeout",
            0.15
        )

        self.max_valid_wheel_speed = float(
            self.get_parameter(
                "max_valid_wheel_speed"
            ).value
        )

        self.encoder_stale_timeout = float(
            self.get_parameter(
                "encoder_stale_timeout"
            ).value
        )

        self.imu_stale_timeout = float(
            self.get_parameter(
                "imu_stale_timeout"
            ).value
        )

        # ==================================================================
        # ENCODERS
        # ==================================================================

        self.prev_left_angle = None
        self.prev_right_angle = None

        self.prev_left_time = None
        self.prev_right_time = None

        self.left_raw_velocity = 0.0
        self.right_raw_velocity = 0.0

        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0

        self.left_initialized = False
        self.right_initialized = False

        self.last_left_receive_time = None
        self.last_right_receive_time = None

        # ==================================================================
        # IMU
        # ==================================================================

        self.imu_ax = 0.0
        self.imu_wz = 0.0

        self.imu_received = False
        self.last_imu_receive_time = None

        # ==================================================================
        # STEERING
        # ==================================================================

        self.delta_angle = 0.0

        # ==================================================================
        # HYBRID VEHICLE SPEED
        # ==================================================================

        self.vehicle_speed = 0.0

        self.last_wheel_speed = 0.0
        self.last_wheel_trust = 1.0
        self.last_accel_used = 0.0

        # V2 diagnostics.
        self.last_accel_floor = 0.0
        self.last_accel_floor_correction = 0.0
        self.accel_floor_active = False

        # ==================================================================
        # ODOMETRY STATE
        # ==================================================================

        self.x = float(
            self.get_parameter(
                "init_x"
            ).value
        )

        self.y = float(
            self.get_parameter(
                "init_y"
            ).value
        )

        self.theta = float(
            self.get_parameter(
                "init_theta"
            ).value
        )

        self.odom_prev_time = (
            self.get_clock().now()
        )

        self.total_distance = 0.0

        # ==================================================================
        # ROS I/O
        # ==================================================================

        self.right_encoder_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/right_encoder",
            self.right_encoder_callback,
            20
        )

        self.left_encoder_sub = self.create_subscription(
            JointState,
            "/autodrive/roboracer_1/left_encoder",
            self.left_encoder_callback,
            20
        )

        self.steer_sub = self.create_subscription(
            Float32,
            "/autodrive/roboracer_1/steering",
            self.steer_callback,
            20
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/autodrive/roboracer_1/imu",
            self.imu_callback,
            20
        )

        self.odom_pub = self.create_publisher(
            Odometry,
            "/roboracer/odom",
            20
        )

        # 50 Hz output / integration.
        self.timer = self.create_timer(
            0.02,
            self.update_odom
        )

        self.status_timer = self.create_timer(
            1.0,
            self.status_callback
        )

        # ==================================================================
        # MESSAGE
        # ==================================================================

        self.odom_msg = Odometry()

        self.odom_msg.header.frame_id = "odom"
        self.odom_msg.child_frame_id = "roboracer_1"

        self.get_logger().info(
            "Ackermann Odom IMU-HYBRID V2 started: "
            f"wheel_trust ax=[{self.accel_low:.2f}, "
            f"{self.accel_high:.2f}], "
            f"wheel_gain={self.wheel_correction_gain:.2f}, "
            f"accel_floor_margin={self.accel_slip_margin:.2f} m/s, "
            f"floor_alpha={self.accel_floor_alpha:.2f}"
        )

    # ======================================================================
    # HELPERS
    # ======================================================================

    def slip_multiplier(
        self,
        wheel_speed
    ):
        return clamp(
            self.slip_base
            -
            self.slip_speed_coeff
            *
            abs(
                wheel_speed
            ),
            self.slip_min,
            1.0
        )

    def filter_wheel_velocity(
        self,
        raw_velocity,
        previous_velocity,
        initialized
    ):
        corrected = (
            raw_velocity
            *
            self.slip_multiplier(
                raw_velocity
            )
        )

        if not initialized:
            return corrected

        # Make wheel speed react quickly whenever magnitude falls.
        if (
            abs(
                corrected
            )
            <
            abs(
                previous_velocity
            )
        ):
            alpha = (
                self.wheel_decel_alpha
            )
        else:
            alpha = (
                self.wheel_alpha
            )

        return (
            alpha
            *
            corrected
            +
            (
                1.0 -
                alpha
            )
            *
            previous_velocity
        )

    def compute_wheel_trust(
        self,
        longitudinal_acceleration
    ):
        """
        Asymmetric trust.

        Braking / coast:
            trust wheel strongly.

        Strong positive acceleration:
            reject wheel spin and use IMU prediction.
        """

        ax = (
            longitudinal_acceleration
        )

        if ax <= self.accel_low:
            return 1.0

        if ax >= self.accel_high:
            return 0.0

        return (
            self.accel_high -
            ax
        ) / (
            self.accel_high -
            self.accel_low
        )

    # ======================================================================
    # RIGHT ENCODER
    # ======================================================================

    def right_encoder_callback(
        self,
        msg: JointState
    ):
        if not msg.position:
            return

        current_angle = float(
            msg.position[0]
        )

        current_time = Time.from_msg(
            msg.header.stamp
        )

        if (
            self.prev_right_angle
            is not None
            and
            self.prev_right_time
            is not None
        ):

            dt = (
                current_time -
                self.prev_right_time
            ).nanoseconds * 1e-9

            if dt > 1e-6:

                raw_velocity = (
                    self.r
                    *
                    (
                        current_angle -
                        self.prev_right_angle
                    )
                    /
                    dt
                )

                if (
                    abs(
                        raw_velocity
                    )
                    <=
                    self.max_valid_wheel_speed
                ):

                    self.right_raw_velocity = (
                        raw_velocity
                    )

                    self.right_wheel_velocity = (
                        self.filter_wheel_velocity(
                            raw_velocity,
                            self.right_wheel_velocity,
                            self.right_initialized
                        )
                    )

                    self.right_initialized = True

        self.prev_right_angle = (
            current_angle
        )

        self.prev_right_time = (
            current_time
        )

        self.last_right_receive_time = (
            self.get_clock().now()
        )

    # ======================================================================
    # LEFT ENCODER
    # ======================================================================

    def left_encoder_callback(
        self,
        msg: JointState
    ):
        if not msg.position:
            return

        current_angle = float(
            msg.position[0]
        )

        current_time = Time.from_msg(
            msg.header.stamp
        )

        if (
            self.prev_left_angle
            is not None
            and
            self.prev_left_time
            is not None
        ):

            dt = (
                current_time -
                self.prev_left_time
            ).nanoseconds * 1e-9

            if dt > 1e-6:

                raw_velocity = (
                    self.r
                    *
                    (
                        current_angle -
                        self.prev_left_angle
                    )
                    /
                    dt
                )

                if (
                    abs(
                        raw_velocity
                    )
                    <=
                    self.max_valid_wheel_speed
                ):

                    self.left_raw_velocity = (
                        raw_velocity
                    )

                    self.left_wheel_velocity = (
                        self.filter_wheel_velocity(
                            raw_velocity,
                            self.left_wheel_velocity,
                            self.left_initialized
                        )
                    )

                    self.left_initialized = True

        self.prev_left_angle = (
            current_angle
        )

        self.prev_left_time = (
            current_time
        )

        self.last_left_receive_time = (
            self.get_clock().now()
        )

    # ======================================================================
    # IMU
    # ======================================================================

    def imu_callback(
        self,
        msg: Imu
    ):
        ax = float(
            msg.linear_acceleration.x
        )

        if (
            abs(
                ax
            )
            <
            self.imu_accel_deadband
        ):
            ax = 0.0

        self.imu_ax = (
            self.imu_accel_scale
            *
            ax
        )

        self.imu_wz = float(
            msg.angular_velocity.z
        )

        self.imu_received = True

        self.last_imu_receive_time = (
            self.get_clock().now()
        )

    # ======================================================================
    # STEERING
    # ======================================================================

    def steer_callback(
        self,
        msg: Float32
    ):
        self.delta_angle = clamp(
            float(
                msg.data
            ),
            -self.max_steer,
            self.max_steer
        )

    # ======================================================================
    # UPDATE
    # ======================================================================

    def update_odom(
        self
    ):
        now = self.get_clock().now()

        dt = (
            now -
            self.odom_prev_time
        ).nanoseconds * 1e-9

        if (
            dt <= 0.0
            or
            dt > 0.10
        ):
            self.odom_prev_time = now
            return

        self.odom_prev_time = now

        # ------------------------------------------------------------------
        # Sensor freshness
        # ------------------------------------------------------------------

        left_fresh = False
        right_fresh = False
        imu_fresh = False

        if (
            self.last_left_receive_time
            is not None
        ):
            left_age = (
                now -
                self.last_left_receive_time
            ).nanoseconds * 1e-9

            left_fresh = (
                left_age
                <=
                self.encoder_stale_timeout
            )

        if (
            self.last_right_receive_time
            is not None
        ):
            right_age = (
                now -
                self.last_right_receive_time
            ).nanoseconds * 1e-9

            right_fresh = (
                right_age
                <=
                self.encoder_stale_timeout
            )

        if (
            self.last_imu_receive_time
            is not None
        ):
            imu_age = (
                now -
                self.last_imu_receive_time
            ).nanoseconds * 1e-9

            imu_fresh = (
                imu_age
                <=
                self.imu_stale_timeout
            )

        # ------------------------------------------------------------------
        # Wheel-speed measurement
        # ------------------------------------------------------------------

        if (
            left_fresh
            and
            right_fresh
            and
            self.left_initialized
            and
            self.right_initialized
        ):

            wheel_speed = 0.5 * (
                self.left_wheel_velocity
                +
                self.right_wheel_velocity
            )

        else:

            wheel_speed = (
                self.vehicle_speed
            )

        self.last_wheel_speed = (
            wheel_speed
        )

        # ------------------------------------------------------------------
        # IMU prediction
        # ------------------------------------------------------------------

        accel = (
            self.imu_ax
            if imu_fresh
            else 0.0
        )

        self.last_accel_used = (
            accel
        )

        predicted_speed = (
            self.vehicle_speed
            +
            accel
            *
            dt
        )

        # RoboRacer is forward-only here.
        predicted_speed = max(
            0.0,
            predicted_speed
        )

        # ------------------------------------------------------------------
        # Asymmetric wheel correction
        # ------------------------------------------------------------------

        if (
            left_fresh
            and
            right_fresh
        ):

            wheel_trust = (
                self.compute_wheel_trust(
                    accel
                )
            )

            correction_gain = (
                self.wheel_correction_gain
                *
                wheel_trust
            )

            fused_speed = (
                predicted_speed
                +
                correction_gain
                *
                (
                    wheel_speed
                    -
                    predicted_speed
                )
            )

        else:

            wheel_trust = 0.0
            fused_speed = predicted_speed

        self.last_wheel_trust = (
            wheel_trust
        )

        # ------------------------------------------------------------------
        # V2 strong-acceleration safety floor
        #
        # Do NOT blend toward raw spinning-wheel speed.
        #
        # Only when positive acceleration is high, create a conservative
        # lower-bound from wheel speed minus the measured high-acceleration
        # slip margin.  If IMU integration has drifted below that floor,
        # apply only a tiny bounded correction.
        #
        # Calibration basis from the recorded wheel-only run:
        #     hard-acceleration wheel overspeed median ~2 m/s
        #     large values >3 m/s were common
        #
        # Using 2.5 m/s therefore remains deliberately conservative.
        # ------------------------------------------------------------------

        self.accel_floor_active = False
        self.last_accel_floor_correction = 0.0
        self.last_accel_floor = 0.0

        if (
            left_fresh
            and
            right_fresh
            and
            accel >= self.accel_floor_activation
        ):

            wheel_floor = max(
                0.0,
                wheel_speed
                -
                self.accel_slip_margin
            )

            self.last_accel_floor = (
                wheel_floor
            )

            if fused_speed < wheel_floor:

                requested_correction = (
                    self.accel_floor_alpha
                    *
                    (
                        wheel_floor
                        -
                        fused_speed
                    )
                )

                applied_correction = min(
                    requested_correction,
                    self.accel_floor_max_step
                )

                fused_speed += (
                    applied_correction
                )

                self.last_accel_floor_correction = (
                    applied_correction
                )

                self.accel_floor_active = True

        self.vehicle_speed = max(
            0.0,
            fused_speed
        )

        # ------------------------------------------------------------------
        # Pose integration
        #
        # Keep measured steering bicycle kinematics here so this odometry
        # remains independent from IMU yaw.  robot_localization can continue
        # fusing its dedicated IMU yaw-rate source.
        # ------------------------------------------------------------------

        wz_model = (
            self.vehicle_speed
            /
            self.l
            *
            math.tan(
                self.delta_angle
            )
        )

        mid_theta = (
            self.theta
            +
            0.5
            *
            wz_model
            *
            dt
        )

        ds = (
            self.vehicle_speed
            *
            dt
        )

        self.x += (
            ds
            *
            math.cos(
                mid_theta
            )
        )

        self.y += (
            ds
            *
            math.sin(
                mid_theta
            )
        )

        self.theta = wrap_angle(
            self.theta
            +
            wz_model
            *
            dt
        )

        self.total_distance += abs(
            ds
        )

        # ------------------------------------------------------------------
        # Rear axle -> vehicle frame
        # ------------------------------------------------------------------

        x_com = (
            self.x
            +
            self.com_x
            *
            math.cos(
                self.theta
            )
        )

        y_com = (
            self.y
            +
            self.com_x
            *
            math.sin(
                self.theta
            )
        )

        q = quaternion_from_euler(
            0.0,
            0.0,
            self.theta
        )

        # ------------------------------------------------------------------
        # Message
        # ------------------------------------------------------------------

        self.odom_msg.header.stamp = (
            now.to_msg()
        )

        self.odom_msg.pose.pose.position.x = (
            x_com
        )

        self.odom_msg.pose.pose.position.y = (
            y_com
        )

        self.odom_msg.pose.pose.position.z = 0.0

        self.odom_msg.pose.pose.orientation.x = q[0]
        self.odom_msg.pose.pose.orientation.y = q[1]
        self.odom_msg.pose.pose.orientation.z = q[2]
        self.odom_msg.pose.pose.orientation.w = q[3]

        self.odom_msg.twist.twist.linear.x = (
            self.vehicle_speed
        )

        self.odom_msg.twist.twist.linear.y = 0.0
        self.odom_msg.twist.twist.linear.z = 0.0

        self.odom_msg.twist.twist.angular.x = 0.0
        self.odom_msg.twist.twist.angular.y = 0.0

        # Preserve the previous wheel-odom interface.
        self.odom_msg.twist.twist.angular.z = (
            wz_model
        )

        # ------------------------------------------------------------------
        # Covariance
        # ------------------------------------------------------------------

        pose_cov = [
            0.0
        ] * 36

        sigma_xy = min(
            0.40,
            0.02
            +
            0.005
            *
            self.total_distance
        )

        sigma_yaw = min(
            0.30,
            0.025
            +
            0.004
            *
            self.total_distance
        )

        pose_cov[0] = sigma_xy ** 2
        pose_cov[7] = sigma_xy ** 2

        pose_cov[14] = 1e6
        pose_cov[21] = 1e6
        pose_cov[28] = 1e6

        pose_cov[35] = sigma_yaw ** 2

        self.odom_msg.pose.covariance = (
            pose_cov
        )

        twist_cov = [
            0.0
        ] * 36

        # During hard acceleration the estimate is IMU-dominant.
        # Increase uncertainty slightly as wheel trust falls.
        sigma_vx = (
            0.04
            +
            0.05
            *
            (
                1.0 -
                wheel_trust
            )
            +
            0.01
            *
            abs(
                self.vehicle_speed
            )
        )

        twist_cov[0] = sigma_vx ** 2

        # Non-holonomic lateral constraint.
        twist_cov[7] = 0.05 ** 2

        twist_cov[14] = 1e6
        twist_cov[21] = 1e6
        twist_cov[28] = 1e6

        sigma_wz = (
            0.05
            +
            0.02
            *
            abs(
                wz_model
            )
        )

        twist_cov[35] = sigma_wz ** 2

        self.odom_msg.twist.covariance = (
            twist_cov
        )

        self.odom_pub.publish(
            self.odom_msg
        )

    # ======================================================================
    # STATUS
    # ======================================================================

    def status_callback(
        self
    ):
        self.get_logger().info(
            "Ackermann-HYBRID-V2: "
            f"v={self.vehicle_speed:.2f} m/s, "
            f"wheel={self.last_wheel_speed:.2f}, "
            f"ax={self.last_accel_used:.2f}, "
            f"wheel_trust={self.last_wheel_trust:.2f}, "
            f"floor={self.last_accel_floor:.2f}, "
            f"floor_dv={self.last_accel_floor_correction:.3f}, "
            f"floor_active={self.accel_floor_active}, "
            f"pose=({self.x:.2f}, "
            f"{self.y:.2f}, "
            f"{self.theta:.2f})"
        )


def main(
    args=None
):
    rclpy.init(
        args=args
    )

    node = AckermannOdom()

    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
