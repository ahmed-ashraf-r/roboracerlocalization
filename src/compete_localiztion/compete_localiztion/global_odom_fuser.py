#!/usr/bin/env python3

import math
from collections import deque
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf_transformations import quaternion_from_euler


# =============================================================================
# Helpers
# =============================================================================

def wrap_angle(angle: float) -> float:
    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def quaternion_to_yaw(q) -> float:
    """
    Convert geometry_msgs/Quaternion to planar yaw.
    """
    siny_cosp = 2.0 * (
        q.w * q.z +
        q.x * q.y
    )

    cosy_cosp = 1.0 - 2.0 * (
        q.y * q.y +
        q.z * q.z
    )

    return math.atan2(
        siny_cosp,
        cosy_cosp
    )


def stamp_to_sec(stamp) -> float:
    return (
        float(stamp.sec) +
        float(stamp.nanosec) * 1e-9
    )


def compose_pose(a, b):
    """
    SE(2) pose composition.

    a = T_A_B = [x, y, yaw]
    b = T_B_C = [x, y, yaw]

    returns:
        T_A_C
    """

    ax, ay, ath = a
    bx, by, bth = b

    c = math.cos(ath)
    s = math.sin(ath)

    return [
        ax + c * bx - s * by,
        ay + s * bx + c * by,
        wrap_angle(
            ath + bth
        ),
    ]


def inverse_pose(p):
    """
    SE(2) inverse.
    """

    x, y, yaw = p

    c = math.cos(yaw)
    s = math.sin(yaw)

    return [
        -c * x - s * y,
        s * x - c * y,
        wrap_angle(
            -yaw
        ),
    ]


@dataclass
class OdomSample:
    t: float
    x: float
    y: float
    yaw: float


# =============================================================================
# Global Odom Fuser V2
# =============================================================================

class GlobalOdomFuser(Node):
    """
    High-rate global localization propagator.

    Inputs
    ------
    /odometry/filtered
        Smooth LOCAL EKF pose in frame "odom", ~50 Hz.

    /pf/pose/odom
        GLOBAL PF vehicle pose in frame "map", ~20 Hz.

    Output
    ------
    /localization/odom
        PF-corrected GLOBAL vehicle pose propagated by EKF at ~50 Hz.

    -------------------------------------------------------------------------
    IMPORTANT V2 CHANGE
    -------------------------------------------------------------------------

    The previous implementation limited updates in map->odom transform space.

    That is dangerous because even a modest change in map->odom yaw can rotate
    a vehicle that is many metres away from the odom origin and therefore move
    the published vehicle position by a very large amount.

    V2 instead:

        1. Predicts the GLOBAL BASE pose at the PF timestamp.
        2. Compares that predicted vehicle pose with the PF vehicle pose.
        3. Limits the correction DIRECTLY in vehicle-pose space:
               dx, dy, dyaw
        4. Builds a corrected GLOBAL BASE pose.
        5. Derives a new map->odom transform from that already-bounded pose.

    Therefore, if the maximum allowed vehicle translation correction is
    0.25 m/update, a yaw transform update cannot unexpectedly move the
    controller pose by 0.7-1.0 m.

    There is NO feedback from /localization/odom into either EKF or PF.
    This node broadcasts NO TF.
    """

    def __init__(self):
        super().__init__(
            'global_odom_fuser'
        )

        # =====================================================================
        # Topics / frames
        # =====================================================================

        self.declare_parameter(
            'ekf_topic',
            '/odometry/filtered'
        )

        self.declare_parameter(
            'pf_topic',
            '/pf/pose/odom'
        )

        self.declare_parameter(
            'output_topic',
            '/localization/odom'
        )

        self.declare_parameter(
            'map_frame',
            'map'
        )

        self.declare_parameter(
            'odom_frame',
            'odom'
        )

        self.declare_parameter(
            'base_frame',
            'roboracer_1'
        )


        # =====================================================================
        # EKF history
        # =====================================================================

        # 300 samples at 50 Hz ~= 6 seconds.
        self.declare_parameter(
            'odom_buffer_size',
            300
        )


        # =====================================================================
        # GLOBAL correction tuning
        # =====================================================================

        # Use the full VALID PF innovation request.
        # The pose-space limits below still constrain how much of it can
        # actually be applied in one PF update.
        self.declare_parameter(
            'correction_alpha',
            1.0
        )

        # Maximum vehicle-position correction RATE.
        self.declare_parameter(
            'max_pose_correction_speed',
            5.0
        )

        # Absolute translation cap per PF update.
        #
        # This is the key protection that prevents large controller jumps even
        # if a PF period is unusually long.
        self.declare_parameter(
            'max_pose_correction_step',
            0.25
        )

        # Maximum vehicle-yaw correction RATE.
        self.declare_parameter(
            'max_pose_correction_yaw_rate',
            4.0
        )

        # Absolute yaw cap per PF update.
        # 0.20 rad ~= 11.46 degrees.
        self.declare_parameter(
            'max_pose_correction_yaw_step',
            0.20
        )


        # =====================================================================
        # PF sanity rejection
        # =====================================================================

        # A PF innovation larger than these values is considered inconsistent
        # with the propagated global pose and is rejected.
        #
        # These remain intentionally much larger than the normal bounded
        # correction step.
        self.declare_parameter(
            'reject_translation',
            0.75
        )

        self.declare_parameter(
            'reject_yaw',
            0.50
        )


        # =====================================================================
        # Diagnostics
        # =====================================================================

        self.declare_parameter(
            'pf_stale_warn_sec',
            0.30
        )


        # =====================================================================
        # Read parameters
        # =====================================================================

        self.ekf_topic = self.get_parameter(
            'ekf_topic'
        ).value

        self.pf_topic = self.get_parameter(
            'pf_topic'
        ).value

        self.output_topic = self.get_parameter(
            'output_topic'
        ).value

        self.map_frame = self.get_parameter(
            'map_frame'
        ).value

        self.odom_frame = self.get_parameter(
            'odom_frame'
        ).value

        self.base_frame = self.get_parameter(
            'base_frame'
        ).value

        self.odom_buffer_size = int(
            self.get_parameter(
                'odom_buffer_size'
            ).value
        )

        self.correction_alpha = float(
            self.get_parameter(
                'correction_alpha'
            ).value
        )

        self.max_pose_correction_speed = float(
            self.get_parameter(
                'max_pose_correction_speed'
            ).value
        )

        self.max_pose_correction_step = float(
            self.get_parameter(
                'max_pose_correction_step'
            ).value
        )

        self.max_pose_correction_yaw_rate = float(
            self.get_parameter(
                'max_pose_correction_yaw_rate'
            ).value
        )

        self.max_pose_correction_yaw_step = float(
            self.get_parameter(
                'max_pose_correction_yaw_step'
            ).value
        )

        self.reject_translation = float(
            self.get_parameter(
                'reject_translation'
            ).value
        )

        self.reject_yaw = float(
            self.get_parameter(
                'reject_yaw'
            ).value
        )

        self.pf_stale_warn_sec = float(
            self.get_parameter(
                'pf_stale_warn_sec'
            ).value
        )


        # =====================================================================
        # Internal state
        # =====================================================================

        self.odom_buffer = deque(
            maxlen=self.odom_buffer_size
        )

        # T_map_odom = [x, y, yaw]
        self.map_to_odom = None

        self.last_pf_stamp = None
        self.last_pf_receive_time = None

        self.latest_pf_covariance = (
            [0.0] * 36
        )

        self.accepted_pf_updates = 0
        self.rejected_pf_updates = 0
        self.limited_pf_updates = 0
        self.publish_count = 0

        # Last correction diagnostics.
        self.last_innovation_dist = 0.0
        self.last_innovation_yaw = 0.0

        self.last_applied_translation = 0.0
        self.last_applied_yaw = 0.0


        # =====================================================================
        # ROS I/O
        # =====================================================================

        self.localization_pub = self.create_publisher(
            Odometry,
            self.output_topic,
            10
        )

        self.ekf_sub = self.create_subscription(
            Odometry,
            self.ekf_topic,
            self.ekf_callback,
            50
        )

        self.pf_sub = self.create_subscription(
            Odometry,
            self.pf_topic,
            self.pf_callback,
            20
        )

        self.status_timer = self.create_timer(
            1.0,
            self.status_callback
        )

        self.get_logger().info(
            'Global Odom Fuser V2 started: '
            f'PF={self.pf_topic}, '
            f'EKF={self.ekf_topic}, '
            f'output={self.output_topic}, '
            f'alpha={self.correction_alpha:.2f}, '
            f'max_pose_step={self.max_pose_correction_step:.3f} m, '
            f'max_yaw_step={self.max_pose_correction_yaw_step:.3f} rad'
        )


    # =========================================================================
    # EKF callback / interpolation
    # =========================================================================

    def ekf_callback(
        self,
        msg: Odometry
    ):

        t = stamp_to_sec(
            msg.header.stamp
        )

        pose = msg.pose.pose

        sample = OdomSample(
            t=t,
            x=float(
                pose.position.x
            ),
            y=float(
                pose.position.y
            ),
            yaw=quaternion_to_yaw(
                pose.orientation
            ),
        )


        # ---------------------------------------------------------------------
        # Clock reset / duplicate handling
        # ---------------------------------------------------------------------

        if self.odom_buffer:

            previous = self.odom_buffer[-1]

            if (
                t <
                previous.t - 0.1
            ):

                self.get_logger().warn(
                    'EKF timestamp moved backwards. '
                    'Resetting EKF history and global correction.'
                )

                self.odom_buffer.clear()

                self.map_to_odom = None
                self.last_pf_stamp = None

            elif abs(
                t - previous.t
            ) < 1e-9:

                self.odom_buffer[-1] = sample

            else:

                self.odom_buffer.append(
                    sample
                )

        else:

            self.odom_buffer.append(
                sample
            )


        # ---------------------------------------------------------------------
        # High-rate propagation
        # ---------------------------------------------------------------------

        if (
            self.map_to_odom
            is not None
        ):

            self.publish_global_odom(
                msg,
                sample
            )


    def interpolate_ekf_pose(
        self,
        query_t: float
    ):
        """
        Return T_odom_base(query_t) = [x, y, yaw].

        No high-speed extrapolation is performed.
        """

        if len(
            self.odom_buffer
        ) < 2:

            return None


        first = self.odom_buffer[0]
        last = self.odom_buffer[-1]

        epsilon = 1e-6

        if (
            query_t <
            first.t - epsilon
            or
            query_t >
            last.t + epsilon
        ):

            return None


        if abs(
            query_t - first.t
        ) <= epsilon:

            return [
                first.x,
                first.y,
                first.yaw
            ]


        if abs(
            query_t - last.t
        ) <= epsilon:

            return [
                last.x,
                last.y,
                last.yaw
            ]


        # Search backward because PF timestamps are normally close to the
        # newest EKF samples.
        for i in range(
            len(self.odom_buffer) - 1,
            0,
            -1
        ):

            a = self.odom_buffer[
                i - 1
            ]

            b = self.odom_buffer[
                i
            ]

            if (
                a.t <=
                query_t <=
                b.t
            ):

                dt = (
                    b.t -
                    a.t
                )

                if dt <= 1e-9:

                    return [
                        a.x,
                        a.y,
                        a.yaw
                    ]

                alpha = (
                    query_t -
                    a.t
                ) / dt

                x = (
                    a.x +
                    alpha *
                    (
                        b.x -
                        a.x
                    )
                )

                y = (
                    a.y +
                    alpha *
                    (
                        b.y -
                        a.y
                    )
                )

                dyaw = wrap_angle(
                    b.yaw -
                    a.yaw
                )

                yaw = wrap_angle(
                    a.yaw +
                    alpha *
                    dyaw
                )

                return [
                    x,
                    y,
                    yaw
                ]


        return None


    # =========================================================================
    # PF correction
    # =========================================================================

    def pf_callback(
        self,
        msg: Odometry
    ):

        pf_t = stamp_to_sec(
            msg.header.stamp
        )


        # ---------------------------------------------------------------------
        # Synchronize EKF to PF timestamp
        # ---------------------------------------------------------------------

        ekf_at_pf = self.interpolate_ekf_pose(
            pf_t
        )

        if ekf_at_pf is None:

            if self.odom_buffer:

                self.get_logger().warn(
                    'Skipping PF correction: '
                    'PF timestamp outside EKF history. '
                    f'PF={pf_t:.6f}, '
                    f'EKF=['
                    f'{self.odom_buffer[0].t:.6f}, '
                    f'{self.odom_buffer[-1].t:.6f}]'
                )

            return


        # ---------------------------------------------------------------------
        # PF global vehicle pose
        # ---------------------------------------------------------------------

        pf_msg_pose = (
            msg.pose.pose
        )

        pf_pose = [
            float(
                pf_msg_pose.position.x
            ),

            float(
                pf_msg_pose.position.y
            ),

            quaternion_to_yaw(
                pf_msg_pose.orientation
            ),
        ]

        self.latest_pf_covariance = list(
            msg.pose.covariance
        )

        self.last_pf_receive_time = (
            self.get_clock().now()
        )


        # ---------------------------------------------------------------------
        # INITIALIZATION
        # ---------------------------------------------------------------------

        if self.map_to_odom is None:

            self.map_to_odom = compose_pose(
                pf_pose,
                inverse_pose(
                    ekf_at_pf
                )
            )

            self.last_pf_stamp = pf_t

            self.accepted_pf_updates += 1

            self.get_logger().info(
                'Initialized map->odom from synchronized PF/EKF: '
                f'x={self.map_to_odom[0]:.3f}, '
                f'y={self.map_to_odom[1]:.3f}, '
                f'yaw={self.map_to_odom[2]:.3f}'
            )

            return


        # ---------------------------------------------------------------------
        # Predict GLOBAL vehicle pose using current map->odom + synchronized EKF
        # ---------------------------------------------------------------------

        predicted_global_pose = compose_pose(
            self.map_to_odom,
            ekf_at_pf
        )


        # ---------------------------------------------------------------------
        # VEHICLE-POSE innovation
        # ---------------------------------------------------------------------

        innovation_x = (
            pf_pose[0] -
            predicted_global_pose[0]
        )

        innovation_y = (
            pf_pose[1] -
            predicted_global_pose[1]
        )

        innovation_dist = math.hypot(
            innovation_x,
            innovation_y
        )

        innovation_yaw = wrap_angle(
            pf_pose[2] -
            predicted_global_pose[2]
        )

        self.last_innovation_dist = (
            innovation_dist
        )

        self.last_innovation_yaw = (
            innovation_yaw
        )


        # ---------------------------------------------------------------------
        # Reject impossible / corrupted PF updates
        # ---------------------------------------------------------------------

        if (
            innovation_dist >
            self.reject_translation
            or
            abs(
                innovation_yaw
            ) >
            self.reject_yaw
        ):

            self.rejected_pf_updates += 1

            self.last_pf_stamp = pf_t

            self.get_logger().warn(
                'Rejected PF correction: '
                f'position innovation='
                f'{innovation_dist:.3f} m, '
                f'yaw innovation='
                f'{innovation_yaw:.3f} rad'
            )

            return


        # ---------------------------------------------------------------------
        # PF update period
        # ---------------------------------------------------------------------

        dt_pf = 0.05

        if (
            self.last_pf_stamp
            is not None
        ):

            measured_dt = (
                pf_t -
                self.last_pf_stamp
            )

            if (
                0.005 <
                measured_dt <
                0.20
            ):

                dt_pf = measured_dt


        # ---------------------------------------------------------------------
        # Request correction in VEHICLE GLOBAL POSE space
        # ---------------------------------------------------------------------

        requested_dx = (
            self.correction_alpha *
            innovation_x
        )

        requested_dy = (
            self.correction_alpha *
            innovation_y
        )

        requested_dyaw = (
            self.correction_alpha *
            innovation_yaw
        )


        # ---------------------------------------------------------------------
        # Bound vehicle TRANSLATION correction
        # ---------------------------------------------------------------------

        requested_translation = math.hypot(
            requested_dx,
            requested_dy
        )

        rate_translation_limit = (
            self.max_pose_correction_speed *
            dt_pf
        )

        translation_limit = min(
            self.max_pose_correction_step,
            rate_translation_limit
        )

        applied_dx = requested_dx
        applied_dy = requested_dy

        translation_was_limited = False

        if (
            requested_translation >
            translation_limit
            and
            requested_translation >
            1e-12
        ):

            scale = (
                translation_limit /
                requested_translation
            )

            applied_dx *= scale
            applied_dy *= scale

            translation_was_limited = True


        # ---------------------------------------------------------------------
        # Bound vehicle YAW correction
        # ---------------------------------------------------------------------

        rate_yaw_limit = (
            self.max_pose_correction_yaw_rate *
            dt_pf
        )

        yaw_limit = min(
            self.max_pose_correction_yaw_step,
            rate_yaw_limit
        )

        applied_dyaw = max(
            -yaw_limit,
            min(
                requested_dyaw,
                yaw_limit
            )
        )

        yaw_was_limited = (
            abs(
                applied_dyaw -
                requested_dyaw
            ) >
            1e-12
        )


        if (
            translation_was_limited
            or
            yaw_was_limited
        ):

            self.limited_pf_updates += 1


        self.last_applied_translation = math.hypot(
            applied_dx,
            applied_dy
        )

        self.last_applied_yaw = (
            applied_dyaw
        )


        # ---------------------------------------------------------------------
        # Build corrected GLOBAL VEHICLE POSE
        #
        # This is the critical V2 step:
        #     the vehicle pose itself is bounded BEFORE map->odom is recomputed.
        # ---------------------------------------------------------------------

        corrected_global_pose = [
            predicted_global_pose[0] +
            applied_dx,

            predicted_global_pose[1] +
            applied_dy,

            wrap_angle(
                predicted_global_pose[2] +
                applied_dyaw
            ),
        ]


        # ---------------------------------------------------------------------
        # Derive NEW map->odom from already-safe vehicle pose
        #
        #     T_map_odom =
        #         T_map_base_corrected
        #         * inverse(T_odom_base_at_pf_time)
        # ---------------------------------------------------------------------

        self.map_to_odom = compose_pose(
            corrected_global_pose,
            inverse_pose(
                ekf_at_pf
            )
        )


        self.last_pf_stamp = pf_t
        self.accepted_pf_updates += 1


    # =========================================================================
    # Publish high-rate global odometry
    # =========================================================================

    def publish_global_odom(
        self,
        ekf_msg: Odometry,
        ekf_sample: OdomSample
    ):

        odom_pose = [
            ekf_sample.x,
            ekf_sample.y,
            ekf_sample.yaw,
        ]

        global_pose = compose_pose(
            self.map_to_odom,
            odom_pose
        )


        out = Odometry()

        out.header.stamp = (
            ekf_msg.header.stamp
        )

        out.header.frame_id = (
            self.map_frame
        )

        out.child_frame_id = (
            self.base_frame
        )


        out.pose.pose.position.x = (
            global_pose[0]
        )

        out.pose.pose.position.y = (
            global_pose[1]
        )

        out.pose.pose.position.z = 0.0


        q = quaternion_from_euler(
            0.0,
            0.0,
            global_pose[2]
        )

        out.pose.pose.orientation.x = q[0]
        out.pose.pose.orientation.y = q[1]
        out.pose.pose.orientation.z = q[2]
        out.pose.pose.orientation.w = q[3]


        # Keep PF global covariance as the best available global covariance
        # estimate.  EKF twist + covariance remain the fast local velocity
        # estimate used by the controller.
        out.pose.covariance = list(
            self.latest_pf_covariance
        )

        out.twist = ekf_msg.twist


        self.localization_pub.publish(
            out
        )

        self.publish_count += 1


    # =========================================================================
    # Diagnostics
    # =========================================================================

    def status_callback(
        self
    ):

        if self.map_to_odom is None:

            self.get_logger().info(
                'Waiting for first synchronized PF correction...'
            )

            return


        pf_age = None

        if (
            self.last_pf_receive_time
            is not None
        ):

            pf_age = (
                self.get_clock().now() -
                self.last_pf_receive_time
            ).nanoseconds * 1e-9


        stale_text = ''

        if (
            pf_age is not None
            and
            pf_age >
            self.pf_stale_warn_sec
        ):

            stale_text = (
                ' PF_STALE'
            )


        self.get_logger().info(
            'Global fuser V2: '
            f'published={self.publish_count}, '
            f'PF accepted={self.accepted_pf_updates}, '
            f'rejected={self.rejected_pf_updates}, '
            f'limited={self.limited_pf_updates}, '
            f'innovation={self.last_innovation_dist:.3f} m/'
            f'{self.last_innovation_yaw:.3f} rad, '
            f'applied={self.last_applied_translation:.3f} m/'
            f'{self.last_applied_yaw:.3f} rad, '
            f'PF_age='
            f'{pf_age if pf_age is not None else -1.0:.3f}s'
            f'{stale_text}'
        )


# =============================================================================
# Main
# =============================================================================

def main(
    args=None
):

    rclpy.init(
        args=args
    )

    node = GlobalOdomFuser()

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


if __name__ == '__main__':
    main()
