#!/usr/bin/env python3

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf_transformations import quaternion_from_euler


# =============================================================================
# HELPERS
# =============================================================================

def wrap_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle)
    )


def quaternion_to_yaw(q):
    return math.atan2(
        2.0 * (
            q.w * q.z
            +
            q.x * q.y
        ),
        1.0
        -
        2.0 * (
            q.y * q.y
            +
            q.z * q.z
        )
    )


def stamp_to_sec(stamp):
    return (
        float(stamp.sec)
        +
        float(stamp.nanosec)
        *
        1e-9
    )


def compose_pose(a, b):
    """
    T_A_C = T_A_B * T_B_C
    """

    ax, ay, ath = a
    bx, by, bth = b

    c = math.cos(
        ath
    )

    s = math.sin(
        ath
    )

    return np.array([
        ax
        +
        c * bx
        -
        s * by,

        ay
        +
        s * bx
        +
        c * by,

        wrap_angle(
            ath
            +
            bth
        )
    ], dtype=float)


def inverse_pose(p):
    """
    SE(2) inverse.
    """

    x, y, yaw = p

    c = math.cos(
        yaw
    )

    s = math.sin(
        yaw
    )

    return np.array([
        -c * x
        -
        s * y,

        s * x
        -
        c * y,

        wrap_angle(
            -yaw
        )
    ], dtype=float)


def clamp(value, low, high):
    return max(
        low,
        min(
            value,
            high
        )
    )


@dataclass
class OdomSample:
    t: float
    x: float
    y: float
    yaw: float


# =============================================================================
# GLOBAL ODOM FUSER - RACING V3
# =============================================================================

class GlobalOdomFuser(Node):
    """
    Covariance-aware high-rate global localization fuser.

    Inputs:
      /odometry/filtered   local EKF pose, ~50 Hz
      /pf/pose/odom       global PF pose, ~20 Hz

    Output:
      /localization/odom  global pose propagated at EKF rate

    -------------------------------------------------------------------------
    WHY V3
    -------------------------------------------------------------------------

    At 100% target speed the PF becomes longitudinally uncertain in the long
    corridor, then recovers when distinctive corner geometry returns.

    The old fuser failed in two ways:

      1) It treated every PF X/Y direction equally even when PF covariance
         clearly said one direction was poorly observed.

      2) Once PF recovered by >0.75 m, the old hard innovation gate rejected
         the CORRECT recovery and left /localization/odom locked ~1.5-2 m
         ahead of the real car.

    V3 does this instead:

      * Read the full PF 2x2 X/Y covariance.
      * Eigendecompose it.
      * Apply PF innovation strongly only along low-uncertainty directions.
      * Suppress PF innovation along high-uncertainty corridor directions.
      * When covariance becomes small again, accept the recovered PF anchor.
      * Move /localization/odom smoothly toward that anchor at 50 Hz in
        VEHICLE-POSE space, not transform space.

    Therefore:
      - corridor ambiguity does not drag the global state unnecessarily,
      - PF recovery is never permanently blocked by the old 0.75 m gate,
      - controller does not receive a single giant pose jump,
      - no TF is published.
    """

    def __init__(self):
        super().__init__(
            'global_odom_fuser'
        )

        # =====================================================================
        # TOPICS / FRAMES
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

        self.declare_parameter(
            'odom_buffer_size',
            400
        )

        # ---------------------------------------------------------------------
        # Backward-compatible parameter declarations.
        #
        # Older launch/YAML files may still provide these V2 names.  V3 no
        # longer uses them for its correction logic, but declaring them keeps
        # existing launch files compatible.
        # ---------------------------------------------------------------------
        self.declare_parameter('correction_alpha', 1.0)
        self.declare_parameter('max_pose_correction_speed', 5.0)
        self.declare_parameter('max_pose_correction_step', 0.25)
        self.declare_parameter('max_pose_correction_yaw_rate', 4.0)
        self.declare_parameter('max_pose_correction_yaw_step', 0.20)
        self.declare_parameter('reject_translation', 0.75)
        self.declare_parameter('reject_yaw', 0.50)

        # =====================================================================
        # PF COVARIANCE CONFIDENCE
        # =====================================================================

        # One-sigma PF position uncertainty:
        #
        # <= good  -> fully trusted
        # >= bad   -> ignored along that covariance eigen-direction
        # between  -> linearly blended
        #
        # The 100% bag showed corridor longitudinal sigma roughly 0.3-0.6 m,
        # while recovered PF sigma fell below ~0.15-0.20 m.
        self.declare_parameter(
            'pf_sigma_pos_good',
            0.18
        )

        self.declare_parameter(
            'pf_sigma_pos_bad',
            0.45
        )

        self.declare_parameter(
            'pf_sigma_yaw_good',
            0.05
        )

        self.declare_parameter(
            'pf_sigma_yaw_bad',
            0.20
        )

        # =====================================================================
        # TARGET SAFETY GATES
        # =====================================================================

        # This is a LAST-RESORT corruption guard, not a normal PF innovation
        # gate.  The previous 0.75 m threshold was much too small for racing
        # reacquisition and caused permanent lockout.
        self.declare_parameter(
            'hard_reject_translation',
            3.0
        )

        self.declare_parameter(
            'hard_reject_yaw',
            0.80
        )

        # =====================================================================
        # SMOOTH 50-HZ REACQUISITION
        # =====================================================================

        # A confident PF recovery can correct global position quickly.
        #
        # 12 m/s correction rate at 50 Hz -> max ~0.24 m/tick before the
        # absolute step cap below.  A 1.8 m error can therefore disappear
        # smoothly in roughly 0.15-0.20 s.
        self.declare_parameter(
            'high_conf_correction_speed',
            12.0
        )

        # When PF is only partly confident, move much more gently.
        self.declare_parameter(
            'low_conf_correction_speed',
            1.0
        )

        self.declare_parameter(
            'max_correction_step',
            0.22
        )

        self.declare_parameter(
            'high_conf_yaw_rate',
            5.0
        )

        self.declare_parameter(
            'low_conf_yaw_rate',
            0.50
        )

        self.declare_parameter(
            'max_yaw_step',
            0.10
        )

        # =====================================================================
        # DIAGNOSTICS
        # =====================================================================

        self.declare_parameter(
            'pf_stale_warn_sec',
            0.30
        )

        # =====================================================================
        # READ PARAMETERS
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

        self.pf_sigma_pos_good = float(
            self.get_parameter(
                'pf_sigma_pos_good'
            ).value
        )

        self.pf_sigma_pos_bad = float(
            self.get_parameter(
                'pf_sigma_pos_bad'
            ).value
        )

        self.pf_sigma_yaw_good = float(
            self.get_parameter(
                'pf_sigma_yaw_good'
            ).value
        )

        self.pf_sigma_yaw_bad = float(
            self.get_parameter(
                'pf_sigma_yaw_bad'
            ).value
        )

        self.hard_reject_translation = float(
            self.get_parameter(
                'hard_reject_translation'
            ).value
        )

        self.hard_reject_yaw = float(
            self.get_parameter(
                'hard_reject_yaw'
            ).value
        )

        self.high_conf_correction_speed = float(
            self.get_parameter(
                'high_conf_correction_speed'
            ).value
        )

        self.low_conf_correction_speed = float(
            self.get_parameter(
                'low_conf_correction_speed'
            ).value
        )

        self.max_correction_step = float(
            self.get_parameter(
                'max_correction_step'
            ).value
        )

        self.high_conf_yaw_rate = float(
            self.get_parameter(
                'high_conf_yaw_rate'
            ).value
        )

        self.low_conf_yaw_rate = float(
            self.get_parameter(
                'low_conf_yaw_rate'
            ).value
        )

        self.max_yaw_step = float(
            self.get_parameter(
                'max_yaw_step'
            ).value
        )

        self.pf_stale_warn_sec = float(
            self.get_parameter(
                'pf_stale_warn_sec'
            ).value
        )

        # =====================================================================
        # STATE
        # =====================================================================

        self.odom_buffer = deque(
            maxlen=self.odom_buffer_size
        )

        # Current transform actually used to publish global odom.
        self.current_map_to_odom = None

        # PF-derived transform we are smoothly moving toward.
        self.target_map_to_odom = None

        self.target_position_confidence = 0.0
        self.target_yaw_confidence = 0.0

        self.latest_pf_covariance = (
            [0.0] * 36
        )

        self.last_pf_receive_time = None
        self.last_ekf_publish_time = None

        # Diagnostics.
        self.accepted_pf_updates = 0
        self.rejected_pf_updates = 0
        self.publish_count = 0

        self.last_raw_innovation = 0.0
        self.last_weighted_innovation = 0.0
        self.last_yaw_innovation = 0.0

        self.last_pf_sigma_min = 0.0
        self.last_pf_sigma_max = 0.0

        self.last_pf_weight_min = 0.0
        self.last_pf_weight_max = 0.0

        self.last_applied_step = 0.0

        # =====================================================================
        # ROS
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
            'Global Odom Fuser RACING V3 started | '
            f'PF sigma good/bad='
            f'{self.pf_sigma_pos_good:.2f}/'
            f'{self.pf_sigma_pos_bad:.2f} m | '
            f'correction='
            f'{self.low_conf_correction_speed:.1f}->'
            f'{self.high_conf_correction_speed:.1f} m/s | '
            f'hard_reject={self.hard_reject_translation:.1f} m'
        )

    # =========================================================================
    # CONFIDENCE
    # =========================================================================

    @staticmethod
    def confidence_from_sigma(
        sigma,
        good,
        bad
    ):

        if bad <= good:
            return (
                1.0
                if sigma <= good
                else 0.0
            )

        return float(
            np.clip(
                (
                    bad
                    -
                    sigma
                )
                /
                (
                    bad
                    -
                    good
                ),
                0.0,
                1.0
            )
        )

    # =========================================================================
    # EKF BUFFER
    # =========================================================================

    def interpolate_ekf_pose(
        self,
        query_t
    ):

        if len(
            self.odom_buffer
        ) < 2:
            return None

        first = self.odom_buffer[0]
        last = self.odom_buffer[-1]

        if (
            query_t
            <
            first.t
            -
            1e-6
            or
            query_t
            >
            last.t
            +
            1e-6
        ):
            return None

        # Search from newest sample backwards because PF timestamps normally
        # sit near the tail of the buffer.
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
                a.t
                <=
                query_t
                <=
                b.t
            ):

                dt = (
                    b.t
                    -
                    a.t
                )

                if dt <= 1e-9:

                    return np.array([
                        a.x,
                        a.y,
                        a.yaw
                    ])

                ratio = (
                    query_t
                    -
                    a.t
                ) / dt

                dyaw = wrap_angle(
                    b.yaw
                    -
                    a.yaw
                )

                return np.array([
                    a.x
                    +
                    ratio
                    *
                    (
                        b.x
                        -
                        a.x
                    ),

                    a.y
                    +
                    ratio
                    *
                    (
                        b.y
                        -
                        a.y
                    ),

                    wrap_angle(
                        a.yaw
                        +
                        ratio
                        *
                        dyaw
                    )
                ])

        return None

    # =========================================================================
    # PF CALLBACK
    # =========================================================================

    def pf_callback(
        self,
        msg
    ):

        pf_t = stamp_to_sec(
            msg.header.stamp
        )

        ekf_at_pf = self.interpolate_ekf_pose(
            pf_t
        )

        if ekf_at_pf is None:

            return

        pose_msg = msg.pose.pose

        pf_pose = np.array([
            float(
                pose_msg.position.x
            ),

            float(
                pose_msg.position.y
            ),

            quaternion_to_yaw(
                pose_msg.orientation
            )
        ])

        covariance = np.asarray(
            msg.pose.covariance,
            dtype=float
        )

        self.latest_pf_covariance = list(
            msg.pose.covariance
        )

        self.last_pf_receive_time = (
            self.get_clock().now()
        )

        # ---------------------------------------------------------------------
        # FIRST GLOBAL ANCHOR
        # ---------------------------------------------------------------------

        if self.current_map_to_odom is None:

            initial_transform = compose_pose(
                pf_pose,
                inverse_pose(
                    ekf_at_pf
                )
            )

            self.current_map_to_odom = (
                initial_transform.copy()
            )

            self.target_map_to_odom = (
                initial_transform.copy()
            )

            self.target_position_confidence = 1.0
            self.target_yaw_confidence = 1.0

            self.accepted_pf_updates += 1

            self.get_logger().info(
                'Initialized Racing V3 global anchor.'
            )

            return

        # ---------------------------------------------------------------------
        # PREDICTED GLOBAL POSE AT THE PF MEASUREMENT TIME
        # ---------------------------------------------------------------------

        predicted_pose = compose_pose(
            self.current_map_to_odom,
            ekf_at_pf
        )

        raw_position_innovation = (
            pf_pose[:2]
            -
            predicted_pose[:2]
        )

        raw_distance = float(
            np.linalg.norm(
                raw_position_innovation
            )
        )

        yaw_innovation = wrap_angle(
            pf_pose[2]
            -
            predicted_pose[2]
        )

        self.last_raw_innovation = (
            raw_distance
        )

        self.last_yaw_innovation = (
            yaw_innovation
        )

        # ---------------------------------------------------------------------
        # LAST-RESORT CORRUPTION REJECTION
        #
        # Do NOT use the old 0.75 m racing gate here.
        # ---------------------------------------------------------------------

        if (
            raw_distance
            >
            self.hard_reject_translation
            or
            abs(
                yaw_innovation
            )
            >
            self.hard_reject_yaw
        ):

            self.rejected_pf_updates += 1

            self.get_logger().warn(
                'Hard-rejected PF update: '
                f'innovation={raw_distance:.2f} m, '
                f'yaw={yaw_innovation:.2f} rad'
            )

            return

        # ---------------------------------------------------------------------
        # COVARIANCE-AWARE X/Y INNOVATION
        #
        # Build the PF X/Y covariance block and work in its eigen-directions.
        #
        # A corridor typically has:
        #   one small sigma  -> well observed lateral direction
        #   one large sigma  -> poorly observed longitudinal direction
        #
        # We correct strongly only along the low-sigma direction.
        # ---------------------------------------------------------------------

        covariance_xy = np.array([
            [
                covariance[0],
                covariance[1]
            ],
            [
                covariance[6],
                covariance[7]
            ]
        ], dtype=float)

        covariance_xy = 0.5 * (
            covariance_xy
            +
            covariance_xy.T
        )

        # Protect against tiny numerical negative eigenvalues.
        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance_xy
        )

        eigenvalues = np.maximum(
            eigenvalues,
            0.0
        )

        sigmas = np.sqrt(
            eigenvalues
        )

        weights = np.array([
            self.confidence_from_sigma(
                sigma,
                self.pf_sigma_pos_good,
                self.pf_sigma_pos_bad
            )
            for sigma in sigmas
        ], dtype=float)

        innovation_eigen = (
            eigenvectors.T
            @
            raw_position_innovation
        )

        weighted_position_innovation = (
            eigenvectors
            @
            (
                weights
                *
                innovation_eigen
            )
        )

        weighted_distance = float(
            np.linalg.norm(
                weighted_position_innovation
            )
        )

        self.last_weighted_innovation = (
            weighted_distance
        )

        self.last_pf_sigma_min = float(
            np.min(
                sigmas
            )
        )

        self.last_pf_sigma_max = float(
            np.max(
                sigmas
            )
        )

        self.last_pf_weight_min = float(
            np.min(
                weights
            )
        )

        self.last_pf_weight_max = float(
            np.max(
                weights
            )
        )

        # ---------------------------------------------------------------------
        # YAW CONFIDENCE
        # ---------------------------------------------------------------------

        sigma_yaw = math.sqrt(
            max(
                0.0,
                float(
                    covariance[35]
                )
            )
        )

        yaw_weight = (
            self.confidence_from_sigma(
                sigma_yaw,
                self.pf_sigma_yaw_good,
                self.pf_sigma_yaw_bad
            )
        )

        # ---------------------------------------------------------------------
        # BUILD A COVARIANCE-WEIGHTED PF TARGET POSE
        #
        # Important:
        # Uncertain directions stay at the current prediction.
        # Confident directions move toward PF.
        # ---------------------------------------------------------------------

        weighted_target_pose = np.array([
            predicted_pose[0]
            +
            weighted_position_innovation[0],

            predicted_pose[1]
            +
            weighted_position_innovation[1],

            wrap_angle(
                predicted_pose[2]
                +
                yaw_weight
                *
                yaw_innovation
            )
        ])

        self.target_map_to_odom = compose_pose(
            weighted_target_pose,
            inverse_pose(
                ekf_at_pf
            )
        )

        # If even one spatial direction is confidently observed, the target
        # movement in that direction may run at the high correction rate.
        #
        # Poorly observed directions already have near-zero innovation because
        # of the covariance weighting above.
        self.target_position_confidence = float(
            np.max(
                weights
            )
        )

        self.target_yaw_confidence = float(
            yaw_weight
        )

        self.accepted_pf_updates += 1

    # =========================================================================
    # EKF CALLBACK / SMOOTH TARGET TRACKING
    # =========================================================================

    def ekf_callback(
        self,
        msg
    ):

        t = stamp_to_sec(
            msg.header.stamp
        )

        pose_msg = msg.pose.pose

        ekf_pose = np.array([
            float(
                pose_msg.position.x
            ),

            float(
                pose_msg.position.y
            ),

            quaternion_to_yaw(
                pose_msg.orientation
            )
        ])

        self.odom_buffer.append(
            OdomSample(
                t=t,
                x=ekf_pose[0],
                y=ekf_pose[1],
                yaw=ekf_pose[2]
            )
        )

        if self.current_map_to_odom is None:
            return

        current_global_pose = compose_pose(
            self.current_map_to_odom,
            ekf_pose
        )

        now = self.get_clock().now()

        if self.last_ekf_publish_time is None:

            dt = 0.02

        else:

            dt = (
                now
                -
                self.last_ekf_publish_time
            ).nanoseconds * 1e-9

            dt = clamp(
                dt,
                0.005,
                0.050
            )

        self.last_ekf_publish_time = now

        # ---------------------------------------------------------------------
        # MOVE CURRENT GLOBAL POSE TOWARD PF TARGET IN VEHICLE-POSE SPACE
        # ---------------------------------------------------------------------

        if self.target_map_to_odom is not None:

            target_global_pose = compose_pose(
                self.target_map_to_odom,
                ekf_pose
            )

            position_error = (
                target_global_pose[:2]
                -
                current_global_pose[:2]
            )

            position_error_norm = float(
                np.linalg.norm(
                    position_error
                )
            )

            correction_speed = (
                self.low_conf_correction_speed
                +
                self.target_position_confidence
                *
                (
                    self.high_conf_correction_speed
                    -
                    self.low_conf_correction_speed
                )
            )

            max_translation_step = min(
                self.max_correction_step,
                correction_speed
                *
                dt
            )

            if (
                position_error_norm
                >
                max_translation_step
                and
                position_error_norm
                >
                1e-12
            ):

                applied_position = (
                    position_error
                    *
                    (
                        max_translation_step
                        /
                        position_error_norm
                    )
                )

            else:

                applied_position = (
                    position_error
                )

            yaw_error = wrap_angle(
                target_global_pose[2]
                -
                current_global_pose[2]
            )

            yaw_rate = (
                self.low_conf_yaw_rate
                +
                self.target_yaw_confidence
                *
                (
                    self.high_conf_yaw_rate
                    -
                    self.low_conf_yaw_rate
                )
            )

            max_yaw_step = min(
                self.max_yaw_step,
                yaw_rate
                *
                dt
            )

            applied_yaw = clamp(
                yaw_error,
                -max_yaw_step,
                max_yaw_step
            )

            corrected_global_pose = np.array([
                current_global_pose[0]
                +
                applied_position[0],

                current_global_pose[1]
                +
                applied_position[1],

                wrap_angle(
                    current_global_pose[2]
                    +
                    applied_yaw
                )
            ])

            self.last_applied_step = float(
                np.linalg.norm(
                    applied_position
                )
            )

            # Rebuild current map->odom from the already-bounded VEHICLE pose.
            self.current_map_to_odom = compose_pose(
                corrected_global_pose,
                inverse_pose(
                    ekf_pose
                )
            )

            current_global_pose = (
                corrected_global_pose
            )

        self.publish_global_odom(
            msg,
            current_global_pose
        )

    # =========================================================================
    # OUTPUT
    # =========================================================================

    def publish_global_odom(
        self,
        ekf_msg,
        global_pose
    ):

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

        out.pose.pose.position.x = float(
            global_pose[0]
        )

        out.pose.pose.position.y = float(
            global_pose[1]
        )

        out.pose.pose.position.z = 0.0

        q = quaternion_from_euler(
            0.0,
            0.0,
            float(
                global_pose[2]
            )
        )

        out.pose.pose.orientation.x = q[0]
        out.pose.pose.orientation.y = q[1]
        out.pose.pose.orientation.z = q[2]
        out.pose.pose.orientation.w = q[3]

        out.pose.covariance = list(
            self.latest_pf_covariance
        )

        out.twist = (
            ekf_msg.twist
        )

        self.localization_pub.publish(
            out
        )

        self.publish_count += 1

    # =========================================================================
    # STATUS
    # =========================================================================

    def status_callback(
        self
    ):

        if self.current_map_to_odom is None:

            self.get_logger().info(
                'Racing V3 waiting for synchronized PF/EKF anchor...'
            )

            return

        pf_age = -1.0

        if self.last_pf_receive_time is not None:

            pf_age = (
                self.get_clock().now()
                -
                self.last_pf_receive_time
            ).nanoseconds * 1e-9

        stale = (
            ' PF_STALE'
            if (
                pf_age >= 0.0
                and
                pf_age
                >
                self.pf_stale_warn_sec
            )
            else ''
        )

        self.get_logger().info(
            'Fuser-RACING-V3 | '
            f'pub={self.publish_count} | '
            f'PF accepted={self.accepted_pf_updates} '
            f'rejected={self.rejected_pf_updates} | '
            f'innovation raw/weighted='
            f'{self.last_raw_innovation:.2f}/'
            f'{self.last_weighted_innovation:.2f} m | '
            f'PF sigma='
            f'{self.last_pf_sigma_min:.3f}/'
            f'{self.last_pf_sigma_max:.3f} m | '
            f'weights='
            f'{self.last_pf_weight_min:.2f}/'
            f'{self.last_pf_weight_max:.2f} | '
            f'step={self.last_applied_step:.3f} m | '
            f'PF_age={pf_age:.3f}s'
            f'{stale}'
        )


# =============================================================================
# MAIN
# =============================================================================

def main(args=None):

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
