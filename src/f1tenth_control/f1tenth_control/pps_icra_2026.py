#!/usr/bin/env python3

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32


# ==========================================================================
# RoboRacer ICRA 2026 - High-Rate Global Localization Pure Pursuit
#
# CONTROL STATE
#   Position : /localization/odom
#              PF-corrected global pose propagated by EKF at ~50 Hz
#
#   Speed    : /localization/odom.twist
#              This is the EKF twist copied by global_odom_fuser, therefore
#              it is synchronized with the global pose and arrives at ~50 Hz.
#
#   Yaw      : /autodrive/roboracer_1/imu
#
# DIAGNOSTIC ONLY
#   /pf/pose/odom
#   /autodrive/roboracer_1/odom
#   /autodrive/roboracer_1/ips
#
# LONGITUDINAL CONTROL
#   Feedforward only:
#       throttle = 0.04131 * target_velocity
#
#   NO PID.
#   NO overspeed throttle cut.
# ==========================================================================


# ==========================================================================
# MAIN RACING TUNING
# ==========================================================================

# /localization/odom and EKF are both ~50 Hz, so run control at 50 Hz.
CONTROL_PERIOD = 0.020          # [s] = 50 Hz

# Proven baseline after Global Odom Fuser V2.
# Keep 1.50 for the first latency-compensation validation.
# After several clean laps, the next planned test is 1.30.
VELOCITY_DIVISOR = 1.50

# Measured simulator feedforward relation.
K_FF = 0.04131

# Vehicle geometry.
WHEELBASE = 0.3240
MAX_STEERING_RAD = 0.5236

# Steering lookahead.
STEER_LOOKAHEAD_GAIN = 0.45
STEER_LOOKAHEAD_MIN = 0.70
STEER_LOOKAHEAD_MAX = 2.00

# Measured steering actuator delay from rosbag analysis was about 0.12 s.
# Start with 0.10 s compensation so we anticipate the actuator without
# over-predicting the vehicle motion.
STEERING_LATENCY_COMP = 0.10   # [s]

# Dynamic speed preview.
# ~1.43 m at 5.8 m/s and ~1.98 m at 8.3 m/s.
SPEED_PREVIEW_BASE = 0.15       # [m]
SPEED_PREVIEW_TIME = 0.22       # [s]
SPEED_PREVIEW_MIN = 0.35        # [m]
SPEED_PREVIEW_MAX = 2.05        # [m]

# Offset-lookahead tuning.
ALPHA_MAX = 0.05
BETA_MAX = 0.90
MAX_OFFSET_DIST = 0.45

# Local path association window.
NEAREST_SEARCH_BEHIND = 12
NEAREST_SEARCH_AHEAD = 90

# Development diagnostics.
# Disable plotting for official timed runs.
ENABLE_PLOTTING = False
PLOT_EVERY_N = 25              # 2 Hz
LOG_EVERY_N = 10               # 5 Hz


# ==========================================================================
# GLOBAL STATE
# ==========================================================================

x_position = 0.8
y_position = 3.16

# CONTROL state from /localization/odom.
localization_position = np.array(
    [x_position, y_position],
    dtype=float
)
localization_speed = 0.0
localization_received = False

# IMU yaw and yaw rate.
car_yaw = 0.0
car_yaw_rate = 0.0
imu_received = False

# Diagnostics only.
pf_odom_position = np.array(
    [x_position, y_position],
    dtype=float
)
pf_odom_speed = 0.0

sim_odom_position = np.array(
    [x_position, y_position],
    dtype=float
)
sim_odom_speed = 0.0

ips_position = np.array(
    [x_position, y_position],
    dtype=float
)


# ==========================================================================
# LOAD TRACK
# ==========================================================================

path_data = pd.read_csv(
    '/home/autodrive_devkit/src/f1tenth_control/approved_2.csv',
    header=None,
    names=[
        'positions_X',
        'positions_y',
        'Velocity'
    ]
)

goal = path_data[
    ['positions_X', 'positions_y']
].to_numpy(dtype=float)

vel_profile = path_data[
    'Velocity'
].to_numpy(dtype=float)

path_len = len(goal)


# ==========================================================================
# PATH PRECOMPUTATION
# ==========================================================================

def compute_path_curvature(path_xy):

    n = len(path_xy)
    curvature = np.zeros(n)

    for i in range(n):

        p_prev = path_xy[(i - 1) % n]
        p_curr = path_xy[i]
        p_next = path_xy[(i + 1) % n]

        a = np.linalg.norm(
            p_curr - p_prev
        )

        b = np.linalg.norm(
            p_next - p_curr
        )

        c = np.linalg.norm(
            p_next - p_prev
        )

        cross_z = (
            (p_curr[0] - p_prev[0]) *
            (p_next[1] - p_prev[1])
            -
            (p_curr[1] - p_prev[1]) *
            (p_next[0] - p_prev[0])
        )

        area = 0.5 * abs(cross_z)

        denom = a * b * c

        curvature[i] = (
            4.0 * area / denom
            if denom > 1e-9
            else 0.0
        )

    return curvature


path_curvature = compute_path_curvature(
    goal
)


initial_distances = np.linalg.norm(
    goal -
    np.array(
        [x_position, y_position]
    ),
    axis=1
)

nearest_path_idx = int(
    np.argmin(initial_distances)
)

lookahead_idx = nearest_path_idx
speed_idx = nearest_path_idx


# ==========================================================================
# LOGGING / PLOTTING
# ==========================================================================

control_counter = 0
sim_time = 0.0

MAX_SPEED_POINTS = 750

time_log = []
target_speed_log = []
control_speed_log = []
sim_speed_log = []
pf_speed_log = []

localization_trail_x = []
localization_trail_y = []

pf_trail_x = []
pf_trail_y = []

sim_trail_x = []
sim_trail_y = []

ips_trail_x = []
ips_trail_y = []


if ENABLE_PLOTTING:

    plt.ion()

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        goal[:, 0],
        goal[:, 1],
        'k--',
        label='CSV Path'
    )

    localization_pose_plot, = ax.plot(
        [],
        [],
        'ro',
        markersize=8,
        label='/localization/odom Control Pose'
    )

    localization_trail_plot, = ax.plot(
        [],
        [],
        'b-',
        linewidth=1.8,
        label='/localization/odom Path'
    )

    pf_pose_plot, = ax.plot(
        [],
        [],
        'rs',
        markersize=5,
        label='PF Pose'
    )

    pf_trail_plot, = ax.plot(
        [],
        [],
        'r-',
        linewidth=1.0,
        alpha=0.65,
        label='PF Path'
    )

    lookahead_plot, = ax.plot(
        [],
        [],
        'go',
        markersize=7,
        label='Steering Lookahead'
    )

    offset_lookahead_plot, = ax.plot(
        [],
        [],
        'yo',
        markersize=7,
        label='Offset Lookahead'
    )

    sim_trail_plot, = ax.plot(
        [],
        [],
        'c-',
        linewidth=1.0,
        alpha=0.7,
        label='Simulator Odom Path'
    )

    ips_trail_plot, = ax.plot(
        [],
        [],
        'm-',
        linewidth=1.0,
        alpha=0.7,
        label='IPS Path'
    )

    ax.set_title(
        'High-Rate Global Localization Pure Pursuit'
    )

    ax.set_xlabel(
        'X [m]'
    )

    ax.set_ylabel(
        'Y [m]'
    )

    ax.axis(
        'equal'
    )

    ax.grid(
        True
    )

    ax.legend(
        loc='upper right'
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 4)
    )

    target_speed_plot, = ax2.plot(
        [],
        [],
        'g-',
        linewidth=1.5,
        label='Target Speed'
    )

    control_speed_plot, = ax2.plot(
        [],
        [],
        'b-',
        linewidth=1.5,
        label='Localization/EKF Control Speed'
    )

    sim_speed_plot, = ax2.plot(
        [],
        [],
        'c--',
        linewidth=1.2,
        label='Simulator Reference Speed'
    )

    pf_speed_plot, = ax2.plot(
        [],
        [],
        'r-',
        linewidth=1.2,
        label='PF Published Speed'
    )

    ax2.set_title(
        'Speed Tracking'
    )

    ax2.set_xlabel(
        'Time [s]'
    )

    ax2.set_ylabel(
        'Speed [m/s]'
    )

    ax2.grid(
        True
    )

    ax2.legend(
        loc='upper right'
    )


# ==========================================================================
# CALLBACKS
# ==========================================================================

def localization_odom_callback(msg):

    global localization_position
    global localization_speed
    global localization_received

    localization_position[0] = float(
        msg.pose.pose.position.x
    )

    localization_position[1] = float(
        msg.pose.pose.position.y
    )

    # global_odom_fuser copies fresh EKF twist into /localization/odom.
    vx = float(
        msg.twist.twist.linear.x
    )

    vy = float(
        msg.twist.twist.linear.y
    )

    localization_speed = math.hypot(
        vx,
        vy
    )

    localization_received = True


def yaw_callback(msg):

    global car_yaw
    global car_yaw_rate
    global imu_received

    q = msg.orientation

    rotation = R.from_quat([
        q.x,
        q.y,
        q.z,
        q.w
    ])

    _, _, car_yaw = rotation.as_euler(
        'xyz'
    )

    car_yaw_rate = float(
        msg.angular_velocity.z
    )

    imu_received = True


def pf_odom_callback(msg):

    global pf_odom_position
    global pf_odom_speed

    pf_odom_position[0] = float(
        msg.pose.pose.position.x
    )

    pf_odom_position[1] = float(
        msg.pose.pose.position.y
    )

    vx = float(
        msg.twist.twist.linear.x
    )

    vy = float(
        msg.twist.twist.linear.y
    )

    pf_odom_speed = math.hypot(
        vx,
        vy
    )


def simulator_odom_callback(msg):

    global sim_odom_position
    global sim_odom_speed

    sim_odom_position[0] = float(
        msg.pose.pose.position.x
    )

    sim_odom_position[1] = float(
        msg.pose.pose.position.y
    )

    vx = float(
        msg.twist.twist.linear.x
    )

    vy = float(
        msg.twist.twist.linear.y
    )

    sim_odom_speed = math.hypot(
        vx,
        vy
    )


def ips_callback(msg):

    global ips_position

    ips_position[0] = float(
        msg.x
    )

    ips_position[1] = float(
        msg.y
    )


# ==========================================================================
# PATH INDEXING
# ==========================================================================

def wrapped_indices(
    center_idx,
    behind,
    ahead,
    n
):

    return np.array([
        (center_idx + offset) % n
        for offset in range(
            -behind,
            ahead + 1
        )
    ], dtype=int)


def update_nearest_path_index(
    position,
    previous_idx
):

    indices = wrapped_indices(
        previous_idx,
        NEAREST_SEARCH_BEHIND,
        NEAREST_SEARCH_AHEAD,
        path_len
    )

    local_points = goal[
        indices
    ]

    distances = np.linalg.norm(
        local_points - position,
        axis=1
    )

    return int(
        indices[
            int(
                np.argmin(distances)
            )
        ]
    )


def advance_path_index_by_distance(
    start_idx,
    distance_m
):

    idx = int(
        start_idx
    ) % path_len

    travelled = 0.0

    for _ in range(path_len):

        if travelled >= distance_m:
            break

        next_idx = (
            idx + 1
        ) % path_len

        travelled += float(
            np.linalg.norm(
                goal[next_idx] -
                goal[idx]
            )
        )

        idx = next_idx

    return idx


def find_steering_lookahead_index(
    start_idx,
    position,
    desired_distance
):

    idx = int(
        start_idx
    ) % path_len

    for _ in range(path_len):

        if np.linalg.norm(
            goal[idx] - position
        ) >= desired_distance:
            return idx

        idx = (
            idx + 1
        ) % path_len

    return idx


# ==========================================================================
# STEERING LATENCY COMPENSATION
# ==========================================================================

def predict_control_pose(
    position,
    yaw,
    speed,
    yaw_rate,
    prediction_time
):
    """
    Predict the vehicle pose forward to approximately where the vehicle will
    be when the steering actuator has responded to the command being sent now.

    The prediction is used ONLY for steering.  Speed-profile indexing remains
    tied to the current /localization/odom position.

    Motion model:
      - straight-line prediction when yaw rate is nearly zero
      - constant-yaw-rate circular-arc prediction otherwise
    """

    if prediction_time <= 0.0:
        return (
            position.copy(),
            yaw
        )

    # Nearly straight motion.
    if abs(yaw_rate) < 1e-4:

        predicted_position = np.array([
            position[0]
            + speed
            * prediction_time
            * math.cos(yaw),

            position[1]
            + speed
            * prediction_time
            * math.sin(yaw)
        ])

        return (
            predicted_position,
            yaw
        )

    # Constant-yaw-rate circular arc.
    predicted_yaw = (
        yaw
        + yaw_rate
        * prediction_time
    )

    radius = (
        speed /
        yaw_rate
    )

    predicted_position = np.array([
        position[0]
        + radius
        * (
            math.sin(predicted_yaw)
            - math.sin(yaw)
        ),

        position[1]
        - radius
        * (
            math.cos(predicted_yaw)
            - math.cos(yaw)
        )
    ])

    predicted_yaw = math.atan2(
        math.sin(predicted_yaw),
        math.cos(predicted_yaw)
    )

    return (
        predicted_position,
        predicted_yaw
    )


# ==========================================================================
# PURE PURSUIT
# ==========================================================================

def transformation(
    position_world,
    point_world,
    yaw
):

    c = math.cos(
        yaw
    )

    s = math.sin(
        yaw
    )

    rotation_world_to_car = np.array([
        [c, s],
        [-s, c]
    ])

    return (
        rotation_world_to_car @
        (
            point_world -
            position_world
        )
    )


def curvature_calc(
    point_car_frame
):

    x = float(
        point_car_frame[0]
    )

    y = float(
        point_car_frame[1]
    )

    lookahead_actual = math.hypot(
        x,
        y
    )

    if lookahead_actual < 1e-6:
        return 0.0

    return (
        2.0 * y /
        (
            lookahead_actual *
            lookahead_actual
        )
    )


def compute_offset_lookahead(
    position,
    nearest_idx,
    destination_idx
):

    pw = goal[
        nearest_idx
    ]

    pd = goal[
        destination_idx
    ]

    p_wd = (
        pd - pw
    )

    len_wd = np.linalg.norm(
        p_wd
    )

    theta_pwd = math.atan2(
        p_wd[1],
        p_wd[0]
    )

    previous_idx = (
        nearest_idx - 1
    ) % path_len

    next_idx = (
        nearest_idx + 1
    ) % path_len

    tangent = (
        goal[next_idx] -
        goal[previous_idx]
    )

    theta_path = math.atan2(
        tangent[1],
        tangent[0]
    )

    theta = math.atan2(
        math.sin(
            theta_path -
            theta_pwd
        ),
        math.cos(
            theta_path -
            theta_pwd
        )
    )

    distance_vehicle_to_pw = np.linalg.norm(
        position - pw
    )

    alpha = min(
        distance_vehicle_to_pw /
        ALPHA_MAX,
        1.0
    )

    curvature_here = path_curvature[
        nearest_idx
    ]

    curvature_destination = path_curvature[
        destination_idx
    ]

    if curvature_here >= curvature_destination:
        beta = 0.0

    elif (
        curvature_destination -
        curvature_here
    ) < BETA_MAX:

        beta = (
            curvature_destination -
            curvature_here
        ) / BETA_MAX

    else:
        beta = 1.0

    tau = (
        1.0 -
        alpha
    ) * beta

    offset_distance = min(
        len_wd *
        math.tan(
            abs(theta)
        ),
        MAX_OFFSET_DIST
    )

    direction = (
        1.0
        if theta >= 0.0
        else -1.0
    )

    offset_angle = (
        theta_pwd +
        direction *
        math.pi / 2.0
    )

    point = np.array([
        pd[0] +
        tau *
        offset_distance *
        math.cos(
            offset_angle
        ),

        pd[1] +
        tau *
        offset_distance *
        math.sin(
            offset_angle
        )
    ])

    return point


# ==========================================================================
# FEEDFORWARD THROTTLE
# ==========================================================================

def speed_control(
    target_speed
):

    throttle = (
        K_FF *
        target_speed
    )

    return float(
        np.clip(
            throttle,
            0.0,
            1.0
        )
    )


# ==========================================================================
# PLOTTING
# ==========================================================================

def update_plots(
    control_position,
    offset_point,
    steering_target_idx
):

    if not ENABLE_PLOTTING:
        return

    localization_pose_plot.set_data(
        [control_position[0]],
        [control_position[1]]
    )

    localization_trail_plot.set_data(
        localization_trail_x,
        localization_trail_y
    )

    pf_pose_plot.set_data(
        [pf_odom_position[0]],
        [pf_odom_position[1]]
    )

    pf_trail_plot.set_data(
        pf_trail_x,
        pf_trail_y
    )

    lookahead_plot.set_data(
        [goal[steering_target_idx, 0]],
        [goal[steering_target_idx, 1]]
    )

    offset_lookahead_plot.set_data(
        [offset_point[0]],
        [offset_point[1]]
    )

    sim_trail_plot.set_data(
        sim_trail_x,
        sim_trail_y
    )

    ips_trail_plot.set_data(
        ips_trail_x,
        ips_trail_y
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    target_speed_plot.set_data(
        time_log,
        target_speed_log
    )

    control_speed_plot.set_data(
        time_log,
        control_speed_log
    )

    sim_speed_plot.set_data(
        time_log,
        sim_speed_log
    )

    pf_speed_plot.set_data(
        time_log,
        pf_speed_log
    )

    ax2.relim()
    ax2.autoscale_view()

    fig2.canvas.draw_idle()
    fig2.canvas.flush_events()


# ==========================================================================
# CONTROL LOOP
# ==========================================================================

def timer_func(
    node,
    steering_pub,
    throttle_pub
):

    global control_counter
    global sim_time
    global nearest_path_idx
    global lookahead_idx
    global speed_idx

    if (
        not localization_received
        or not imu_received
    ):

        if (
            control_counter %
            LOG_EVERY_N
        ) == 0:

            node.get_logger().warn(
                'Waiting for /localization/odom and IMU...'
            )

        control_counter += 1
        return

    control_position = (
        localization_position.copy()
    )

    control_speed = float(
        localization_speed
    )

    control_yaw = float(
        car_yaw
    )


    # ----------------------------------------------------------------------
    # Track branch association.
    # ----------------------------------------------------------------------

    nearest_path_idx = update_nearest_path_index(
        control_position,
        nearest_path_idx
    )


    # ----------------------------------------------------------------------
    # Steering with measured actuator-latency compensation.
    #
    # IMPORTANT:
    #   - current global pose is still used for speed-profile indexing
    #   - predicted pose is used ONLY for steering
    # ----------------------------------------------------------------------

    steering_position, steering_yaw = (
        predict_control_pose(
            control_position,
            control_yaw,
            control_speed,
            car_yaw_rate,
            STEERING_LATENCY_COMP
        )
    )

    # Associate the predicted steering pose with the same local branch of
    # the racing line.  Do not overwrite the current nearest_path_idx because
    # speed planning must remain tied to the actual localization pose.
    steering_nearest_idx = update_nearest_path_index(
        steering_position,
        nearest_path_idx
    )

    steering_lookahead = float(
        np.clip(
            STEER_LOOKAHEAD_GAIN *
            control_speed,
            STEER_LOOKAHEAD_MIN,
            STEER_LOOKAHEAD_MAX
        )
    )

    lookahead_idx = find_steering_lookahead_index(
        steering_nearest_idx,
        steering_position,
        steering_lookahead
    )

    offset_point = compute_offset_lookahead(
        steering_position,
        steering_nearest_idx,
        lookahead_idx
    )

    point_car_frame = transformation(
        steering_position,
        offset_point,
        steering_yaw
    )

    curvature_command = curvature_calc(
        point_car_frame
    )

    steering_angle = math.atan(
        WHEELBASE *
        curvature_command
    )

    normalized_steering = float(
        np.clip(
            steering_angle /
            MAX_STEERING_RAD,
            -1.0,
            1.0
        )
    )


    # ----------------------------------------------------------------------
    # Feedforward target-speed preview.
    #
    # Speed is used only to choose how far forward in the CSV to read.
    # It is NOT used as throttle feedback.
    # ----------------------------------------------------------------------

    speed_preview_distance = float(
        np.clip(
            SPEED_PREVIEW_BASE +
            SPEED_PREVIEW_TIME *
            control_speed,
            SPEED_PREVIEW_MIN,
            SPEED_PREVIEW_MAX
        )
    )

    speed_idx = advance_path_index_by_distance(
        nearest_path_idx,
        speed_preview_distance
    )

    target_velocity = float(
        vel_profile[
            speed_idx
        ] /
        VELOCITY_DIVISOR
    )

    throttle_command = speed_control(
        target_velocity
    )


    # ----------------------------------------------------------------------
    # Publish.
    # ----------------------------------------------------------------------

    steering_msg = Float32()
    steering_msg.data = normalized_steering

    throttle_msg = Float32()
    throttle_msg.data = throttle_command

    steering_pub.publish(
        steering_msg
    )

    throttle_pub.publish(
        throttle_msg
    )


    # ----------------------------------------------------------------------
    # Diagnostics.
    # ----------------------------------------------------------------------

    sim_time += CONTROL_PERIOD

    localization_trail_x.append(
        control_position[0]
    )

    localization_trail_y.append(
        control_position[1]
    )

    pf_trail_x.append(
        pf_odom_position[0]
    )

    pf_trail_y.append(
        pf_odom_position[1]
    )

    sim_trail_x.append(
        sim_odom_position[0]
    )

    sim_trail_y.append(
        sim_odom_position[1]
    )

    ips_trail_x.append(
        ips_position[0]
    )

    ips_trail_y.append(
        ips_position[1]
    )

    time_log.append(
        sim_time
    )

    target_speed_log.append(
        target_velocity
    )

    control_speed_log.append(
        control_speed
    )

    sim_speed_log.append(
        sim_odom_speed
    )

    pf_speed_log.append(
        pf_odom_speed
    )

    if len(
        time_log
    ) > MAX_SPEED_POINTS:

        del time_log[0]
        del target_speed_log[0]
        del control_speed_log[0]
        del sim_speed_log[0]
        del pf_speed_log[0]


    if (
        control_counter %
        LOG_EVERY_N
    ) == 0:

        node.get_logger().info(
            f'v={control_speed:.2f} m/s, '
            f'target={target_velocity:.2f} m/s, '
            f'throttle={throttle_command:.3f}, '
            f'steer={normalized_steering:.3f}, '
            f'steer_L={steering_lookahead:.2f} m, '
            f'speed_preview={speed_preview_distance:.2f} m, '
            f'latency_comp={STEERING_LATENCY_COMP:.2f} s, '
            f'idx={nearest_path_idx}'
        )


    if (
        ENABLE_PLOTTING
        and
        control_counter %
        PLOT_EVERY_N
        == 0
    ):

        update_plots(
            control_position,
            offset_point,
            lookahead_idx
        )


    control_counter += 1


# ==========================================================================
# MAIN
# ==========================================================================

def main(
    args=None
):

    rclpy.init(
        args=args
    )

    node = rclpy.create_node(
        'pps_icra_2026'
    )


    # CONTROL STATE
    localization_sub = node.create_subscription(
        Odometry,
        '/localization/odom',
        localization_odom_callback,
        10
    )

    imu_sub = node.create_subscription(
        Imu,
        '/autodrive/roboracer_1/imu',
        yaw_callback,
        10
    )


    # DIAGNOSTIC SOURCES
    pf_sub = node.create_subscription(
        Odometry,
        '/pf/pose/odom',
        pf_odom_callback,
        10
    )

    simulator_odom_sub = node.create_subscription(
        Odometry,
        '/autodrive/roboracer_1/odom',
        simulator_odom_callback,
        10
    )

    ips_sub = node.create_subscription(
        Point,
        '/autodrive/roboracer_1/ips',
        ips_callback,
        10
    )


    # COMMAND OUTPUTS
    steering_pub = node.create_publisher(
        Float32,
        '/autodrive/roboracer_1/steering_command',
        10
    )

    throttle_pub = node.create_publisher(
        Float32,
        '/autodrive/roboracer_1/throttle_command',
        10
    )


    timer = node.create_timer(
        CONTROL_PERIOD,
        lambda: timer_func(
            node,
            steering_pub,
            throttle_pub
        )
    )


    node.get_logger().info(
        'High-rate global localization controller started. '
        f'control={1.0 / CONTROL_PERIOD:.1f} Hz, '
        f'velocity_divisor={VELOCITY_DIVISOR:.2f}, '
        f'preview_max={SPEED_PREVIEW_MAX:.2f} m, '
        f'steering_latency_comp={STEERING_LATENCY_COMP:.2f} s, '
        f'plotting={ENABLE_PLOTTING}'
    )


    try:
        rclpy.spin(
            node
        )

    except KeyboardInterrupt:
        pass

    finally:

        if ENABLE_PLOTTING:
            plt.close(
                'all'
            )

        np.savetxt(
            '/home/autodrive_devkit/localization_control_path.csv',
            np.column_stack((
                localization_trail_x,
                localization_trail_y
            )),
            delimiter=',',
            header='x,y',
            comments=''
        )

        node.destroy_timer(
            timer
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
