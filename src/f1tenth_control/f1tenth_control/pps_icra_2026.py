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
# Regulated Pure Pursuit + Feedforward Throttle
# ICRA 2026 Competition Round
#
# CONTROL SOURCES:
#   Position : /pf/pose/odom
#   Speed    : /odometry/filtered (EKF)
#   Yaw      : /autodrive/roboracer_1/imu
#
#   IPS       : plotting / comparison only
#   Simulator Odom : plotting / comparison only
# ==========================================================================


# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------

# Initial car position
x_postition = 0.8
y_postition = 3.16

# IPS-based position
# Used ONLY for comparison / plotting
postition = np.array([x_postition, y_postition])

# Simulator wheel odometry
# Used ONLY for comparison / plotting
odom_postition = np.array([x_postition, y_postition])
odom_current_vel_x = 0.0
odom_current_vel_y = 0.0
odom_current_speed = 0.0

# IMU yaw
car_yaw = 0.0

# Particle Filter Odom
# Position is used for actual vehicle control.
# Speed is kept only for plotting / comparison.
pf_odom_position = np.array([x_postition, y_postition])
pf_odom_speed = 0.0
pf_odom_received = False

# EKF odometry
# Speed source used for actual vehicle control.
ekf_speed = 0.0
ekf_speed_received = False


# -------------------------------------------------------------------------
# Centerline path of ICRA 2026 Competition
# CSV has NO header -> assign column names manually
# -------------------------------------------------------------------------

path_data = pd.read_csv(
    '/home/autodrive_devkit/src/f1tenth_control/approved_2.csv',
    header=None,
    names=['positions_X', 'positions_y', 'Velocity']
)

goal_list = list(zip(
    path_data['positions_X'],
    path_data['positions_y']
))

goal = np.array(goal_list)
path_len = len(goal)


# -------------------------------------------------------------------------
# Path curvature
# -------------------------------------------------------------------------

def compute_path_curvature(path_xy):

    n = len(path_xy)
    curvature = np.zeros(n)

    for i in range(n):

        p_prev = path_xy[(i - 1) % n]
        p_curr = path_xy[i]
        p_next = path_xy[(i + 1) % n]

        a = np.hypot(
            p_curr[0] - p_prev[0],
            p_curr[1] - p_prev[1]
        )

        b = np.hypot(
            p_next[0] - p_curr[0],
            p_next[1] - p_curr[1]
        )

        c = np.hypot(
            p_next[0] - p_prev[0],
            p_next[1] - p_prev[1]
        )

        area = 0.5 * abs(
            (p_curr[0] - p_prev[0]) *
            (p_next[1] - p_prev[1])
            -
            (p_next[0] - p_prev[0]) *
            (p_curr[1] - p_prev[1])
        )

        denom = a * b * c

        curvature[i] = (
            4 * area / denom
            if denom > 1e-9
            else 0.0
        )

    return curvature


path_curvature = compute_path_curvature(goal)

vel_profile = path_data['Velocity'].to_numpy()


# -------------------------------------------------------------------------
# Pure Pursuit parameters
# -------------------------------------------------------------------------

velocity = 0.13

look_ahead = 2.0

wheelbase = 0.3240


# -------------------------------------------------------------------------
# Offset look-ahead parameters
# -------------------------------------------------------------------------

ALPHA_MAX = 0.05
BETA_MAX = 0.9


# -------------------------------------------------------------------------
# Initial waypoint
# -------------------------------------------------------------------------

distances = np.sqrt(
    (goal[:, 0] - x_postition) ** 2 +
    (goal[:, 1] - y_postition) ** 2
)

index = np.argmin(distances)

count = index

speed_count = 0

search_len = path_len / 5

search_end = min(
    count + int(search_len),
    path_len
)


# -------------------------------------------------------------------------
# Dynamic CSV speed-profile preview
# -------------------------------------------------------------------------
#
# This RoboRacer is a small-scale vehicle and the approved path has
# approximately 0.10 m waypoint spacing.  The preview is intentionally
# modest: at the maximum target speed (~5.8 m/s after the /1.7 scaling)
# it is about 1.43 m (~14 waypoints), not several metres.
#
# The gain has units of seconds: distance ~= speed * preview_time.
# It compensates for the measured transient lag without using PID or
# abruptly cutting throttle.
#
SPEED_PREVIEW_BASE = 0.15       # [m]
SPEED_PREVIEW_TIME = 0.22       # [s]
SPEED_PREVIEW_MIN = 0.35        # [m]
SPEED_PREVIEW_MAX = 1.45        # [m]


# -------------------------------------------------------------------------
# Throttle
# -------------------------------------------------------------------------

K_FF = 0.04131


# -------------------------------------------------------------------------
# Plotting setup
# -------------------------------------------------------------------------

plot_counter = 0

car_trail_x = []
car_trail_y = []

# IPS trail
ips_trail_x = []
ips_trail_y = []

# PF trail
pf_trail_x = []
pf_trail_y = []


# -------------------------------------------------------------------------
# Speed plotting
# -------------------------------------------------------------------------

time_log = []

target_speed_log = []

actual_speed_log = []

odom_velx_log = []

pf_speed_log = []

sim_time = 0.0

MAX_SPEED_POINTS = 750


# -------------------------------------------------------------------------
# Figure 1
# -------------------------------------------------------------------------

plt.ion()

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(
    goal[:, 0],
    goal[:, 1],
    'k--',
    label='CSV Path'
)

car_plot, = ax.plot(
    [],
    [],
    'ro',
    markersize=8,
    label='PF Control Pose'
)

target_plot, = ax.plot(
    [],
    [],
    'go',
    markersize=8,
    label='Lookahead Point'
)

trail_plot, = ax.plot(
    [],
    [],
    'b-',
    linewidth=1.5,
    label='Wheel Odom Path'
)

offset_lk, = ax.plot(
    [],
    [],
    'yo',
    markersize=8,
    label='Offset Lookahead'
)

ips_plot, = ax.plot(
    [],
    [],
    'm^',
    markersize=8,
    label='IPS Pose'
)

ips_trail_plot, = ax.plot(
    [],
    [],
    'm-',
    linewidth=1.2,
    alpha=0.7,
    label='IPS Path'
)

pf_plot, = ax.plot(
    [],
    [],
    'rs',
    markersize=6,
    label='/pf/pose/odom Pose'
)

pf_trail_plot, = ax.plot(
    [],
    [],
    'r-',
    linewidth=1.5,
    alpha=0.8,
    label='/pf/pose/odom Path'
)

ax.set_title(
    "Pure Pursuit Tracking & Odom Debugging"
)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")

ax.legend(
    loc='upper right'
)

ax.grid(True)

ax.axis('equal')

fig.canvas.draw()
fig.canvas.flush_events()


# -------------------------------------------------------------------------
# Figure 2
# -------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(8, 4))

target_speed_plot, = ax2.plot(
    [],
    [],
    'g-',
    linewidth=1.5,
    label='Target Speed'
)

actual_speed_plot, = ax2.plot(
    [],
    [],
    'b-',
    linewidth=1.5,
    label='EKF Control Speed'
)

odom_velx_plot, = ax2.plot(
    [],
    [],
    'c--',
    linewidth=1.2,
    label='Wheel Vel X'
)

pf_speed_plot, = ax2.plot(
    [],
    [],
    'r-',
    linewidth=1.5,
    label='PF Odom Speed'
)

ax2.set_title(
    "Speed Tracking: Target vs EKF Speed"
)

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Speed [m/s]")

ax2.legend(
    loc='upper right'
)

ax2.grid(True)

fig2.canvas.draw()
fig2.canvas.flush_events()


# ==========================================================================
# CALLBACK FUNCTIONS
# ==========================================================================


# -------------------------------------------------------------------------
# Simulator wheel odometry callback
# Used ONLY for plotting / comparison
# -------------------------------------------------------------------------

def odom_callback(odom_msg):

    global odom_postition
    global odom_current_vel_x
    global odom_current_vel_y
    global odom_current_speed

    odom_postition[0] = (
        odom_msg.pose.pose.position.x
    )

    odom_postition[1] = (
        odom_msg.pose.pose.position.y
    )

    odom_current_vel_x = (
        odom_msg.twist.twist.linear.x
    )

    odom_current_vel_y = (
        odom_msg.twist.twist.linear.y
    )

    odom_current_speed = math.sqrt(
        odom_current_vel_x ** 2 +
        odom_current_vel_y ** 2
    )


# -------------------------------------------------------------------------
# Particle Filter Odometry callback
#
# Used for:
#   Position -> CONTROL
#   Speed    -> plotting / comparison only
#
# NOT used for yaw
# -------------------------------------------------------------------------

def pf_odom_callback(msg):

    global pf_odom_position
    global pf_odom_speed
    global pf_odom_received

    # -------------------------------------------------
    # Position used for path tracking
    # -------------------------------------------------

    pf_odom_position[0] = (
        msg.pose.pose.position.x
    )

    pf_odom_position[1] = (
        msg.pose.pose.position.y
    )

    # -------------------------------------------------
    # PF speed kept only for plotting / comparison.
    # EKF speed is the control-speed source.
    # -------------------------------------------------

    vx = msg.twist.twist.linear.x

    vy = msg.twist.twist.linear.y

    pf_odom_speed = math.sqrt(
        vx ** 2 +
        vy ** 2
    )

    pf_odom_received = True


# -------------------------------------------------------------------------
# EKF odometry callback
#
# Used for:
#   Speed -> CONTROL
#
# Position from this topic is NOT used for global path tracking.
# -------------------------------------------------------------------------

def ekf_odom_callback(msg):

    global ekf_speed
    global ekf_speed_received

    vx = float(msg.twist.twist.linear.x)
    vy = float(msg.twist.twist.linear.y)

    # Speed magnitude from the smooth 50 Hz EKF estimate.
    # The vehicle races forward, but hypot() also makes this robust
    # to a small lateral component in the EKF output.
    ekf_speed = math.hypot(vx, vy)

    ekf_speed_received = True


# -------------------------------------------------------------------------
# IPS callback
#
# Used ONLY for plotting / comparison
# -------------------------------------------------------------------------

def ips_callback(point_msg):

    global postition

    postition[0] = point_msg.x

    postition[1] = point_msg.y


# -------------------------------------------------------------------------
# IMU callback
#
# IMU is the ONLY source of yaw for control
# -------------------------------------------------------------------------

def yaw_callback(imu_msg):

    global car_yaw

    qx = imu_msg.orientation.x
    qy = imu_msg.orientation.y
    qz = imu_msg.orientation.z
    qw = imu_msg.orientation.w

    r = R.from_quat([
        qx,
        qy,
        qz,
        qw
    ])

    roll, pitch, car_yaw = r.as_euler('xyz')


# ==========================================================================
# PURE PURSUIT FUNCTIONS
# ==========================================================================


def transformation(
    xy_world_arr,
    point_world_arr,
    yaw
):

    R_T = np.array([
        [
            np.cos(yaw),
            np.sin(yaw)
        ],
        [
            -np.sin(yaw),
            np.cos(yaw)
        ]
    ])

    point_car_frame = (
        R_T @
        (point_world_arr - xy_world_arr)
    )

    return point_car_frame


def curvature_calc(xy_car_frame):

    x = xy_car_frame[0]

    y = xy_car_frame[1]

    Lf_actual = math.hypot(
        x,
        y
    )

    if Lf_actual < 1e-6:
        return 0.0

    curvature = (
        2 * y
    ) / (
        Lf_actual * Lf_actual
    )

    return curvature


def compute_offset_lookahead(
    position,
    pw_idx,
    pd_idx,
    path_xy,
    path_curvature,
    alpha_max,
    beta_max
):

    n = len(path_xy)

    pw = path_xy[pw_idx]

    pd = path_xy[pd_idx]

    p_wd = pd - pw

    len_wd = np.hypot(
        p_wd[0],
        p_wd[1]
    )

    theta_pwd = math.atan2(
        p_wd[1],
        p_wd[0]
    )

    prev_idx = (
        pw_idx - 1
    ) % n

    next_idx = (
        pw_idx + 1
    ) % n

    tangent = (
        path_xy[next_idx] -
        path_xy[prev_idx]
    )

    theta_pwl = math.atan2(
        tangent[1],
        tangent[0]
    )

    theta = (
        theta_pwl -
        theta_pwd
    )

    theta = math.atan2(
        math.sin(theta),
        math.cos(theta)
    )

    dist_v_pw = np.hypot(
        position[0] - pw[0],
        position[1] - pw[1]
    )

    alpha = min(
        dist_v_pw / alpha_max,
        1.0
    )

    cur_pw = path_curvature[pw_idx]

    cur_pd = path_curvature[pd_idx]

    if cur_pw >= cur_pd:

        beta = 0.0

    elif (
        cur_pd - cur_pw
    ) < beta_max:

        beta = (
            cur_pd - cur_pw
        ) / beta_max

    else:

        beta = 1.0

    tau = (
        1.0 - alpha
    ) * beta

    MAX_OFFSET_DIST = 0.45

    len_dl = min(
        len_wd *
        math.tan(abs(theta)),
        MAX_OFFSET_DIST
    )

    sign_theta = (
        1.0
        if theta >= 0
        else -1.0
    )

    theta_dl = (
        theta_pwd +
        sign_theta *
        (math.pi / 2.0)
    )

    p_l = np.array([
        pd[0] +
        tau *
        len_dl *
        math.cos(theta_dl),

        pd[1] +
        tau *
        len_dl *
        math.sin(theta_dl)
    ])

    return p_l, tau


def steering_func(
    wh_base,
    gamma
):

    steering_angle = np.arctan(
        wh_base * gamma
    )

    return steering_angle


# ==========================================================================
# THROTTLE
# ==========================================================================


def speed_control(
    target_speed
):

    # Feedforward-only throttle law identified from the simulator's
    # steady-state throttle-to-speed regression.
    output = (
        K_FF *
        target_speed
    )

    throttle = max(
        min(output, 1.0),
        0.0
    )

    return throttle


def advance_path_index_by_distance(
    start_idx,
    path_xy,
    distance_m
):
    """
    Move forward along the CSV path by arc length.

    This uses path distance, not Euclidean distance.  That is important
    near bends, because a point that is geometrically close can be much
    farther ahead along the racing line.
    """

    n = len(path_xy)

    if n == 0:
        return 0

    idx = int(start_idx) % n
    travelled = 0.0

    # Protect against a malformed path containing many zero-length
    # segments by limiting traversal to one complete lap.
    for _ in range(n):

        if travelled >= distance_m:
            break

        next_idx = (idx + 1) % n

        segment_length = float(
            np.linalg.norm(
                path_xy[next_idx] -
                path_xy[idx]
            )
        )

        travelled += segment_length
        idx = next_idx

    return idx


# ==========================================================================
# ROS 2 TIMER
# ==========================================================================


def timer_func(
    node,
    st_pub,
    thr_pub
):

    global pf_odom_position
    global car_yaw
    global pf_odom_received
    global ekf_speed
    global ekf_speed_received
    global count
    global plot_counter
    global look_ahead

    global car_trail_x
    global car_trail_y
    global ips_trail_x
    global ips_trail_y
    global pf_trail_x
    global pf_trail_y

    global odom_current_vel_x
    global odom_current_vel_y
    global odom_current_speed
    global pf_odom_speed

    global sim_time
    global time_log
    global target_speed_log
    global actual_speed_log
    global odom_velx_log
    global pf_speed_log

    global speed_count


    # ----------------------------------------------------------------------
    # ROS messages
    # ----------------------------------------------------------------------

    st = Float32()

    thr = Float32()


    # ----------------------------------------------------------------------
    # Wait for the two control-state sources
    # ----------------------------------------------------------------------

    if not pf_odom_received or not ekf_speed_received:

        node.get_logger().warn(
            'Waiting for PF position and EKF speed before '
            'publishing control commands...'
        )

        return


    # ----------------------------------------------------------------------
    # CONTROL SOURCES
    #
    # Position -> Particle Filter       /pf/pose/odom
    # Yaw      -> IMU                   /autodrive/roboracer_1/imu
    # Speed    -> EKF                   /odometry/filtered
    #
    # PF speed remains available only for plotting / comparison.
    # ----------------------------------------------------------------------

    control_position = (
        pf_odom_position
    )

    control_yaw = (
        car_yaw
    )

    control_speed = (
        ekf_speed
    )


    # ----------------------------------------------------------------------
    # Current waypoint
    # ----------------------------------------------------------------------

    start = count

    search_end = min(
        count + int(search_len),
        path_len
    )


    # ----------------------------------------------------------------------
    # Trails for plotting
    # ----------------------------------------------------------------------

    ips_trail_x.append(
        postition[0]
    )

    ips_trail_y.append(
        postition[1]
    )

    car_trail_x.append(
        odom_postition[0]
    )

    car_trail_y.append(
        odom_postition[1]
    )

    pf_trail_x.append(
        pf_odom_position[0]
    )

    pf_trail_y.append(
        pf_odom_position[1]
    )


    node.get_logger().info(
        "Publishing : >_<"
    )


    # ----------------------------------------------------------------------
    # Search for lookahead point
    # ----------------------------------------------------------------------

    search_indices = [
        (count + i) % path_len
        for i in range(int(search_len))
    ]

    search_goals = goal[
        search_indices
    ]


    check_distance = np.linalg.norm(
        search_goals -
        control_position,
        axis=1
    )


    nearest_idx = np.where(
        check_distance >= look_ahead
    )[0]


    if len(nearest_idx) > 0:

        count = search_indices[
            nearest_idx[0]
        ]

    else:

        count = (
            count + 1
        ) % path_len


    if count >= path_len:

        count = 10


    # ----------------------------------------------------------------------
    # Find nearest waypoint for offset lookahead and speed preview
    # ----------------------------------------------------------------------

    pw_idx = int(
        np.argmin(
            np.sqrt(
                (goal[:, 0] -
                 control_position[0]) ** 2 +

                (goal[:, 1] -
                 control_position[1]) ** 2
            )
        )
    )


    # ----------------------------------------------------------------------
    # Dynamic feedforward speed-profile preview
    #
    # Small-scale-car tuning:
    #   1.0 m/s -> 0.37 m
    #   3.0 m/s -> 0.81 m
    #   5.0 m/s -> 1.25 m
    #   5.8 m/s -> 1.43 m
    #
    # Only the point sampled from the CSV moves forward.  Throttle remains
    # the same feedforward law: throttle = 0.04131 * target_velocity.
    # There is no PID and no throttle cut.
    # ----------------------------------------------------------------------

    speed_preview_distance = float(
        np.clip(
            SPEED_PREVIEW_BASE +
            SPEED_PREVIEW_TIME * control_speed,
            SPEED_PREVIEW_MIN,
            SPEED_PREVIEW_MAX
        )
    )

    speed_count = advance_path_index_by_distance(
        pw_idx,
        goal,
        speed_preview_distance
    )


    p_l, tau_val = (
        compute_offset_lookahead(
            control_position,
            pw_idx,
            count,
            goal,
            path_curvature,
            ALPHA_MAX,
            BETA_MAX
        )
    )


    # ======================================================================
    # 1. STEERING
    #
    # Position = PF
    # Yaw      = IMU
    # ======================================================================

    xy_cf = transformation(
        control_position,
        p_l,
        control_yaw
    )

    curve = curvature_calc(
        xy_cf
    )

    steer = (
        steering_func(
            wheelbase,
            curve
        )
        / 0.5236
    )

    st.data = float(
        steer
    )


    # ======================================================================
    # 2. THROTTLE
    #
    # Throttle = feedforward from CSV target speed
    # EKF speed is NOT fed back into throttle
    # ======================================================================

    target_velocity = (
        vel_profile[speed_count]
        / 1.7
    )

    look_ahead = np.clip(
        0.45 * control_speed,
        0.7,
        2.0
    )


    throttle_cmd = speed_control(
        target_velocity
    )

    thr.data = float(
        throttle_cmd
    )


    # ----------------------------------------------------------------------
    # Publish commands
    # ----------------------------------------------------------------------

    st_pub.publish(
        st
    )

    thr_pub.publish(
        thr
    )


    # ======================================================================
    # LOGGING
    # ======================================================================

    sim_time += 0.01

    time_log.append(
        sim_time
    )

    target_speed_log.append(
        float(target_velocity)
    )

    actual_speed_log.append(
        float(control_speed)
    )

    odom_velx_log.append(
        float(odom_current_vel_x)
    )

    pf_speed_log.append(
        float(pf_odom_speed)
    )


    if len(time_log) > MAX_SPEED_POINTS:

        del time_log[0]

        del target_speed_log[0]

        del actual_speed_log[0]

        del odom_velx_log[0]

        del pf_speed_log[0]


    # ======================================================================
    # PLOTTING
    # ======================================================================

    plot_counter += 1


    if plot_counter % 10 == 0:

        # PF control pose
        car_plot.set_data(
            [control_position[0]],
            [control_position[1]]
        )

        # Lookahead point
        target_plot.set_data(
            [goal[count, 0]],
            [goal[count, 1]]
        )

        # IPS path
        ips_trail_plot.set_data(
            ips_trail_x,
            ips_trail_y
        )

        # Offset lookahead
        offset_lk.set_data(
            [p_l[0]],
            [p_l[1]]
        )

        # Wheel odometry
        ips_plot.set_data(
            [odom_postition[0]],
            [odom_postition[1]]
        )

        trail_plot.set_data(
            car_trail_x,
            car_trail_y
        )

        # PF odometry
        pf_plot.set_data(
            [pf_odom_position[0]],
            [pf_odom_position[1]]
        )

        pf_trail_plot.set_data(
            pf_trail_x,
            pf_trail_y
        )


        fig.canvas.draw_idle()

        fig.canvas.flush_events()


        # Speed plots
        target_speed_plot.set_data(
            time_log,
            target_speed_log
        )

        actual_speed_plot.set_data(
            time_log,
            actual_speed_log
        )

        odom_velx_plot.set_data(
            time_log,
            odom_velx_log
        )

        pf_speed_plot.set_data(
            time_log,
            pf_speed_log
        )


        ax2.relim()

        ax2.autoscale_view()


        fig2.canvas.draw_idle()

        fig2.canvas.flush_events()


    # ======================================================================
    # DEBUG LOG
    # ======================================================================

    node.get_logger().info(
        f"IMU yaw angle : "
        f"{round(control_yaw, 3)}"
    )

    node.get_logger().info(
        f"steering command value : "
        f"{round(steer, 3)} >_<"
    )

    node.get_logger().info(
        f"throttle command value : "
        f"{throttle_cmd} >_<"
    )

    node.get_logger().info(
        f"Lookahead : "
        f"{look_ahead} >_<"
    )

    node.get_logger().info(
        f"Speed preview : "
        f"{round(speed_preview_distance, 3)} m, "
        f"speed index : {speed_count} >_<"
    )

    node.get_logger().info(
        f"index : "
        f"{count} >_<"
    )


# ==========================================================================
# MAIN
# ==========================================================================


def main(args=None):

    rclpy.init(
        args=args
    )

    my_node = rclpy.create_node(
        'pps_icra_2026'
    )


    # ----------------------------------------------------------------------
    # Simulator wheel odometry
    # Comparison / plotting ONLY
    # ----------------------------------------------------------------------

    car_odom = my_node.create_subscription(
        Odometry,
        '/autodrive/roboracer_1/odom',
        odom_callback,
        10
    )


    # ----------------------------------------------------------------------
    # IPS
    # Comparison / plotting ONLY
    # ----------------------------------------------------------------------

    car_pose = my_node.create_subscription(
        Point,
        '/autodrive/roboracer_1/ips',
        ips_callback,
        10
    )


    # ----------------------------------------------------------------------
    # IMU
    #
    # THIS IS THE YAW SOURCE FOR CONTROL
    # ----------------------------------------------------------------------

    imu_sub = my_node.create_subscription(
        Imu,
        '/autodrive/roboracer_1/imu',
        yaw_callback,
        10
    )


    # ----------------------------------------------------------------------
    # Particle Filter Odometry
    #
    # Position -> control
    # Speed    -> plotting / comparison only
    # ----------------------------------------------------------------------

    pf_odom_sub = my_node.create_subscription(
        Odometry,
        '/pf/pose/odom',
        pf_odom_callback,
        10
    )


    # ----------------------------------------------------------------------
    # EKF Odometry
    #
    # Speed -> control
    # ----------------------------------------------------------------------

    ekf_odom_sub = my_node.create_subscription(
        Odometry,
        '/odometry/filtered',
        ekf_odom_callback,
        10
    )


    # ----------------------------------------------------------------------
    # Steering publisher
    # ----------------------------------------------------------------------

    steer_pub = my_node.create_publisher(
        Float32,
        '/autodrive/roboracer_1/steering_command',
        10
    )


    # ----------------------------------------------------------------------
    # Throttle publisher
    # ----------------------------------------------------------------------

    throttle_pub = my_node.create_publisher(
        Float32,
        '/autodrive/roboracer_1/throttle_command',
        10
    )


    # ----------------------------------------------------------------------
    # Control loop
    # ----------------------------------------------------------------------

    timer = my_node.create_timer(
        0.01,
        lambda: timer_func(
            my_node,
            steer_pub,
            throttle_pub
        )
    )


    # ----------------------------------------------------------------------
    # Spin
    # ----------------------------------------------------------------------

    rclpy.spin(
        my_node
    )


    # ----------------------------------------------------------------------
    # Shutdown / save path
    # ----------------------------------------------------------------------

    plt.close(
        'all'
    )

    np.savetxt(
        '/home/autodrive_devkit/actual_path.csv',
        np.column_stack(
            (
                car_trail_x,
                car_trail_y
            )
        ),
        delimiter=',',
        header='x,y',
        comments=''
    )


    my_node.destroy_timer(
        timer
    )

    my_node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()

