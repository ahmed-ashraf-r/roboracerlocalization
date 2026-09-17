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
from std_msgs.msg import Float32, String


# ==========================================================================
# Regulated Pure Pursuit + Feedforward Throttle
# ICRA 2026 Competition Round
# ==========================================================================

# -------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------

# Car current position & orientation
x_postition = 0.8
y_postition = 3.16
postition = np.array([x_postition, y_postition])

# IPS-based position (used ONLY for comparison / plotting, not for control)
odom_postition = np.array([x_postition, y_postition])
odom_current_vel_x = 0.0
odom_current_vel_y = 0.0
odom_current_speed = 0.0
car_yaw = 0.0

# --- NEW: Particle Filter Odom position & velocity ---
pf_odom_position = np.array([x_postition, y_postition])
pf_odom_speed = 0.0

# Centerline path of ICRA 2026 Competition
# CSV has NO header -> we assign column names manually
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
        a = np.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        b = np.hypot(p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        c = np.hypot(p_next[0] - p_prev[0], p_next[1] - p_prev[1])
        area = 0.5 * abs(
            (p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1])
            - (p_next[0] - p_prev[0]) * (p_curr[1] - p_prev[1])
        )
        denom = a * b * c
        curvature[i] = (4 * area / denom) if denom > 1e-9 else 0.0
    return curvature


path_curvature = compute_path_curvature(goal)
vel_profile = path_data['Velocity'].to_numpy()

# Pure pursuit parameters
velocity = 0.13
look_ahead = 2.0
wheelbase = 0.3240

# Offset look-ahead (cutting-corner fix) parameters
ALPHA_MAX = 0.05   # [m] max distance from path before offset stops growing
BETA_MAX = 0.9     # [1/m] max curvature difference before offset saturates

# goal is Nx2 -> [:,0] = x, [:,1] = y
distances = np.sqrt((goal[:, 0] - x_postition) ** 2 + (goal[:, 1] - y_postition) ** 2)
index = np.argmin(distances)

count = index
speed_count = 0
target_speed_idx = 0

search_len = path_len / 5
search_end = min(count + int(search_len), path_len)


# -------------------------------------------------------------------------
# Control throttle
# -------------------------------------------------------------------------

K_FF = 0.0418   # feedforward gain


# -------------------------------------------------------------------------
# Plotting setup
# -------------------------------------------------------------------------

plot_counter = 0

car_trail_x = []
car_trail_y = []

# IPS trail (ground-truth / comparison path)
ips_trail_x = []
ips_trail_y = []

# --- NEW: Particle Filter Odom trail ---
pf_trail_x = []
pf_trail_y = []

# Speed plotting
time_log = []
target_speed_log = []
actual_speed_log = []
odom_velx_log = []
pf_speed_log = []    # --- NEW: Log for PF Odom speed ---
sim_time = 0.0

MAX_SPEED_POINTS = 750

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(goal[:, 0], goal[:, 1], 'k--', label='CSV Path')
car_plot, = ax.plot([], [], 'ro', markersize=8, label='Current Pose')
target_plot, = ax.plot([], [], 'go', markersize=8, label='Lookahead Point')
trail_plot, = ax.plot([], [], 'b-', linewidth=1.5, label='Actual Path')
offset_lk, = ax.plot([], [], 'yo', markersize=8, label='Offset Lookahead')

ips_plot, = ax.plot([], [], 'm^', markersize=8, label='Odom Pose')
ips_trail_plot, = ax.plot([], [], 'm-', linewidth=1.2, alpha=0.7, label='Odom Path')

# --- NEW: PF Odom plot elements (RED) ---
pf_plot, = ax.plot([], [], 'rs', markersize=6, label='/pf/pose/odom Pose')
pf_trail_plot, = ax.plot([], [], 'r-', linewidth=1.5, alpha=0.8, label='/pf/pose/odom Path')

ax.set_title("Pure Pursuit Tracking & Odom Debugging")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.legend(loc='upper right')
ax.grid(True)
ax.axis('equal')

fig.canvas.draw()
fig.canvas.flush_events()


# Figure 2: Speed comparison over time
fig2, ax2 = plt.subplots(figsize=(8, 4))

target_speed_plot, = ax2.plot([], [], 'g-', linewidth=1.5, label='Target Speed (profile)')
actual_speed_plot, = ax2.plot([], [], 'b-', linewidth=1.5, label='Wheel Odom Speed')
odom_velx_plot, = ax2.plot([], [], 'c--', linewidth=1.2, label='Wheel Vel X') # Changed to cyan
pf_speed_plot, = ax2.plot([], [], 'r-', linewidth=1.5, label='PF Odom Speed') # --- NEW: PF Speed plot (Red) ---

ax2.set_title("Speed Tracking: Target vs Wheel Odom vs PF Odom")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Speed [m/s]")
ax2.legend(loc='upper right')
ax2.grid(True)

fig2.canvas.draw()
fig2.canvas.flush_events()


# -------------------------------------------------------------------------
# Callback functions
# -------------------------------------------------------------------------

def odom_callback(odom_msg):
    global odom_postition, odom_current_vel_x, odom_current_vel_y, odom_current_speed

    odom_postition[0] = odom_msg.pose.pose.position.x
    odom_postition[1] = odom_msg.pose.pose.position.y
    odom_current_vel_x = odom_msg.twist.twist.linear.x
    odom_current_vel_y = odom_msg.twist.twist.linear.y
    odom_current_speed = math.sqrt(odom_current_vel_x ** 2 + odom_current_vel_y ** 2)

def pf_odom_callback(msg):
    """ NEW: Callback for the Particle Filter Odometry (Pose & Speed) """
    global pf_odom_position, pf_odom_speed
    
    # Extract Position
    pf_odom_position[0] = msg.pose.pose.position.x
    pf_odom_position[1] = msg.pose.pose.position.y
    
    # Extract Velocity
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    pf_odom_speed = math.sqrt(vx**2 + vy**2)

def ips_callback(point_msg):
    """IPS callback - used ONLY for plotting / comparison against odom."""
    global postition

    postition[0] = point_msg.x
    postition[1] = point_msg.y


def yaw_callback(imu_msg):
    global car_yaw

    # Quaternion from IMU data
    qx = imu_msg.orientation.x
    qy = imu_msg.orientation.y
    qz = imu_msg.orientation.z
    qw = imu_msg.orientation.w

    # Convert to Euler to get yaw
    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, car_yaw = r.as_euler('xyz')


# -------------------------------------------------------------------------
# Pure pursuit functions
# -------------------------------------------------------------------------

def transformation(xy_world_arr, point_world_arr, yaw):
    R_T = np.array([
        [np.cos(yaw), np.sin(yaw)],
        [-np.sin(yaw), np.cos(yaw)]
    ])
    point_car_frame = R_T @ (point_world_arr - xy_world_arr)
    return point_car_frame


def curvature_calc(xy_car_frame):
    x = xy_car_frame[0]
    y = xy_car_frame[1]
    Lf_actual = math.hypot(x, y)
    if Lf_actual < 1e-6:
        return 0.0
    curvature = (2 * y) / (Lf_actual * Lf_actual)
    return curvature


def compute_offset_lookahead(position, pw_idx, pd_idx, path_xy, path_curvature, alpha_max, beta_max):
    """Section 3.2: shift look-ahead point pd -> pl to fight cutting-corner problem."""
    n = len(path_xy)
    pw = path_xy[pw_idx]
    pd = path_xy[pd_idx]

    p_wd = pd - pw
    len_wd = np.hypot(p_wd[0], p_wd[1])
    theta_pwd = math.atan2(p_wd[1], p_wd[0])

    prev_idx = (pw_idx - 1) % n
    next_idx = (pw_idx + 1) % n
    tangent = path_xy[next_idx] - path_xy[prev_idx]
    theta_pwl = math.atan2(tangent[1], tangent[0])

    theta = theta_pwl - theta_pwd
    theta = math.atan2(math.sin(theta), math.cos(theta))  # wrap to [-pi, pi]

    dist_v_pw = np.hypot(position[0] - pw[0], position[1] - pw[1])
    alpha = min(dist_v_pw / alpha_max, 1.0)

    cur_pw = path_curvature[pw_idx]
    cur_pd = path_curvature[pd_idx]
    if cur_pw >= cur_pd:
        beta = 0.0
    elif (cur_pd - cur_pw) < beta_max:
        beta = (cur_pd - cur_pw) / beta_max
    else:
        beta = 1.0

    tau = (1.0 - alpha) * beta

    MAX_OFFSET_DIST = 0.45
    len_dl = min(len_wd * math.tan(abs(theta)), MAX_OFFSET_DIST)
    sign_theta = 1.0 if theta >= 0 else -1.0
    theta_dl = theta_pwd + sign_theta * (math.pi / 2.0)

    p_l = np.array([
        pd[0] + tau * len_dl * math.cos(theta_dl),
        pd[1] + tau * len_dl * math.sin(theta_dl)
    ])
    return p_l, tau


def steering_func(wh_base, gamma):
    steering_angle = np.arctan(wh_base * gamma)
    return steering_angle


# -------------------------------------------------------------------------
# Throttle function
# -------------------------------------------------------------------------

def speed_control(target_speed, actual_speed):
    """Compute throttle command using feedforward on speed (m/s)."""
    output = K_FF * target_speed
    throttle = max(min(output, 1.0), 0.0)
    return throttle


# -------------------------------------------------------------------------
# ROS 2 timer function
# -------------------------------------------------------------------------

def timer_func(node, st_pub, thr_pub):
    global postition, odom_postition, pf_odom_position, car_yaw, count, plot_counter, look_ahead
    global car_trail_x, car_trail_y, ips_trail_x, ips_trail_y, pf_trail_x, pf_trail_y
    global odom_current_vel_x, odom_current_vel_y, odom_current_speed, pf_odom_speed
    global sim_time, time_log, target_speed_log, actual_speed_log, odom_velx_log, pf_speed_log
    global speed_count, target_speed_idx

    st = Float32()
    thr = Float32()
    start = count
    search_end = min(count + int(search_len), path_len)

    ips_trail_x.append(postition[0])
    ips_trail_y.append(postition[1])

    car_trail_x.append(odom_postition[0])
    car_trail_y.append(odom_postition[1])

    # --- NEW: Append to PF trail ---
    pf_trail_x.append(pf_odom_position[0])
    pf_trail_y.append(pf_odom_position[1])

    node.get_logger().info("Publishing : >_<")

    # Create a search window that wraps around path_len seamlessly
    search_indices = [(count + i) % path_len for i in range(int(search_len))]
    search_goals = goal[search_indices]

    check_distance = np.linalg.norm(search_goals - postition, axis=1)
    nearest_idx = np.where(check_distance >= look_ahead)[0]

    if len(nearest_idx) > 0:
        count = search_indices[nearest_idx[0]]
    else:
        count = (count + 1) % path_len

    target_speed_idx = np.where(check_distance >= 0.3)[0]

    if len(target_speed_idx) > 0:
        speed_count = start + int(target_speed_idx[0])
    else:
        speed_count = count

    if count >= path_len:
        count = 10
        speed_count = 10
    speed_count = min(speed_count, path_len - 1)

    # Find nearest waypoint (pw) for offset-lookahead correction
    pw_idx = int(np.argmin(
        np.sqrt((goal[:, 0] - postition[0]) ** 2 + (goal[:, 1] - postition[1]) ** 2)
    ))
    p_l, tau_val = compute_offset_lookahead(
        postition, pw_idx, count, goal, path_curvature, ALPHA_MAX, BETA_MAX
    )

    # 1. Steering calculation via Pure Pursuit (using corrected look-ahead point p_l)
    xy_cf = transformation(postition, p_l, car_yaw)
    curve = curvature_calc(xy_cf)
    steer = steering_func(wheelbase, curve) / 0.5236
    st.data = float(steer)

    # 2. Throttle calculation via speed profile + feedforward
    target_velocity = vel_profile[speed_count] / 1.2
    look_ahead = np.clip(0.45 * odom_current_speed, 0.7, 2.0)

    throttle_cmd = speed_control(target_velocity, odom_current_speed)
    thr.data = float(throttle_cmd)

    st_pub.publish(st)
    thr_pub.publish(thr)

    # Log speed history for Figure 2
    sim_time += 0.01
    time_log.append(sim_time)
    target_speed_log.append(float(target_velocity))
    actual_speed_log.append(float(odom_current_speed))
    odom_velx_log.append(float(odom_current_vel_x))
    
    # --- NEW: Log PF Speed ---
    pf_speed_log.append(float(pf_odom_speed))

    if len(time_log) > MAX_SPEED_POINTS:
        del time_log[0]
        del target_speed_log[0]
        del actual_speed_log[0]
        del odom_velx_log[0]
        del pf_speed_log[0]    # --- NEW: Delete oldest PF speed ---

    # Non-blocking plotting update (every 10 cycles = 10 Hz)
    plot_counter += 1
    if plot_counter % 10 == 0:
        car_plot.set_data([postition[0]], [postition[1]])
        target_plot.set_data([goal[count, 0]], [goal[count, 1]])
        ips_trail_plot.set_data(ips_trail_x, ips_trail_y)
        offset_lk.set_data([p_l[0]], [p_l[1]])

        ips_plot.set_data([odom_postition[0]], [odom_postition[1]])
        trail_plot.set_data(car_trail_x, car_trail_y)

        # --- NEW: Update PF Odom Plot ---
        pf_plot.set_data([pf_odom_position[0]], [pf_odom_position[1]])
        pf_trail_plot.set_data(pf_trail_x, pf_trail_y)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        target_speed_plot.set_data(time_log, target_speed_log)
        actual_speed_plot.set_data(time_log, actual_speed_log)
        odom_velx_plot.set_data(time_log, odom_velx_log)
        
        # --- NEW: Update PF Speed plot ---
        pf_speed_plot.set_data(time_log, pf_speed_log)

        ax2.relim()
        ax2.autoscale_view()

        fig2.canvas.draw_idle()
        fig2.canvas.flush_events()

    node.get_logger().info(f" yaw angle  : {round(car_yaw, 3)}")
    node.get_logger().info(f" steering command value : {round(steer, 3)} >_<")
    node.get_logger().info(f" throttle command value : {throttle_cmd} >_<")
    node.get_logger().info(f" Lookahead  : {look_ahead} >_<")
    node.get_logger().info(f" index   : {count} >_<")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    my_node = rclpy.create_node('pps_icra_2026')

    car_odom = my_node.create_subscription(
        Odometry, '/autodrive/roboracer_1/odom', odom_callback, 10
    )
    car_pose = my_node.create_subscription(
        Point, '/autodrive/roboracer_1/ips', ips_callback, 10
    )
    imu_sub = my_node.create_subscription(
        Imu, '/autodrive/roboracer_1/imu', yaw_callback, 10
    )
    
    # --- NEW: Subscribe to PF Odom ---
    pf_odom_sub = my_node.create_subscription(
        Odometry, '/pf/pose/odom', pf_odom_callback, 10
    )

    steer_pub = my_node.create_publisher(
        Float32, "/autodrive/roboracer_1/steering_command", 10
    )
    throttle_pub = my_node.create_publisher(
        Float32, "/autodrive/roboracer_1/throttle_command", 10
    )

    timer = my_node.create_timer(
        0.01, lambda: timer_func(my_node, steer_pub, throttle_pub)
    )

    rclpy.spin(my_node)

    plt.close('all')
    np.savetxt(
        '/home/autodrive_devkit/actual_path.csv',
        np.column_stack((car_trail_x, car_trail_y)),
        delimiter=',', header='x,y', comments=''
    )

    my_node.destroy_timer(timer)
    my_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()