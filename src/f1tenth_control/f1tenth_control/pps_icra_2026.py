#!/usr/bin/env python3

import rclpy 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String , Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point 
import math

#==========================================================================
# with change point by fixed look ahead distance , PID Throttle
# on ICRA 2026 Competition Round 
# throttle 0.13 
# look_ahead = 1.5
#==========================================================================

#-----------------------------------------------------------------------
#-----------------------Global variables-------------------------------- 
#-----------------------------------------------------------------------

## car current position & orientation (IPS Ground Truth)
x_postition =  0.8
y_postition =  3.16
postition = np.array([x_postition , y_postition ])

# Wheel Odometry position & velocity
odom_postition = np.array([x_postition , y_postition ])
odom_current_vel_x = 0.0 
odom_current_vel_y = 0.0 
odom_current_speed = 0.0 
car_yaw = 0.0

# Particle Filter Odom position & velocity
pf_odom_position = np.array([x_postition, y_postition])
pf_odom_speed = 0.0

# Centerline Path of ICRA 2026 Competition 
path_data = pd.read_csv(
    '/home/autodrive_devkit/src/f1tenth_control/approved_2.csv',
    header=None,
    names=['positions_X', 'positions_y', 'Velocity']
)

#-----------------------------------------------------------------------------------
goal_list = list(zip(
    path_data['positions_X']  , 
    path_data['positions_y'] ))

goal = np.array(goal_list)
path_len = len(goal)

vel_profile = path_data['Velocity'].to_numpy()

# pure pursuit parameter 
velocity = 0.13  
look_ahead = 2.0 
wheelbase = 0.3240 

distances = np.sqrt((goal[:,0] - x_postition)**2 + (goal[:,1] - y_postition)**2)
index = np.argmin(distances)  # index of closest point

count = index   # start index  
speed_count = 0 
target_speed_idx = 0 

search_len = path_len / 5
search_end = min(count + int(search_len), path_len) # to avoid being out of range 

#-----------------------------------------------------------------------------------
#---------------------- PID CONTROL TROTTLE ----------------------------------------
#-----------------------------------------------------------------------------------

# Speed PID Controller parameters
K_FF = 0.0418 

speed_integral = 0.0
prev_speed_error = 0.0
prev_target_speed = 0.0 
dt_pid = 0.01  # Timer loop period (100 Hz -> 0.01s)

#------------------------------------------------------------- 
#---------------------- PLOTTING SETUP -----------------------
#------------------------------------------------------------- 

# Plotting state counter
plot_counter = 0 

# Wheel Odom trail
car_trail_x = []
car_trail_y = []

# IPS trail (ground-truth / comparison path)
ips_trail_x = []
ips_trail_y = []

# Particle Filter Odom trail
pf_trail_x = []
pf_trail_y = []

#------------------speed plotting -------------------------
time_log = []
target_speed_log = []
actual_speed_log = []
odom_velx_log = []
pf_speed_log = []    # --- NEW: Log for PF Odom speed ---
sim_time = 0.0  
  
MAX_SPEED_POINTS = 750

#-------------------------------

plt.ion() # Enable interactive mode
fig, ax = plt.subplots(figsize=(8, 8))

# Static plot elements
ax.plot(goal[:, 0], goal[:, 1], 'k--', label='CSV Path') 
car_plot, = ax.plot([], [], 'ro', markersize=8, label='Current Pose') 
target_plot, = ax.plot([], [], 'go', markersize=8, label='Lookahead Point') 
trail_plot,  = ax.plot([], [], 'b-', linewidth=1.5, label='Wheel Odom Path') 

# IPS plot elements
ips_plot, = ax.plot([], [], 'm^', markersize=8, label='IPS Pose')
ips_trail_plot, = ax.plot([], [], 'm-', linewidth=1.2, alpha=0.7, label='IPS Path')

# PF Odom plot elements (RED)
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

# ---- Figure 2: Speed comparison over time ----
fig2, ax2 = plt.subplots(figsize=(8, 4))
 
target_speed_plot, = ax2.plot([], [], 'g-', linewidth=1.5, label='Target Speed (profile)')
actual_speed_plot, = ax2.plot([], [], 'b-', linewidth=1.5, label='Wheel Odom Speed')
odom_velx_plot,    = ax2.plot([], [], 'c--', linewidth=1.2, label='Wheel Vel X') # Changed to cyan to free up red
pf_speed_plot,     = ax2.plot([], [], 'r-', linewidth=1.5, label='PF Odom Speed') # --- NEW: PF Speed plot (Red) ---
 
ax2.set_title("Speed Tracking: Target vs Wheel Odom vs PF Odom")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Speed [m/s]")
ax2.legend(loc='upper right')
ax2.grid(True)
 
fig2.canvas.draw()
fig2.canvas.flush_events()

#-------------------------------------------------------------
#-----------------------call back functions-------------------
#------------------------------------------------------------- 

def odom_callback(odom_msg):
    global odom_postition, odom_current_vel_x, odom_current_vel_y, odom_current_speed
    odom_postition[0] = odom_msg.pose.pose.position.x        
    odom_postition[1] = odom_msg.pose.pose.position.y
    odom_current_vel_x  = odom_msg.twist.twist.linear.x 
    odom_current_vel_y  = odom_msg.twist.twist.linear.y 
    odom_current_speed = math.sqrt(odom_current_vel_x**2 + odom_current_vel_y**2)

def pf_odom_callback(msg):
    """ Callback for the Particle Filter Odometry (Pose & Speed) """
    global pf_odom_position, pf_odom_speed
    
    # Extract Position
    pf_odom_position[0] = msg.pose.pose.position.x
    pf_odom_position[1] = msg.pose.pose.position.y
    
    # --- NEW: Extract Velocity ---
    vx = msg.twist.twist.linear.x
    vy = msg.twist.twist.linear.y
    pf_odom_speed = math.sqrt(vx**2 + vy**2)

def ips_callback(point_msg):
    global postition
    postition[0] = point_msg.x 
    postition[1] = point_msg.y 
 
def yaw_callback(imu_msg): 
    global car_yaw 
    qx = imu_msg.orientation.x
    qy = imu_msg.orientation.y
    qz = imu_msg.orientation.z
    qw = imu_msg.orientation.w
    r = R.from_quat([qx,qy,qz,qw])
    roll, pitch, car_yaw = r.as_euler('xyz')

#---------------------------------------------------------------------------
#-------------------------Pure pursuit functions----------------------------
#---------------------------------------------------------------------------

def transformation(xy_world_arr, point_world_arr, yaw): 
    R_T = np.array([ [np.cos(yaw) , np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]   ])
    point_car_frame = R_T @ (point_world_arr-xy_world_arr)
    return point_car_frame

def curvature_calc(xy_car_frame):
    x = xy_car_frame[0]
    y = xy_car_frame[1]
    curvature = (2* y) / (look_ahead*look_ahead)  
    return curvature

def steering_func(wh_base, gamma):
    steering_angle = np.arctan(wh_base * gamma) 
    return steering_angle

#---------------------------------------------------------------------------
#------------------------- PID throttle function----------------------------
#---------------------------------------------------------------------------
def speed_control(target_speed, actual_speed):
    global speed_integral, prev_speed_error 
    output = K_FF * target_speed 
    throttle = max(min(output, 1.0), 0.0)
    return throttle

#---------------------------------------------------------------------------
#-------------------- ROS 2 Timer_function ---------------------------------
#---------------------------------------------------------------------------

def timer_func(node, st_pub, thr_pub): 
    global postition, odom_postition, pf_odom_position, car_yaw, count, plot_counter, look_ahead 
    global car_trail_x, car_trail_y, ips_trail_x, ips_trail_y, pf_trail_x, pf_trail_y
    global speed_integral, prev_speed_error, odom_current_vel_x, odom_current_vel_y, odom_current_speed, pf_odom_speed
    global sim_time, time_log, target_speed_log, actual_speed_log, odom_velx_log, pf_speed_log
    global speed_count, target_speed_idx

    st = Float32()
    thr = Float32() 
    start = count   
    search_end = min(count + int(search_len), path_len) 

    # Append to trails
    ips_trail_x.append(postition[0])  
    ips_trail_y.append(postition[1])  
    
    car_trail_x.append(odom_postition[0])
    car_trail_y.append(odom_postition[1])

    pf_trail_x.append(pf_odom_position[0])
    pf_trail_y.append(pf_odom_position[1])
 
    node.get_logger().info("Publishing : >_<" )

    # Find lookahead target using IPS (as was originally implemented)
    check_distance = np.sqrt((goal[count:search_end ,0] - postition[0])**2 + (goal[count:search_end ,1] - postition[1])**2)
    nearest_idx = np.where(check_distance >= (look_ahead))[0]   
    target_speed_idx = np.where(check_distance >= (0.3))[0]   

    if len(nearest_idx) > 0:  
        count = start + nearest_idx[0]
    else:
        count += 1  

    if len(target_speed_idx) > 0:
        speed_count = start + int(target_speed_idx[0])
    else:
        speed_count = count

    if (count >= path_len):  
        count = 10
        speed_count = 10
    speed_count = min(speed_count, path_len - 1)
    
    # Steering calculation
    xy_cf = transformation(postition, goal[count], car_yaw)
    curve = curvature_calc(xy_cf)
    steer = steering_func(wheelbase, curve) / 0.5236
    st.data = float(steer)

    # Throttle calculation
    target_velocity = 2.0 

    if (target_velocity > 5.0):
        look_ahead = 2.5
    else:
        look_ahead = 1.5

    throttle_cmd = speed_control(target_velocity, odom_current_speed)
    thr.data = float(throttle_cmd)
    
    st_pub.publish(st) 
    thr_pub.publish(thr) 

    # Log speed history
    sim_time += dt_pid
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
        del pf_speed_log[0]     # --- NEW: Delete oldest PF speed ---
    
    # PLOTTING UPDATE (10 Hz)
    plot_counter += 1
    if plot_counter % 10 == 0:
        car_plot.set_data([postition[0]], [postition[1]])
        target_plot.set_data([goal[count, 0]], [goal[count, 1]])
        
        # Update IPS
        ips_plot.set_data([postition[0]], [postition[1]])
        ips_trail_plot.set_data(ips_trail_x, ips_trail_y)

        # Update Wheel Odom
        trail_plot.set_data(car_trail_x, car_trail_y)   
        
        # Update PF Odom
        pf_plot.set_data([pf_odom_position[0]], [pf_odom_position[1]])
        pf_trail_plot.set_data(pf_trail_x, pf_trail_y)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # Update Speed Plots
        target_speed_plot.set_data(time_log, target_speed_log)
        actual_speed_plot.set_data(time_log, actual_speed_log)
        odom_velx_plot.set_data(time_log, odom_velx_log)
        
        # --- NEW: Update PF Speed plot ---
        pf_speed_plot.set_data(time_log, pf_speed_log)
 
        ax2.relim()
        ax2.autoscale_view()
        fig2.canvas.draw_idle()
        fig2.canvas.flush_events()

#------------------------- Main ------------------------
def main(args=None):    
    rclpy.init(args=args)    
    my_node = rclpy.create_node('pps_icra_2026')

    # Subscribers
    car_odom = my_node.create_subscription(Odometry, '/autodrive/roboracer_1/odom', odom_callback, 10)
    car_pose = my_node.create_subscription(Point, '/autodrive/roboracer_1/ips', ips_callback, 10)
    imu_sub = my_node.create_subscription(Imu, '/autodrive/roboracer_1/imu', yaw_callback, 10)
    
    # Subscribe to PF Odom
    pf_odom = my_node.create_subscription(Odometry, '/pf/pose/odom', pf_odom_callback, 10)

    # Publishers
    steer_pub = my_node.create_publisher(Float32, "/autodrive/roboracer_1/steering_command", 10)  
    throttle_pub = my_node.create_publisher(Float32, "/autodrive/roboracer_1/throttle_command", 10)

    timer = my_node.create_timer(0.01, lambda:timer_func(my_node, steer_pub, throttle_pub))
    rclpy.spin(my_node)

    plt.close('all')
    np.savetxt('/home/autodrive_devkit/actual_path.csv',
            np.column_stack((car_trail_x, car_trail_y)),
            delimiter=',', header='x,y', comments='')
    
    my_node.destroy_timer(timer)
    my_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()