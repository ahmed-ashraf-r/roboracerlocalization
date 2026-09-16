#!/usr/bin/env python3

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32


class PurePursuitNode(Node):
    """
    Pure Pursuit path tracker with feedforward throttle.

    Throttle relation (from regression):
        throttle_norm = 0.04131 * target_velocity_mps
    """

    # Path to CSV with columns [positions_X, positions_y, Velocity]
    CSV_PATH = '/home/autodrive_devkit/src/f1tenth_control/previous_compete_test.csv'

    # Feedforward throttle gain
    # If throttle command is 0-100 scale, change to 4.131
    THROTTLE_RATIO = 0.04131

    # Loop rate
    DT = 0.01

    # Pure Pursuit parameters
    WHEELBASE = 0.3240
    MAX_STEER = 0.5236

    # Plot buffers
    MAX_SPEED_POINTS = 750
    PLOT_EVERY_N = 10

    def __init__(self):
        super().__init__('control_node')

        # ----- Car state -----
        self.position = np.array([0.8, 3.16])
        self.odom_position = np.array([0.8, 3.16])
        self.odom_vel_x = 0.0
        self.odom_vel_y = 0.0
        self.odom_speed = 0.0
        self.car_yaw = 0.0

        # ----- Path data -----
        self._load_path()

        # ----- Pure Pursuit state -----
        self.look_ahead = 2.0
        self.count = self._initial_index()
        self.speed_count = 0
        self.search_len = self.path_len / 5

        # ----- Buffers for plotting -----
        self.plot_counter = 0
        self.car_trail_x = []
        self.car_trail_y = []
        self.ips_trail_x = []
        self.ips_trail_y = []
        self.sim_time = 0.0
        self.time_log = []
        self.target_speed_log = []
        self.actual_speed_log = []
        self.odom_velx_log = []

        # ----- ROS interfaces -----
        self._setup_ros_interfaces()

        # ----- Matplotlib figures -----
        self._setup_figures()

        # ----- Main loop timer -----
        self.timer = self.create_timer(self.DT, self.timer_callback)

        self.get_logger().info('Pure Pursuit node started (ODOM ONLY).')

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_path(self):
        path_data = pd.read_csv(
            self.CSV_PATH,
            header=None,
            names=['positions_X', 'positions_y', 'Velocity'],
        )
        goal_list = list(zip(path_data['positions_X'], path_data['positions_y']))
        self.goal = np.array(goal_list)
        self.path_len = len(self.goal)
        self.vel_profile = path_data['Velocity'].to_numpy()

    def _initial_index(self):
        distances = np.sqrt(
            (self.goal[:, 0] - self.position[0]) ** 2
            + (self.goal[:, 1] - self.position[1]) ** 2
        )
        return int(np.argmin(distances))

    def _setup_ros_interfaces(self):
        # Only Odom is used for localization now
        self.create_subscription(
            Odometry,
            '/autodrive/roboracer_1/odom',
            self.odom_callback,
            10,
        )
        
        self.steer_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/steering_command',
            10,
        )
        self.throttle_pub = self.create_publisher(
            Float32,
            '/autodrive/roboracer_1/throttle_command',
            10,
        )

    def _setup_figures(self):
        plt.ion()

        # Figure 1: tracking
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.plot(self.goal[:, 0], self.goal[:, 1], 'k--', label='CSV Path')
        (self.car_plot,) = self.ax.plot([], [], 'ro', ms=8, label='Current Pose')
        (self.target_plot,) = self.ax.plot([], [], 'go', ms=8, label='Lookahead Point')
        (self.trail_plot,) = self.ax.plot([], [], 'b-', lw=1.5, label='Actual Path')
        (self.ips_plot,) = self.ax.plot([], [], 'm^', ms=8, label='Odom Pose')
        (self.ips_trail_plot,) = self.ax.plot([], [], 'm-', lw=1.2, alpha=0.7, label='Odom Path')
        self.ax.set_title('Pure Pursuit Tracking')
        self.ax.set_xlabel('X [m]')
        self.ax.set_ylabel('Y [m]')
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Figure 2: speed
        self.fig2, self.ax2 = plt.subplots(figsize=(8, 4))
        (self.target_speed_plot,) = self.ax2.plot([], [], 'g-', lw=1.5, label='Target Speed')
        (self.actual_speed_plot,) = self.ax2.plot([], [], 'b-', lw=1.5, label='Actual Speed')
        (self.odom_velx_plot,) = self.ax2.plot([], [], 'r--', lw=1.2, label='Odom Vel X')
        self.ax2.set_title('Speed Tracking')
        self.ax2.set_xlabel('Time [s]')
        self.ax2.set_ylabel('Speed [m/s]')
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True)
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        # Update both position variables to maintain plotting logic
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.position[0] = x
        self.position[1] = y
        self.odom_position[0] = x
        self.odom_position[1] = y

        # Velocity
        self.odom_vel_x = msg.twist.twist.linear.x
        self.odom_vel_y = msg.twist.twist.linear.y
        self.odom_speed = math.sqrt(self.odom_vel_x ** 2 + self.odom_vel_y ** 2)

        # Yaw from Quaternion
        q = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ]
        _, _, self.car_yaw = R.from_quat(q).as_euler('xyz')

    # ------------------------------------------------------------------
    # Pure Pursuit math
    # ------------------------------------------------------------------

    @staticmethod
    def _to_car_frame(xy_world, point_world, yaw):
        rot = np.array([
            [np.cos(yaw), np.sin(yaw)],
            [-np.sin(yaw), np.cos(yaw)],
        ])
        return rot @ (point_world - xy_world)

    def _curvature(self, xy_car_frame):
        y = xy_car_frame[1]
        return (2.0 * y) / (self.look_ahead ** 2)

    def _steering_angle(self, curvature):
        return np.arctan(self.WHEELBASE * curvature)

    # ------------------------------------------------------------------
    # Feedforward throttle
    # ------------------------------------------------------------------

    def _feedforward_throttle(self, target_speed):
        output = self.THROTTLE_RATIO * target_speed
        return max(min(output, 1.0), 0.0)

    # ------------------------------------------------------------------
    # Lookahead search
    # ------------------------------------------------------------------

    def _update_lookahead_index(self):
        start = self.count
        search_end = min(self.count + int(self.search_len), self.path_len)

        check_distance = np.sqrt(
            (self.goal[start:search_end, 0] - self.position[0]) ** 2
            + (self.goal[start:search_end, 1] - self.position[1]) ** 2
        )

        nearest = np.where(check_distance >= self.look_ahead)[0]
        speed_candidates = np.where(check_distance >= 0.0)[0]

        if len(nearest) > 0:
            self.count = start + int(nearest[0])
        else:
            self.count += 1

        if len(speed_candidates) > 0:
            self.speed_count = start + int(speed_candidates[0])
        else:
            self.speed_count = self.count

        if self.count >= self.path_len:
            self.count = 10
            self.speed_count = 10

        self.speed_count = min(self.speed_count, self.path_len - 1)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _update_plots(self):
        self.car_plot.set_data([self.position[0]], [self.position[1]])
        self.target_plot.set_data([self.goal[self.count, 0]], [self.goal[self.count, 1]])
        self.ips_trail_plot.set_data(self.ips_trail_x, self.ips_trail_y)
        self.ips_plot.set_data([self.odom_position[0]], [self.odom_position[1]])
        self.trail_plot.set_data(self.car_trail_x, self.car_trail_y)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.target_speed_plot.set_data(self.time_log, self.target_speed_log)
        self.actual_speed_plot.set_data(self.time_log, self.actual_speed_log)
        self.odom_velx_plot.set_data(self.time_log, self.odom_velx_log)
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.fig2.canvas.draw_idle()
        self.fig2.canvas.flush_events()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def timer_callback(self):
        # Log trail
        self.ips_trail_x.append(self.position[0])
        self.ips_trail_y.append(self.position[1])
        self.car_trail_x.append(self.odom_position[0])
        self.car_trail_y.append(self.odom_position[1])

        self.get_logger().info('Publishing : >_<')

        # Find lookahead point
        self._update_lookahead_index()

        # Steering
        xy_cf = self._to_car_frame(self.position, self.goal[self.count], self.car_yaw)
        curvature = self._curvature(xy_cf)
        steer = self._steering_angle(curvature) / self.MAX_STEER

        # Target velocity from profile
        target_velocity = 2.3 + self.vel_profile[self.speed_count] / 2.3

        # Dynamic lookahead
        self.look_ahead = 2.5 if target_velocity > 5.0 else 1.5

        # Feedforward throttle
        throttle_cmd = self._feedforward_throttle(target_velocity)

        # Publish
        st_msg = Float32()
        st_msg.data = float(steer)
        thr_msg = Float32()
        thr_msg.data = float(throttle_cmd)
        self.steer_pub.publish(st_msg)
        self.throttle_pub.publish(thr_msg)

        # Speed logging
        self.sim_time += self.DT
        self.time_log.append(self.sim_time)
        self.target_speed_log.append(float(target_velocity))
        self.actual_speed_log.append(float(self.odom_speed))
        self.odom_velx_log.append(float(self.odom_vel_x))

        if len(self.time_log) > self.MAX_SPEED_POINTS:
            del self.time_log[0]
            del self.target_speed_log[0]
            del self.actual_speed_log[0]
            del self.odom_velx_log[0]

        # Plotting (10 Hz)
        self.plot_counter += 1
        if self.plot_counter % self.PLOT_EVERY_N == 0:
            self._update_plots()

        # Debug
        self.get_logger().info(f'yaw angle: {round(self.car_yaw, 3)}')
        self.get_logger().info(f'steering command: {round(steer, 3)} >_<')
        self.get_logger().info(f'throttle command: {throttle_cmd} >_<')
        self.get_logger().info(f'Lookahead: {self.look_ahead} >_<')
        self.get_logger().info(f'index: {self.count} >_<')

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def save_path(self, filename='/home/autodrive_devkit/actual_path.csv'):
        np.savetxt(
            filename,
            np.column_stack((self.car_trail_x, self.car_trail_y)),
            delimiter=',',
            header='x,y',
            comments='',
        )


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close('all')
        node.save_path()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()