#!/usr/bin/env python3

import math
import time
from collections import deque
from threading import Lock

import numpy as np
import range_libc
import rclpy
from rclpy.node import Node
from rclpy.time import Time as RosTime
from tf2_ros import TransformBroadcaster
import tf_transformations

from geometry_msgs.msg import (
    PointStamped,
    PolygonStamped,
    PoseArray,
    PoseStamped,
    PoseWithCovarianceStamped,
    TransformStamped,
)
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan

from particle_filter import utils as Utils


VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


class ParticleFiler(Node):
    """Monte-Carlo localization for RoboRacer.

    Important convention in this corrected implementation:
      * particles always represent map -> base_frame (roboracer_1)
      * RangeLib ray-casting is performed from map -> lidar_frame by applying
        the configured base->lidar rigid transform to every particle
      * odometry is used only as relative local motion
      * each physical LiDAR scan is consumed once
      * resampling is performed only when effective sample size is low
    """

    def __init__(self):
        super().__init__('particle_filter')

        # ------------------------------------------------------------------
        # Existing parameters (kept for compatibility with localize.yaml)
        # ------------------------------------------------------------------
        self.declare_parameter('angle_step', 12)
        self.declare_parameter('max_particles', 2000)
        self.declare_parameter('max_viz_particles', 60)
        self.declare_parameter('squash_factor', 2.2)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('theta_discretization', 112)
        self.declare_parameter('range_method', 'cddt')
        self.declare_parameter('rangelib_variant', 2)
        self.declare_parameter('fine_timing', 0)
        self.declare_parameter('publish_odom', 1)
        self.declare_parameter('viz', 1)
        self.declare_parameter('z_short', 0.01)
        self.declare_parameter('z_max', 0.07)
        self.declare_parameter('z_rand', 0.12)
        self.declare_parameter('z_hit', 0.75)
        self.declare_parameter('sigma_hit', 8.0)
        self.declare_parameter('motion_dispersion_x', 0.005)
        self.declare_parameter('motion_dispersion_y', 0.025)
        self.declare_parameter('motion_dispersion_theta', 0.25)
        self.declare_parameter('scan_topic', '/autodrive/roboracer_1/lidar')
        self.declare_parameter('odometry_topic', '/odometry/filtered')

        # ------------------------------------------------------------------
        # Parameters that existed in your YAML but were previously ignored
        # ------------------------------------------------------------------
        self.declare_parameter('odom_axis_mode', 'normal')
        self.declare_parameter('use_initial_pose', True)
        self.declare_parameter('initial_pose_x', 0.7994)
        self.declare_parameter('initial_pose_y', 3.1583)
        self.declare_parameter('initial_pose_yaw', -1.5708)
        self.declare_parameter('max_odom_delta', 0.50)
        self.declare_parameter('max_odom_dtheta', 0.70)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'roboracer_1')
        self.declare_parameter('lidar_frame', 'lidar')
        self.declare_parameter('lidar_x_offset', 0.2733)
        self.declare_parameter('lidar_y_offset', 0.0)
        self.declare_parameter('lidar_yaw_offset', 0.0)

        # ------------------------------------------------------------------
        # New parameters for robust initialization / resampling / motion noise
        # ------------------------------------------------------------------
        self.declare_parameter('initial_std_x', 0.02)
        self.declare_parameter('initial_std_y', 0.02)
        self.declare_parameter('initial_std_yaw', math.radians(1.0))
        self.declare_parameter('initialpose_use_message_covariance', False)
        self.declare_parameter('ess_threshold_ratio', 0.50)
        self.declare_parameter('publish_pf_tf', False)

        # motion_dispersion_* are now SCALE factors, not fixed noise per update.
        self.declare_parameter('motion_noise_floor_x', 0.0005)
        self.declare_parameter('motion_noise_floor_y', 0.0005)
        self.declare_parameter('motion_noise_floor_theta', 0.0010)
        self.declare_parameter('motion_noise_theta_distance_scale', 0.01)

        # ------------------------------------------------------------------
        # High-speed synchronization / LiDAR deskew
        # ------------------------------------------------------------------
        # LaserScan.header.stamp is treated according to the ROS LaserScan
        # convention: acquisition time of the first ray.  The scan is
        # referenced to its temporal midpoint for the PF update.
        self.declare_parameter('enable_lidar_deskew', True)
        self.declare_parameter('odom_buffer_seconds', 3.0)
        self.declare_parameter('max_pending_scans', 6)

        # A fixed max_odom_delta=0.5 m can reject valid high-speed motion at
        # 20 Hz.  Keep it as a jump floor, but additionally validate motion by
        # implied speed/yaw-rate.  Only motion exceeding BOTH limits is
        # rejected as a reset/glitch.
        self.declare_parameter('max_odom_speed', 20.0)
        self.declare_parameter('max_odom_yaw_rate', 20.0)

        # ------------------------------------------------------------------
        # Read parameters
        # ------------------------------------------------------------------
        self.ANGLE_STEP = int(self.get_parameter('angle_step').value)
        self.MAX_PARTICLES = int(self.get_parameter('max_particles').value)
        self.MAX_VIZ_PARTICLES = int(self.get_parameter('max_viz_particles').value)
        self.INV_SQUASH_FACTOR = 1.0 / float(self.get_parameter('squash_factor').value)
        self.MAX_RANGE_METERS = float(self.get_parameter('max_range').value)
        self.THETA_DISCRETIZATION = int(self.get_parameter('theta_discretization').value)
        self.WHICH_RM = str(self.get_parameter('range_method').value)
        self.RANGELIB_VAR = int(self.get_parameter('rangelib_variant').value)
        self.SHOW_FINE_TIMING = bool(self.get_parameter('fine_timing').value)
        self.PUBLISH_ODOM = bool(self.get_parameter('publish_odom').value)
        self.DO_VIZ = bool(self.get_parameter('viz').value)

        self.Z_SHORT = float(self.get_parameter('z_short').value)
        self.Z_MAX = float(self.get_parameter('z_max').value)
        self.Z_RAND = float(self.get_parameter('z_rand').value)
        self.Z_HIT = float(self.get_parameter('z_hit').value)
        self.SIGMA_HIT = float(self.get_parameter('sigma_hit').value)

        # These are interpreted as uncertainty growth coefficients.
        self.MOTION_DISPERSION_X = float(self.get_parameter('motion_dispersion_x').value)
        self.MOTION_DISPERSION_Y = float(self.get_parameter('motion_dispersion_y').value)
        self.MOTION_DISPERSION_THETA = float(self.get_parameter('motion_dispersion_theta').value)
        self.MOTION_NOISE_FLOOR_X = float(self.get_parameter('motion_noise_floor_x').value)
        self.MOTION_NOISE_FLOOR_Y = float(self.get_parameter('motion_noise_floor_y').value)
        self.MOTION_NOISE_FLOOR_THETA = float(self.get_parameter('motion_noise_floor_theta').value)
        self.MOTION_NOISE_THETA_DISTANCE_SCALE = float(
            self.get_parameter('motion_noise_theta_distance_scale').value
        )

        self.ENABLE_LIDAR_DESKEW = bool(
            self.get_parameter('enable_lidar_deskew').value
        )
        self.ODOM_BUFFER_SECONDS = float(
            self.get_parameter('odom_buffer_seconds').value
        )
        self.MAX_PENDING_SCANS = int(
            self.get_parameter('max_pending_scans').value
        )
        self.MAX_ODOM_SPEED = float(
            self.get_parameter('max_odom_speed').value
        )
        self.MAX_ODOM_YAW_RATE = float(
            self.get_parameter('max_odom_yaw_rate').value
        )

        self.ODOM_AXIS_MODE = str(self.get_parameter('odom_axis_mode').value)
        self.USE_INITIAL_POSE = bool(self.get_parameter('use_initial_pose').value)
        self.INITIAL_POSE_X = float(self.get_parameter('initial_pose_x').value)
        self.INITIAL_POSE_Y = float(self.get_parameter('initial_pose_y').value)
        self.INITIAL_POSE_YAW = float(self.get_parameter('initial_pose_yaw').value)
        self.INIT_STD_X = float(self.get_parameter('initial_std_x').value)
        self.INIT_STD_Y = float(self.get_parameter('initial_std_y').value)
        self.INIT_STD_YAW = float(self.get_parameter('initial_std_yaw').value)
        self.INITPOSE_USE_MSG_COV = bool(
            self.get_parameter('initialpose_use_message_covariance').value
        )
        self.MAX_ODOM_DELTA = float(self.get_parameter('max_odom_delta').value)
        self.MAX_ODOM_DTHETA = float(self.get_parameter('max_odom_dtheta').value)
        self.ESS_THRESHOLD_RATIO = float(self.get_parameter('ess_threshold_ratio').value)

        self.MAP_FRAME = str(self.get_parameter('map_frame').value)
        self.BASE_FRAME = str(self.get_parameter('base_frame').value)
        self.LIDAR_FRAME = str(self.get_parameter('lidar_frame').value)
        self.LIDAR_X_OFFSET = float(self.get_parameter('lidar_x_offset').value)
        self.LIDAR_Y_OFFSET = float(self.get_parameter('lidar_y_offset').value)
        self.LIDAR_YAW_OFFSET = float(self.get_parameter('lidar_yaw_offset').value)
        self.PUBLISH_PF_TF = bool(self.get_parameter('publish_pf_tf').value)

        if self.ANGLE_STEP < 1:
            raise ValueError('angle_step must be >= 1')
        if self.MAX_PARTICLES < 1:
            raise ValueError('max_particles must be >= 1')
        if not 0.0 < self.ESS_THRESHOLD_RATIO <= 1.0:
            raise ValueError('ess_threshold_ratio must be in (0, 1]')
        if self.ODOM_AXIS_MODE != 'normal':
            self.get_logger().warn(
                f"odom_axis_mode='{self.ODOM_AXIS_MODE}' requested, but this corrected "
                "implementation currently supports only 'normal'. Using normal axes."
            )
        if self.ODOM_BUFFER_SECONDS <= 0.0:
            raise ValueError('odom_buffer_seconds must be > 0')
        if self.MAX_PENDING_SCANS < 1:
            raise ValueError('max_pending_scans must be >= 1')
        if self.MAX_ODOM_SPEED <= 0.0:
            raise ValueError('max_odom_speed must be > 0')
        if self.MAX_ODOM_YAW_RATE <= 0.0:
            raise ValueError('max_odom_yaw_rate must be > 0')

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.MAX_RANGE_PX = None
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False

        self.laser_angles = None
        self.selected_ray_indices = None
        self.downsampled_angles = None
        self.downsampled_ranges = None
        self.active_scan_angles = None
        self.active_scan_ranges = None
        self.last_scan_stamp = None
        self.last_processed_scan_ns = None
        self.last_processed_ref_ns = None

        # Timestamped EKF history.  Each entry is:
        # (stamp_ns, pose[x,y,yaw], vx, wz, twist_covariance)
        self.odom_buffer = deque()
        self.pending_scans = deque()

        # Latest absolute local-odometry pose is kept for diagnostics only.
        # PF motion itself is computed from odometry interpolated to LiDAR
        # reference timestamps.
        self.latest_odom_pose = None
        self.last_update_odom_pose = None
        self.last_update_odom_ns = None
        self.current_speed = 0.0
        self.current_wz = 0.0
        self.current_twist_covariance = [0.0] * 36

        self.range_method = None
        self.sensor_model_table = None
        self.first_sensor_update = True
        self.state_lock = Lock()

        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float64)
        self.weights = np.ones(self.MAX_PARTICLES, dtype=np.float64) / float(self.MAX_PARTICLES)
        self.sensor_likelihoods = np.ones(self.MAX_PARTICLES, dtype=np.float64)
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float64)
        self.sensor_poses = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)

        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.viz_queries = None
        self.viz_ranges = None

        self.inferred_pose = None
        self.inferred_covariance = np.zeros((3, 3), dtype=np.float64)
        self.last_neff = float(self.MAX_PARTICLES)
        self.last_resampled = False

        self.iters = 0
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)

        # ------------------------------------------------------------------
        # Map + RangeLib
        # ------------------------------------------------------------------
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_omap()
        self.precompute_sensor_model()

        if self.USE_INITIAL_POSE:
            self.initialize_particles_values(
                self.INITIAL_POSE_X,
                self.INITIAL_POSE_Y,
                self.INITIAL_POSE_YAW,
                self.INIT_STD_X,
                self.INIT_STD_Y,
                self.INIT_STD_YAW,
            )
            self.get_logger().info(
                'Initialized from configured pose: '
                f'x={self.INITIAL_POSE_X:.4f}, y={self.INITIAL_POSE_Y:.4f}, '
                f'yaw={self.INITIAL_POSE_YAW:.4f}'
            )
        else:
            self.initialize_global()

        # ------------------------------------------------------------------
        # Publishers / subscribers
        # ------------------------------------------------------------------
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, '/pf/viz/poly1', 1)

        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, '/pf/pose/odom', 1)

        self.pub_tf = TransformBroadcaster(self)

        self.laser_sub = self.create_subscription(
            LaserScan,
            str(self.get_parameter('scan_topic').value),
            self.lidarCB,
            1,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter('odometry_topic').value),
            self.odomCB,
            10,
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.clicked_pose,
            1,
        )
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_pose,
            1,
        )

        self.get_logger().info(
            'High-speed PF ready: synchronized EKF history, scan-midpoint motion, '
            f'LiDAR deskew={self.ENABLE_LIDAR_DESKEW}, ESS-gated resampling enabled.'
        )

    # ======================================================================
    # Map / sensor model initialization
    # ======================================================================
    def get_omap(self):
        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get map service not available, waiting...')

        valid_map_received = False
        map_msg = None

        while not valid_map_received and rclpy.ok():
            req = GetMap.Request()
            future = self.map_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                map_msg = future.result().map
                self.map_info = map_msg.info
                if self.map_info.resolution > 0.0:
                    valid_map_received = True
                    self.get_logger().info(
                        f'Valid map received with resolution: {self.map_info.resolution}'
                    )
                else:
                    self.get_logger().warn(
                        'Map server returned invalid map (resolution 0.0). Waiting...'
                    )
                    time.sleep(1.0)
            else:
                self.get_logger().error('Failed to call GetMap service, retrying...')
                time.sleep(1.0)

        if map_msg is None:
            raise RuntimeError('Failed to obtain map')

        o_map = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(o_map, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(
                o_map, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION
            )
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(o_map, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(o_map, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(
                o_map, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION
            )
        else:
            raise ValueError(f'Unknown range_method: {self.WHICH_RM}')

        self.get_logger().info('Done loading map')

        array_255 = np.array(map_msg.data).reshape(
            (map_msg.info.height, map_msg.info.width)
        )
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255 == 0] = True
        self.map_initialized = True

    def precompute_sensor_model(self):
        self.get_logger().info('Precomputing sensor model')

        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        for d in range(table_width):
            norm = 0.0
            for r in range(table_width):
                prob = 0.0
                z = float(r - d)

                prob += (
                    self.Z_HIT
                    * np.exp(-(z * z) / (2.0 * self.SIGMA_HIT * self.SIGMA_HIT))
                    / (self.SIGMA_HIT * np.sqrt(2.0 * np.pi))
                )

                if r < d and d > 0:
                    prob += 2.0 * self.Z_SHORT * (d - r) / float(d)

                if int(r) == int(self.MAX_RANGE_PX):
                    prob += self.Z_MAX

                if r < int(self.MAX_RANGE_PX):
                    prob += self.Z_RAND / float(self.MAX_RANGE_PX)

                norm += prob
                self.sensor_model_table[int(r), int(d)] = prob

            if norm > 0.0:
                self.sensor_model_table[:, int(d)] /= norm

        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    # ======================================================================
    # Geometry helpers
    # ======================================================================
    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    @staticmethod
    def _stamp_to_ns(stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _base_to_lidar_poses(self, base_poses, out):
        """Transform Nx3 map->base poses into map->lidar poses."""
        theta = base_poses[:, 2]
        c = np.cos(theta)
        s = np.sin(theta)

        out[:, 0] = (
            base_poses[:, 0]
            + self.LIDAR_X_OFFSET * c
            - self.LIDAR_Y_OFFSET * s
        )
        out[:, 1] = (
            base_poses[:, 1]
            + self.LIDAR_X_OFFSET * s
            + self.LIDAR_Y_OFFSET * c
        )
        out[:, 2] = self._wrap_angle(theta + self.LIDAR_YAW_OFFSET)
        return out

    def _single_base_to_lidar(self, pose):
        c = math.cos(pose[2])
        s = math.sin(pose[2])
        return np.array(
            [
                pose[0] + self.LIDAR_X_OFFSET * c - self.LIDAR_Y_OFFSET * s,
                pose[1] + self.LIDAR_X_OFFSET * s + self.LIDAR_Y_OFFSET * c,
                float(self._wrap_angle(pose[2] + self.LIDAR_YAW_OFFSET)),
            ],
            dtype=np.float32,
        )

    # ======================================================================
    # High-speed odometry synchronization / LiDAR deskew helpers
    # ======================================================================
    def _prune_odom_buffer_locked(self):
        if not self.odom_buffer:
            return
        newest_ns = self.odom_buffer[-1][0]
        keep_after_ns = newest_ns - int(self.ODOM_BUFFER_SECONDS * 1e9)
        while len(self.odom_buffer) > 2 and self.odom_buffer[1][0] < keep_after_ns:
            self.odom_buffer.popleft()

    @staticmethod
    def _snapshot_to_arrays(snapshot):
        times = np.asarray([e[0] for e in snapshot], dtype=np.int64)
        poses = np.asarray([e[1] for e in snapshot], dtype=np.float64)
        vxs = np.asarray([e[2] for e in snapshot], dtype=np.float64)
        wzs = np.asarray([e[3] for e in snapshot], dtype=np.float64)
        return times, poses, vxs, wzs

    def _interpolate_odom(self, times, poses, vxs, wzs, target_ns):
        """Interpolate local odometry at one or many integer nanosecond stamps.

        Returns (pose, vx, wz).  pose is Nx3 for vector input and 3-vector for
        scalar input.  Yaw interpolation follows the shortest wrapped arc.
        """
        scalar = np.isscalar(target_ns)
        targets = np.atleast_1d(np.asarray(target_ns, dtype=np.int64))

        if len(times) < 2:
            raise ValueError('Need at least two odometry samples for interpolation')
        if targets[0] < times[0] or targets[-1] > times[-1]:
            raise ValueError('Requested timestamp is outside odometry history')

        hi = np.searchsorted(times, targets, side='right')
        hi = np.clip(hi, 1, len(times) - 1)
        lo = hi - 1

        # Subtract int64 nanoseconds before converting to float.  This avoids
        # precision loss from converting ~1e18 absolute timestamps directly.
        denom = (times[hi] - times[lo]).astype(np.float64)
        numer = (targets - times[lo]).astype(np.float64)
        alpha = np.zeros_like(denom)
        valid = denom > 0.0
        alpha[valid] = numer[valid] / denom[valid]
        alpha = np.clip(alpha, 0.0, 1.0)

        out = np.empty((len(targets), 3), dtype=np.float64)
        out[:, 0] = poses[lo, 0] + alpha * (poses[hi, 0] - poses[lo, 0])
        out[:, 1] = poses[lo, 1] + alpha * (poses[hi, 1] - poses[lo, 1])

        dyaw = self._wrap_angle(poses[hi, 2] - poses[lo, 2])
        out[:, 2] = self._wrap_angle(poses[lo, 2] + alpha * dyaw)

        vx = vxs[lo] + alpha * (vxs[hi] - vxs[lo])
        wz = wzs[lo] + alpha * (wzs[hi] - wzs[lo])

        if scalar:
            return out[0], float(vx[0]), float(wz[0])
        return out, vx, wz

    def _nearest_twist_covariance(self, snapshot, target_ns):
        times = np.asarray([e[0] for e in snapshot], dtype=np.int64)
        idx = int(np.searchsorted(times, int(target_ns), side='left'))
        if idx <= 0:
            return list(snapshot[0][4])
        if idx >= len(snapshot):
            return list(snapshot[-1][4])
        if abs(int(target_ns) - int(times[idx - 1])) <= abs(int(times[idx]) - int(target_ns)):
            return list(snapshot[idx - 1][4])
        return list(snapshot[idx][4])

    def _deskew_scan(self, scan, times, poses, vxs, wzs):
        """Deskew selected LiDAR beams into the scan-midpoint LiDAR frame.

        The ROS LaserScan convention is used: header.stamp is the acquisition
        time of the first ray.  Finite hit endpoints are transformed using the
        timestamped EKF pose for each beam.  Max/no-hit beams preserve the
        max-range observation while their direction is rotation-deskewed.
        """
        beam_poses, _, _ = self._interpolate_odom(
            times, poses, vxs, wzs, scan['beam_times_ns']
        )
        ref_pose, ref_vx, ref_wz = self._interpolate_odom(
            times, poses, vxs, wzs, scan['ref_ns']
        )

        raw_ranges = scan['ranges'].astype(np.float64, copy=True)
        raw_angles = scan['angles'].astype(np.float64, copy=False)
        max_mask = scan['max_mask']

        # map/odom -> LiDAR pose for every beam time
        theta_i = beam_poses[:, 2]
        c_i = np.cos(theta_i)
        s_i = np.sin(theta_i)
        lidar_x_i = (
            beam_poses[:, 0]
            + self.LIDAR_X_OFFSET * c_i
            - self.LIDAR_Y_OFFSET * s_i
        )
        lidar_y_i = (
            beam_poses[:, 1]
            + self.LIDAR_X_OFFSET * s_i
            + self.LIDAR_Y_OFFSET * c_i
        )
        lidar_yaw_i = self._wrap_angle(theta_i + self.LIDAR_YAW_OFFSET)

        # Reference LiDAR pose at temporal midpoint of the scan.
        c_ref_b = math.cos(ref_pose[2])
        s_ref_b = math.sin(ref_pose[2])
        ref_lidar_x = (
            ref_pose[0]
            + self.LIDAR_X_OFFSET * c_ref_b
            - self.LIDAR_Y_OFFSET * s_ref_b
        )
        ref_lidar_y = (
            ref_pose[1]
            + self.LIDAR_X_OFFSET * s_ref_b
            + self.LIDAR_Y_OFFSET * c_ref_b
        )
        ref_lidar_yaw = float(self._wrap_angle(ref_pose[2] + self.LIDAR_YAW_OFFSET))

        corrected_ranges = raw_ranges.copy()
        corrected_angles = self._wrap_angle(
            lidar_yaw_i + raw_angles - ref_lidar_yaw
        ).astype(np.float64)

        # Finite hit beams have a real endpoint, so compensate both rotation
        # and translation between each beam time and the scan midpoint.
        finite_hit = ~max_mask
        if np.any(finite_hit):
            world_angle = lidar_yaw_i[finite_hit] + raw_angles[finite_hit]
            hit_x = lidar_x_i[finite_hit] + raw_ranges[finite_hit] * np.cos(world_angle)
            hit_y = lidar_y_i[finite_hit] + raw_ranges[finite_hit] * np.sin(world_angle)

            dx = hit_x - ref_lidar_x
            dy = hit_y - ref_lidar_y
            c_ref = math.cos(ref_lidar_yaw)
            s_ref = math.sin(ref_lidar_yaw)
            px_ref = c_ref * dx + s_ref * dy
            py_ref = -s_ref * dx + c_ref * dy

            corrected_ranges[finite_hit] = np.hypot(px_ref, py_ref)
            corrected_angles[finite_hit] = np.arctan2(py_ref, px_ref)

        # No-hit/max-range observations do not have a physical endpoint.  Keep
        # their range in the sensor model's max-range bucket; only deskew the
        # ray direction using vehicle rotation.
        corrected_ranges[max_mask] = self.MAX_RANGE_METERS

        np.nan_to_num(
            corrected_ranges,
            copy=False,
            nan=self.MAX_RANGE_METERS,
            posinf=self.MAX_RANGE_METERS,
            neginf=0.0,
        )
        np.clip(corrected_ranges, 0.0, self.MAX_RANGE_METERS, out=corrected_ranges)

        return (
            corrected_ranges.astype(np.float32),
            corrected_angles.astype(np.float32),
            ref_pose,
            ref_vx,
            ref_wz,
        )

    def _try_process_pending_scans(self):
        """Process queued scans only after EKF history covers the full scan."""
        while True:
            with self.state_lock:
                if not self.pending_scans or len(self.odom_buffer) < 2:
                    return

                scan = self.pending_scans[0]
                oldest_ns = self.odom_buffer[0][0]
                newest_ns = self.odom_buffer[-1][0]

                # Startup/history gap: this scan can never be deskewed now.
                if scan['start_ns'] < oldest_ns:
                    self.pending_scans.popleft()
                    self.get_logger().warn(
                        'Dropping LiDAR scan: odometry history starts after scan start.'
                    )
                    continue

                # Wait for a future EKF callback to cover the last ray.
                if scan['end_ns'] > newest_ns:
                    return

                scan = self.pending_scans.popleft()
                snapshot = list(self.odom_buffer)

            times, poses, vxs, wzs = self._snapshot_to_arrays(snapshot)

            try:
                if self.ENABLE_LIDAR_DESKEW:
                    ranges, angles, ref_pose, ref_vx, ref_wz = self._deskew_scan(
                        scan, times, poses, vxs, wzs
                    )
                else:
                    ref_pose, ref_vx, ref_wz = self._interpolate_odom(
                        times, poses, vxs, wzs, scan['ref_ns']
                    )
                    ranges = scan['ranges'].copy()
                    angles = scan['angles'].copy()

                twist_cov = self._nearest_twist_covariance(snapshot, scan['ref_ns'])
            except ValueError as exc:
                self.get_logger().warn(f'Dropping LiDAR scan: {exc}')
                continue

            self.update_from_scan(
                scan=scan,
                ref_pose=ref_pose,
                observation=ranges,
                scan_angles=angles,
                ref_vx=ref_vx,
                ref_wz=ref_wz,
                twist_covariance=twist_cov,
            )

    # ======================================================================
    # ROS callbacks
    # ======================================================================
    def odomCB(self, msg):
        """Store timestamped EKF odometry and release scans when covered."""
        stamp_ns = self._stamp_to_ns(msg.header.stamp)
        pose = np.array(
            [
                float(msg.pose.pose.position.x),
                float(msg.pose.pose.position.y),
                float(Utils.quaternion_to_angle(msg.pose.pose.orientation)),
            ],
            dtype=np.float64,
        )
        vx = float(msg.twist.twist.linear.x)
        wz = float(msg.twist.twist.angular.z)
        cov = list(msg.twist.covariance)

        with self.state_lock:
            # Handle simulator reset/time jump safely.
            if self.odom_buffer and stamp_ns < self.odom_buffer[-1][0]:
                self.get_logger().warn(
                    'Odometry timestamp moved backwards; clearing synchronization buffers.'
                )
                self.odom_buffer.clear()
                self.pending_scans.clear()
                self.last_update_odom_pose = None
                self.last_update_odom_ns = None

            if self.odom_buffer and stamp_ns == self.odom_buffer[-1][0]:
                self.odom_buffer.pop()

            self.odom_buffer.append((stamp_ns, pose, vx, wz, cov))
            self._prune_odom_buffer_locked()

            self.latest_odom_pose = pose
            self.current_speed = vx
            self.current_wz = wz
            self.current_twist_covariance = cov
            self.odom_initialized = True

        self._try_process_pending_scans()

    def lidarCB(self, msg):
        """Queue each physical LiDAR scan for timestamp-aligned processing."""
        num_measurements = len(msg.ranges)
        if num_measurements == 0:
            return

        if (
            not isinstance(self.selected_ray_indices, np.ndarray)
            or self.laser_angles is None
            or len(self.laser_angles) != num_measurements
        ):
            self.get_logger().info('...Received first/new-shape LiDAR message')
            self.selected_ray_indices = np.arange(
                0, num_measurements, self.ANGLE_STEP, dtype=np.int64
            )
            self.laser_angles = (
                float(msg.angle_min)
                + np.arange(num_measurements, dtype=np.float64)
                * float(msg.angle_increment)
            )
            self.downsampled_angles = self.laser_angles[
                self.selected_ray_indices
            ].astype(np.float32)
            self.viz_queries = np.zeros(
                (len(self.selected_ray_indices), 3), dtype=np.float32
            )
            self.viz_ranges = np.zeros(
                len(self.selected_ray_indices), dtype=np.float32
            )
            self.first_sensor_update = True
            self.get_logger().info(
                f'LiDAR beams={num_measurements}, using '
                f'{len(self.selected_ray_indices)} rays/update, '
                f'deskew={self.ENABLE_LIDAR_DESKEW}'
            )

        start_ns = self._stamp_to_ns(msg.header.stamp)
        if self.last_processed_scan_ns == start_ns:
            return

        # LaserScan.header.stamp is the first-ray acquisition time.  Prefer
        # time_increment because it gives the exact time of each ray.
        if msg.time_increment > 0.0:
            dt_ray_ns = int(round(float(msg.time_increment) * 1e9))
            full_offsets_ns = np.arange(num_measurements, dtype=np.int64) * dt_ray_ns
            end_ns = start_ns + int(full_offsets_ns[-1])
            beam_times_ns = start_ns + full_offsets_ns[self.selected_ray_indices]
        else:
            scan_duration_ns = int(round(max(float(msg.scan_time), 0.0) * 1e9))
            end_ns = start_ns + scan_duration_ns
            if num_measurements > 1:
                fractions = self.selected_ray_indices.astype(np.float64) / float(num_measurements - 1)
                beam_times_ns = start_ns + np.rint(fractions * scan_duration_ns).astype(np.int64)
            else:
                beam_times_ns = np.array([start_ns], dtype=np.int64)

        ref_ns = start_ns + (end_ns - start_ns) // 2

        raw = np.asarray(msg.ranges, dtype=np.float64)[self.selected_ray_indices]
        range_max = float(msg.range_max) if msg.range_max > 0.0 else self.MAX_RANGE_METERS
        max_mask = (~np.isfinite(raw)) | (raw >= range_max - 1e-6)

        ranges = raw.copy()
        ranges[max_mask] = self.MAX_RANGE_METERS
        np.nan_to_num(
            ranges,
            copy=False,
            nan=self.MAX_RANGE_METERS,
            posinf=self.MAX_RANGE_METERS,
            neginf=0.0,
        )
        np.clip(ranges, 0.0, self.MAX_RANGE_METERS, out=ranges)

        scan = {
            'start_ns': int(start_ns),
            'end_ns': int(end_ns),
            'ref_ns': int(ref_ns),
            'beam_times_ns': np.asarray(beam_times_ns, dtype=np.int64),
            'ranges': ranges.astype(np.float32),
            'angles': self.downsampled_angles.copy(),
            'max_mask': np.asarray(max_mask, dtype=bool),
            'scan_time': float(msg.scan_time),
        }

        with self.state_lock:
            if len(self.pending_scans) >= self.MAX_PENDING_SCANS:
                dropped = self.pending_scans.popleft()
                self.get_logger().warn(
                    'Pending LiDAR queue full; dropping oldest scan '
                    f"at {dropped['start_ns']} ns"
                )
            self.pending_scans.append(scan)
            self.lidar_initialized = True

        self._try_process_pending_scans()

    def clicked_pose(self, msg):
        if isinstance(msg, PointStamped):
            self.initialize_global()
            return

        if not isinstance(msg, PoseWithCovarianceStamped):
            return

        yaw = float(Utils.quaternion_to_angle(msg.pose.pose.orientation))
        std_x = self.INIT_STD_X
        std_y = self.INIT_STD_Y
        std_yaw = self.INIT_STD_YAW

        if self.INITPOSE_USE_MSG_COV:
            cov = msg.pose.covariance
            if cov[0] > 0.0:
                std_x = math.sqrt(cov[0])
            if cov[7] > 0.0:
                std_y = math.sqrt(cov[7])
            if cov[35] > 0.0:
                std_yaw = math.sqrt(cov[35])

        self.initialize_particles_values(
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            yaw,
            std_x,
            std_y,
            std_yaw,
        )

        # Re-anchor relative odometry at the next scan-center timestamp.
        with self.state_lock:
            self.last_update_odom_pose = None
            self.last_update_odom_ns = None

    # ======================================================================
    # Initialization
    # ======================================================================
    def initialize_particles_values(self, x, y, yaw, std_x, std_y, std_yaw):
        self.get_logger().info(
            'SETTING POSE '
            f'x={x:.4f}, y={y:.4f}, yaw={yaw:.4f}, '
            f'std=[{std_x:.4f}, {std_y:.4f}, {std_yaw:.4f}]'
        )

        with self.state_lock:
            self.weights.fill(1.0 / float(self.MAX_PARTICLES))
            self.particles[:, 0] = x + np.random.normal(
                0.0, std_x, self.MAX_PARTICLES
            )
            self.particles[:, 1] = y + np.random.normal(
                0.0, std_y, self.MAX_PARTICLES
            )
            self.particles[:, 2] = yaw + np.random.normal(
                0.0, std_yaw, self.MAX_PARTICLES
            )
            self.particles[:, 2] = self._wrap_angle(self.particles[:, 2])
            self.inferred_pose = self.expected_pose()
            self.inferred_covariance = self.weighted_pose_covariance(
                self.inferred_pose
            )
            self.last_neff = float(self.MAX_PARTICLES)
            self.last_resampled = False

    def initialize_global(self):
        self.get_logger().info('GLOBAL INITIALIZATION')
        with self.state_lock:
            permissible_y, permissible_x = np.where(self.permissible_region == 1)
            indices = np.random.randint(
                0, len(permissible_x), size=self.MAX_PARTICLES
            )

            permissible_states = np.zeros((self.MAX_PARTICLES, 3))
            permissible_states[:, 0] = permissible_x[indices]
            permissible_states[:, 1] = permissible_y[indices]
            permissible_states[:, 2] = (
                np.random.random(self.MAX_PARTICLES) * np.pi * 2.0
            )

            Utils.map_to_world(permissible_states, self.map_info)
            permissible_states[:, 2] = self._wrap_angle(
                permissible_states[:, 2]
            )
            self.particles = permissible_states
            self.weights.fill(1.0 / self.MAX_PARTICLES)
            self.inferred_pose = self.expected_pose()
            self.inferred_covariance = self.weighted_pose_covariance(
                self.inferred_pose
            )
            self.last_neff = float(self.MAX_PARTICLES)
            self.last_resampled = False
            self.last_update_odom_pose = None
            self.last_update_odom_ns = None

    # ======================================================================
    # Motion model
    # ======================================================================
    def compute_odom_action(self, current_pose, current_ns):
        """Return [dx_local, dy_local, dtheta] between synchronized scan centers."""
        current_pose = np.asarray(current_pose, dtype=np.float64)
        current_ns = int(current_ns)

        if self.last_update_odom_pose is None or self.last_update_odom_ns is None:
            self.last_update_odom_pose = current_pose.copy()
            self.last_update_odom_ns = current_ns
            return np.zeros(3, dtype=np.float64)

        previous = self.last_update_odom_pose
        previous_ns = self.last_update_odom_ns
        dt = (current_ns - previous_ns) * 1e-9

        if dt <= 0.0:
            self.last_update_odom_pose = current_pose.copy()
            self.last_update_odom_ns = current_ns
            return np.zeros(3, dtype=np.float64)

        dx_world = current_pose[0] - previous[0]
        dy_world = current_pose[1] - previous[1]
        dtheta = float(self._wrap_angle(current_pose[2] - previous[2]))
        distance = math.hypot(dx_world, dy_world)

        implied_speed = distance / dt
        implied_yaw_rate = abs(dtheta) / dt

        # Do not reject legitimate 0.5+ m inter-scan motion at maximum speed.
        # Reject only when both the absolute jump and implied rate are
        # physically implausible, which is characteristic of reset/glitch.
        bad_translation = (
            distance > self.MAX_ODOM_DELTA
            and implied_speed > self.MAX_ODOM_SPEED
        )
        bad_rotation = (
            abs(dtheta) > self.MAX_ODOM_DTHETA
            and implied_yaw_rate > self.MAX_ODOM_YAW_RATE
        )
        if bad_translation or bad_rotation:
            self.get_logger().warn(
                'Rejected odometry discontinuity: '
                f'dt={dt:.3f}s, distance={distance:.3f}m '
                f'({implied_speed:.2f}m/s), dtheta={dtheta:.3f}rad '
                f'({implied_yaw_rate:.2f}rad/s)'
            )
            self.last_update_odom_pose = current_pose.copy()
            self.last_update_odom_ns = current_ns
            return np.zeros(3, dtype=np.float64)

        c = math.cos(previous[2])
        sin_prev = math.sin(previous[2])
        dx_local = c * dx_world + sin_prev * dy_world
        dy_local = -sin_prev * dx_world + c * dy_world

        self.last_update_odom_pose = current_pose.copy()
        self.last_update_odom_ns = current_ns
        return np.array([dx_local, dy_local, dtheta], dtype=np.float64)

    def motion_model(self, proposal_dist, action):
        """Apply relative odometry plus motion-scaled noise in vehicle coordinates."""
        dx_local, dy_local, dtheta = action
        distance = math.hypot(dx_local, dy_local)

        theta = proposal_dist[:, 2]
        c = np.cos(theta)
        s = np.sin(theta)

        # Deterministic motion, transformed from local vehicle coordinates into map.
        self.local_deltas[:, 0] = c * dx_local - s * dy_local
        self.local_deltas[:, 1] = s * dx_local + c * dy_local
        self.local_deltas[:, 2] = dtheta
        proposal_dist[:, :] += self.local_deltas

        # Existing motion_dispersion_* parameters now scale with real motion.
        sigma_long = self.MOTION_NOISE_FLOOR_X + self.MOTION_DISPERSION_X * distance
        sigma_lat = self.MOTION_NOISE_FLOOR_Y + self.MOTION_DISPERSION_Y * distance
        sigma_yaw = (
            self.MOTION_NOISE_FLOOR_THETA
            + self.MOTION_DISPERSION_THETA * abs(dtheta)
            + self.MOTION_NOISE_THETA_DISTANCE_SCALE * distance
        )

        # Sample noise in each particle's local frame, then rotate to map frame.
        noise_long = np.random.normal(0.0, sigma_long, self.MAX_PARTICLES)
        noise_lat = np.random.normal(0.0, sigma_lat, self.MAX_PARTICLES)

        proposal_dist[:, 0] += c * noise_long - s * noise_lat
        proposal_dist[:, 1] += s * noise_long + c * noise_lat
        proposal_dist[:, 2] += np.random.normal(
            0.0, sigma_yaw, self.MAX_PARTICLES
        )
        proposal_dist[:, 2] = self._wrap_angle(proposal_dist[:, 2])

    # ======================================================================
    # Sensor model
    # ======================================================================
    def sensor_model(self, base_particles, obs, scan_angles, likelihoods):
        """Compute p(z|x) for base-frame particles using LiDAR-frame ray casting."""
        num_rays = scan_angles.shape[0]

        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros(
                    (num_rays * self.MAX_PARTICLES, 3), dtype=np.float32
                )
            else:
                self.queries = np.zeros(
                    (self.MAX_PARTICLES, 3), dtype=np.float32
                )

            self.ranges = np.zeros(
                num_rays * self.MAX_PARTICLES, dtype=np.float32
            )
            self.tiled_angles = np.empty(
                num_rays * self.MAX_PARTICLES, dtype=np.float32
            )
            self.first_sensor_update = False

        # Deskew changes ray angles every scan at high speed, so update the
        # tiled angle buffer every measurement cycle.
        if self.tiled_angles is not None:
            self.tiled_angles[:] = np.tile(scan_angles, self.MAX_PARTICLES)

        self._base_to_lidar_poses(base_particles, self.sensor_poses)
        likelihoods.fill(1.0)

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if 'cddt' not in self.WHICH_RM:
                raise RuntimeError(
                    'Radial CDDT optimization requires cddt/pcddt range method'
                )
            self.queries[:, :] = self.sensor_poses[:, :]
            if self.ENABLE_LIDAR_DESKEW:
                # Deskewed angles are not guaranteed to be perfectly uniform,
                # so the radial optimization's uniform-angle assumption is not
                # safe.  Fall back to the generic repeat-angle caster.
                self.range_method.calc_range_repeat_angles(
                    self.queries, scan_angles, self.ranges
                )
            else:
                self.range_method.calc_range_many_radial_optimized(
                    num_rays,
                    scan_angles[0],
                    scan_angles[-1],
                    self.queries,
                    self.ranges,
                )
            self.range_method.eval_sensor_model(
                obs, self.ranges, likelihoods, num_rays, self.MAX_PARTICLES
            )
            np.power(likelihoods, self.INV_SQUASH_FACTOR, out=likelihoods)

        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:, :] = self.sensor_poses[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries,
                scan_angles,
                obs,
                likelihoods,
            )
            np.power(likelihoods, self.INV_SQUASH_FACTOR, out=likelihoods)

        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            t_start = time.time() if self.SHOW_FINE_TIMING else None
            self.queries[:, :] = self.sensor_poses[:, :]
            t_init = time.time() if self.SHOW_FINE_TIMING else None

            self.range_method.calc_range_repeat_angles(
                self.queries, scan_angles, self.ranges
            )
            t_range = time.time() if self.SHOW_FINE_TIMING else None

            self.range_method.eval_sensor_model(
                obs, self.ranges, likelihoods, num_rays, self.MAX_PARTICLES
            )
            t_eval = time.time() if self.SHOW_FINE_TIMING else None

            np.power(likelihoods, self.INV_SQUASH_FACTOR, out=likelihoods)
            t_squash = time.time() if self.SHOW_FINE_TIMING else None

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                total = max(t_squash - t_start, 1e-9)
                self.get_logger().info(
                    str(
                        [
                            'sensor_model fractions:',
                            'init', round((t_init - t_start) / total, 2),
                            'range', round((t_range - t_init) / total, 2),
                            'eval', round((t_eval - t_range) / total, 2),
                            'squash', round((t_squash - t_eval) / total, 2),
                        ]
                    )
                )

        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            self.queries[:, 0] = np.repeat(self.sensor_poses[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(self.sensor_poses[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(self.sensor_poses[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)
            self.range_method.eval_sensor_model(
                obs, self.ranges, likelihoods, num_rays, self.MAX_PARTICLES
            )
            np.power(likelihoods, self.INV_SQUASH_FACTOR, out=likelihoods)

        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            self.queries[:, 0] = np.repeat(self.sensor_poses[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(self.sensor_poses[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(self.sensor_poses[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            obs_px = np.copy(obs) / float(self.map_info.resolution)
            ranges_px = self.ranges / float(self.map_info.resolution)
            obs_px[obs_px > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges_px[ranges_px > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs_px).astype(np.uint16)
            intrng = np.rint(ranges_px).astype(np.uint16)

            for i in range(self.MAX_PARTICLES):
                start = i * num_rays
                stop = (i + 1) * num_rays
                weight = np.prod(
                    self.sensor_model_table[intobs, intrng[start:stop]]
                )
                likelihoods[i] = np.power(
                    weight, self.INV_SQUASH_FACTOR
                )
        else:
            raise ValueError('rangelib_variant must be 0..4')

        # Avoid NaN/Inf/zero collapse before Bayesian multiplication.
        np.nan_to_num(likelihoods, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(likelihoods, 1e-300, out=likelihoods)

    # ======================================================================
    # Resampling / statistics
    # ======================================================================
    def effective_sample_size(self):
        denom = float(np.sum(np.square(self.weights)))
        if denom <= 0.0 or not np.isfinite(denom):
            return 0.0
        return 1.0 / denom

    def systematic_resample(self):
        n = self.MAX_PARTICLES
        positions = (np.arange(n) + np.random.random()) / float(n)
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions, side='left')
        self.particles = self.particles[indices, :].copy()
        self.weights.fill(1.0 / float(n))

    def expected_pose(self):
        x = float(np.sum(self.weights * self.particles[:, 0]))
        y = float(np.sum(self.weights * self.particles[:, 1]))
        yaw = math.atan2(
            float(np.sum(self.weights * np.sin(self.particles[:, 2]))),
            float(np.sum(self.weights * np.cos(self.particles[:, 2]))),
        )
        return np.array([x, y, yaw], dtype=np.float64)

    def weighted_pose_covariance(self, mean_pose):
        dx = self.particles[:, 0] - mean_pose[0]
        dy = self.particles[:, 1] - mean_pose[1]
        dyaw = self._wrap_angle(self.particles[:, 2] - mean_pose[2])
        residuals = np.column_stack((dx, dy, dyaw))
        return (residuals * self.weights[:, None]).T @ residuals

    # ======================================================================
    # Main PF update -- one timestamp-aligned update per physical LiDAR scan
    # ======================================================================
    def update_from_scan(
        self,
        scan,
        ref_pose,
        observation,
        scan_angles,
        ref_vx,
        ref_wz,
        twist_covariance,
    ):
        if not (
            self.lidar_initialized
            and self.odom_initialized
            and self.map_initialized
        ):
            return

        ref_ns = int(scan['ref_ns'])
        ref_stamp = RosTime(nanoseconds=ref_ns).to_msg()

        with self.state_lock:
            action = self.compute_odom_action(ref_pose, ref_ns)

            self.downsampled_ranges = np.asarray(observation, dtype=np.float32)
            self.active_scan_ranges = self.downsampled_ranges
            self.active_scan_angles = np.asarray(scan_angles, dtype=np.float32)
            self.last_processed_scan_ns = int(scan['start_ns'])
            self.last_processed_ref_ns = ref_ns
            self.last_scan_stamp = ref_stamp

            # Publish twist synchronized to the scan midpoint, not callback time.
            self.current_speed = float(ref_vx)
            self.current_wz = float(ref_wz)
            self.current_twist_covariance = list(twist_covariance)

            self.timer.tick()
            self.iters += 1
            t1 = time.time()

            # 1) Motion prediction from EKF pose at previous/current scan centers.
            self.motion_model(self.particles, action)

            # 2) One measurement likelihood for this physical scan.  The rays
            # have already been deskewed into the current scan-center frame.
            self.sensor_model(
                self.particles,
                self.active_scan_ranges,
                self.active_scan_angles,
                self.sensor_likelihoods,
            )

            # 3) Bayesian update.
            self.weights *= self.sensor_likelihoods
            total_weight = float(np.sum(self.weights))
            if total_weight <= 0.0 or not np.isfinite(total_weight):
                self.get_logger().warn(
                    'Particle weights collapsed; resetting to uniform weights.'
                )
                self.weights.fill(1.0 / float(self.MAX_PARTICLES))
            else:
                self.weights /= total_weight

            # 4) State estimate before optional resampling.
            self.inferred_pose = self.expected_pose()
            self.inferred_covariance = self.weighted_pose_covariance(
                self.inferred_pose
            )
            self.last_neff = self.effective_sample_size()

            # 5) ESS-gated low-variance resampling.
            threshold = self.ESS_THRESHOLD_RATIO * self.MAX_PARTICLES
            self.last_resampled = self.last_neff < threshold
            if self.last_resampled:
                self.systematic_resample()

            t2 = time.time()

        self.publish_pose(self.inferred_pose, ref_stamp)
        self.visualize()

        ips = 1.0 / max(t2 - t1, 1e-9)
        self.smoothing.append(ips)
        if self.iters % 10 == 0:
            scan_span_ms = (scan['end_ns'] - scan['start_ns']) * 1e-6
            self.get_logger().info(
                'PF-HS: '
                f'{self.timer.fps():.1f} updates/s, '
                f'compute_capacity={self.smoothing.mean():.1f}/s, '
                f'N_eff={self.last_neff:.1f}/{self.MAX_PARTICLES}, '
                f'resampled={self.last_resampled}, '
                f'deskew={self.ENABLE_LIDAR_DESKEW}, '
                f'scan_span={scan_span_ms:.2f}ms, '
                f'vx={self.current_speed:.2f}m/s, wz={self.current_wz:.2f}rad/s'
            )

    # ======================================================================
    # Publishing
    # ======================================================================
    def publish_pose(self, pose, stamp=None):
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        if self.PUBLISH_PF_TF:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.MAP_FRAME
            t.child_frame_id = 'pf_base'
            t.transform.translation.x = float(pose[0])
            t.transform.translation.y = float(pose[1])
            t.transform.translation.z = 0.0
            q = tf_transformations.quaternion_from_euler(0.0, 0.0, float(pose[2]))
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.pub_tf.sendTransform(t)

        if not self.PUBLISH_ODOM:
            return

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.MAP_FRAME
        odom.child_frame_id = self.BASE_FRAME

        odom.pose.pose.position.x = float(pose[0])
        odom.pose.pose.position.y = float(pose[1])
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Utils.angle_to_quaternion(float(pose[2]))

        # ROS PoseWithCovariance ordering is [x, y, z, roll, pitch, yaw].
        c = self.inferred_covariance
        cov36 = [0.0] * 36
        cov36[0] = float(c[0, 0])
        cov36[1] = float(c[0, 1])
        cov36[5] = float(c[0, 2])
        cov36[6] = float(c[1, 0])
        cov36[7] = float(c[1, 1])
        cov36[11] = float(c[1, 2])
        cov36[30] = float(c[2, 0])
        cov36[31] = float(c[2, 1])
        cov36[35] = float(c[2, 2])

        # z/roll/pitch are not estimated by this 2D PF.
        cov36[14] = 1e6
        cov36[21] = 1e6
        cov36[28] = 1e6
        odom.pose.covariance = cov36

        # Use the smooth local-odometry twist alongside the globally corrected pose.
        odom.twist.twist.linear.x = float(self.current_speed)
        odom.twist.twist.angular.z = float(self.current_wz)
        if len(self.current_twist_covariance) == 36:
            odom.twist.covariance = self.current_twist_covariance

        self.odom_pub.publish(odom)

    def visualize(self):
        if not self.DO_VIZ or self.inferred_pose is None:
            return

        if self.pose_pub.get_subscription_count() > 0:
            ps = PoseStamped()
            ps.header.stamp = self.last_scan_stamp or self.get_clock().now().to_msg()
            ps.header.frame_id = self.MAP_FRAME
            ps.pose.position.x = float(self.inferred_pose[0])
            ps.pose.position.y = float(self.inferred_pose[1])
            ps.pose.orientation = Utils.angle_to_quaternion(
                float(self.inferred_pose[2])
            )
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                # After possible resampling weights may be uniform; weighted selection
                # still works and keeps visualization consistent with the current cloud.
                proposal_indices = np.random.choice(
                    self.particle_indices,
                    self.MAX_VIZ_PARTICLES,
                    replace=True,
                    p=self.weights,
                )
                self.publish_particles(self.particles[proposal_indices, :])
            else:
                self.publish_particles(self.particles)

        if (
            self.pub_fake_scan.get_subscription_count() > 0
            and isinstance(self.ranges, np.ndarray)
            and isinstance(self.active_scan_angles, np.ndarray)
        ):
            lidar_pose = self._single_base_to_lidar(self.inferred_pose)
            self.viz_queries[:, 0] = lidar_pose[0]
            self.viz_queries[:, 1] = lidar_pose[1]
            self.viz_queries[:, 2] = self.active_scan_angles + lidar_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(self.active_scan_angles, self.viz_ranges)

    def publish_particles(self, particles):
        pa = PoseArray()
        pa.header.stamp = self.last_scan_stamp or self.get_clock().now().to_msg()
        pa.header.frame_id = self.MAP_FRAME
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        ls = LaserScan()
        ls.header.stamp = self.last_scan_stamp or self.get_clock().now().to_msg()
        ls.header.frame_id = 'pf_laser'
        ls.angle_min = float(np.min(angles))
        ls.angle_max = float(np.max(angles))
        if len(angles) > 1:
            ls.angle_increment = float(np.abs(angles[0] - angles[1]))
        ls.range_min = 0.0
        ls.range_max = float(np.max(ranges)) if len(ranges) else self.MAX_RANGE_METERS
        ls.ranges = ranges.tolist()
        self.pub_fake_scan.publish(ls)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    try:
        rclpy.spin(pf)
    except KeyboardInterrupt:
        pass
    finally:
        pf.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
