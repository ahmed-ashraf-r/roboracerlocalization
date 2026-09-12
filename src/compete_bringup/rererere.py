"""
"simulator"
xhost local:root

docker run --name autodrive_roboracer_sim --rm -it --entrypoint /bin/bash --network=host --ipc=host -v /tmp/.X11-unix:/tmp.X11-umix:rw --env DISPLAY --privileged autodriveecosystem/autodrive_roboracer_sim:<tag>

./AutoDRIVE\ Simulator.x86_64 -batchmode -nographics -ip 127.0.0.1 -port 4567

#DEVKIT

docker run \
  --name autodrive_roboracer_api \
  --rm -it \
  --entrypoint /bin/bash \
  --network=host \
  --ipc=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/ubuntu/roboracer_compet_ws/src/compete_bringup:/home/autodrive_devkit/src/compete_bringup \
  -v /home/ubuntu/roboracer_compet_ws/src/compete_localiztion:/home/autodrive_devkit/src/compete_localiztion \
  -v /home/ubuntu/roboracer_compet_ws/src/rf2o_laser_odometry-ros2:/home/autodrive_devkit/src/rf2o_laser_odometry-ros2 \
  -v /home/ubuntu/roboracer_compet_ws/src/particle_filter:/home/autodrive_devkit/src/particle_filter \
  -v /home/ubuntu/roboracer_compet_ws/src/range_libc:/home/autodrive_devkit/src/range_libc \
  -v /home/ubuntu/roboracer_compet_ws/src/compete_controller:/home/autodrive_devkit/src/compete_controller \
  --env DISPLAY \
  --privileged \
  roboracer_localization:latest


docker exec -it autodrive_roboracer_api bash 

ros2 launch compete_bringup odom_filterd.launch.xml 

"""
"""
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
"{header: {frame_id: 'map'},
  pose: {
    pose: {
      position: {x: 0.7994, y: 3.1583, z: 0.0},
      orientation: {x: 0.0, y: 0.0, z: -0.7071, w: 0.7071}
    },
    covariance: [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0685]
  }}"

"""