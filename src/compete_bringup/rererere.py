"""
"simulator"
xhost local:root

docker run --name autodrive_roboracer_sim --rm -it --entrypoint /bin/bash --network=host --ipc=host -v /tmp/.X11-unix:/tmp.X11-umix:rw --env DISPLAY --privileged autodriveecosystem/autodrive_roboracer_sim:<tag>

./AutoDRIVE\ Simulator.x86_64 -batchmode -nographics -ip 127.0.0.1 -port 4567

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
  --env DISPLAY \
  --privileged \
  roboracer_localization:latest

ros2 launch roboracer_bringup controller_only_pure.launch.xml 


docker exec -it autodrive_roboracer_api bash 


ros2 run roboracer_controller ackermann_motion_planning.py --ros-args -p waypoint_file:=/home/autodrive_devkit/src/roboracer_controller/maps/raceline_approved.csv

ros2 launch roboracer_controller run_RPP.launch.xml

"""