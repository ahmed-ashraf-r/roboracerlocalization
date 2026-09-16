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
  -v /home/ubuntu/roboracer_compet_ws/src/f1tenth_control:/home/autodrive_devkit/src/f1tenth_control \
  --env DISPLAY \
  --privileged \
  roboracer_localization:latest


docker exec -it autodrive_roboracer_api bash 

ros2 launch compete_bringup final_odom.launch.xml 

ros2 run f1tenth_control pure_pursuit_node

docker run -it \
  --name autodrive_roboracer_api \
  --network=host \
  --ipc=host \
  --privileged \
  mohamedelgohary978/assiut-motorsport-roboracer:qualification

"""