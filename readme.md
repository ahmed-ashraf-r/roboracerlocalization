# AutoDRIVE Roboracer: Filtered Odometry Setup


## 1. Docker Setup & Build

First, create the Docker folder and the Dockerfile.

```bash
mkdir autodrive_roboracer_docker
cd autodrive_roboracer_docker
touch Dockerfile
```

Add the following contents to your Dockerfile to install `robot_localization` and the necessary Python dependencies for your controller script:

```dockerfile
FROM autodriveecosystem/autodrive_roboracer_api:2026-icra-explore

RUN apt-get update &&     apt-get install -y         ros-humble-robot-localization     && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for your controller script
RUN pip3 install --no-cache-dir pandas matplotlib scipy
```

Build the custom Docker image (make sure you are in the same directory as the Dockerfile). 

```bash
docker build --no-cache -t roboracer_localization .
```

## 2. Start the Simulator

Allow local X11 connections, then spin up the simulator container. 
*(Note: Replace `<tag>` with your specific simulator image tag).*

```bash
xhost local:root

docker run --name autodrive_roboracer_sim   --rm -it   --entrypoint /bin/bash   --network=host   --ipc=host   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   --env DISPLAY   --privileged   autodriveecosystem/autodrive_roboracer_sim:<tag>
```

Once inside the simulator container, launch the simulator in batch mode:

```bash
./AutoDRIVE* Simulator.x86_64 -batchmode -nographics -ip 127.0.0.1 -port 4567
```

## 3. Run the Devkit & Launch Odometry

Open a new terminal to start the API/Devkit container, mounting your local ROS2 workspaces directly into the container:

```bash
docker run   --name autodrive_roboracer_api   --rm -it   --entrypoint /bin/bash   --network=host   --ipc=host   -v /tmp/.X11-unix:/tmp/.X11-unix:rw   -v /home/ubuntu/roboracer_compet_ws/src/compete_bringup:/home/autodrive_devkit/src/compete_bringup   -v /home/ubuntu/roboracer_compet_ws/src/compete_localiztion:/home/autodrive_devkit/src/compete_localiztion   -v /home/ubuntu/roboracer_compet_ws/src/rf2o_laser_odometry-ros2:/home/autodrive_devkit/src/rf2o_laser_odometry-ros2   --env DISPLAY   --privileged   roboracer_localization:latest
```

Inside the devkit container, build the workspace, source it, and launch the filtered odometry node:

```bash
# Navigate to the workspace root if needed, then build:
colcon build
source install/setup.bash

# Launch the filtered odometry
ros2 launch compete_bringup odom_filterd.launch.xml
```

You should now have the filtered odometry successfully running for your Roboracer!