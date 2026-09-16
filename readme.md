# AutoDRIVE Roboracer: Localization & Control Setup

This guide explains how to build the custom Docker environment, start the AutoDRIVE simulator, and run the complete localization and control stack.

## 1. Docker Setup & Build

Create the Docker folder and copy `range_libc` into it:

```bash
mkdir autodrive_roboracer_docker
cd autodrive_roboracer_docker
touch Dockerfile
```

Add the following to `Dockerfile`:

```dockerfile
FROM autodriveecosystem/autodrive_roboracer_api:2026-icra-explore

RUN apt-get update && \
    apt-get install -y \
        ros-humble-joy \
        ros-humble-teleop-twist-joy \
        ros-humble-joy-teleop \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-slam-toolbox \
        ros-humble-rqt-reconfigure \
        ros-humble-robot-localization \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    pandas \
    matplotlib \
    scipy \
    Cython

COPY range_libc /tmp/range_libc

RUN cd /tmp/range_libc/pywrapper && \
    python3 setup.py install && \
    rm -rf /tmp/range_libc
```

Build the Docker image:

```bash
docker build --no-cache -t roboracer_localization .
```

---

## 2. Start the Simulator

Allow local X11 connections:

```bash
xhost local:root
```

Start the simulator container:

```bash
docker run \
  --name autodrive_roboracer_sim \
  --rm -it \
  --entrypoint /bin/bash \
  --network=host \
  --ipc=host \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env DISPLAY \
  --privileged \
  autodriveecosystem/autodrive_roboracer_sim:<tag>
```

Replace `<tag>` with your simulator image tag.

Inside the simulator container:

```bash
./AutoDRIVE* Simulator.x86_64 \
  -batchmode \
  -nographics \
  -ip 127.0.0.1 \
  -port 4567
```

---

## 3. Run the Devkit & Localization Stack

Open a new terminal and start the API/Devkit container:
NOTE ghange user name and workspace to match your PC "/home/ubuntu/roboracer_compet_ws " 
change that to your PC

```bash
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
```

Inside the Devkit container:

```bash
cd /home/autodrive_devkit

colcon build

source install/setup.bash
```

Launch the complete localization stack:

```bash
ros2 launch compete_bringup final_odom.launch.xml

This launches the complete RoboRacer localization and control environment.