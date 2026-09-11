open simulator first thenn



in terminal 
first launch odom 

ros2 run compete_localiztion ackermann_odom.py 


second launch EKF
ros2 launch compete_localiztion local_localization.launch.xml 

the output odom name is 

/odometry/filtered
