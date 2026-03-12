docker build -t navbot_ros:v1 .
xhost +local:docker
docker run -it --rm --name navbot --network host \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v $PWD/navbot_ros:/workspace/ros2_ws/src/navbot/navbot_ros \
 --device /dev/video0:/dev/video0 \
 navbot_ros:v1