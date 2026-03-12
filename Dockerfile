
FROM osrf/ros:humble-desktop-full

RUN apt update && apt upgrade -y

RUN apt install -y ros-humble-slam-toolbox python3-pip ffmpeg

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /workspace/ros2_ws

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]