#!/usr/bin/env python3
"""Send a constant forward command with zero steering over UDP."""

import socket
import time

from navbot_ros import parameters
from navbot_ros import robot_python_code


FORWARD_SPEED = 100
STRAIGHT_STEERING = 0
COMMAND_PERIOD_S = 0.1


def main() -> int:
    udp, ok = robot_python_code.create_udp_communication(
        parameters.arduinoIP,
        parameters.localIP,
        parameters.arduinoPort,
        parameters.localPort,
        parameters.bufferSize,
    )
    if not ok:
        return 1

    sender = robot_python_code.MsgSender(
        time.perf_counter(),
        parameters.num_robot_control_signals,
        udp,
    )
    receiver = robot_python_code.MsgReceiver(
        time.perf_counter(),
        parameters.num_robot_sensors,
        udp,
    )
    udp.UDPServerSocket.settimeout(0.02)
    last_encoder = None

    print(
        f"Sending straight command to {parameters.arduinoIP}:{parameters.arduinoPort} "
        f"(speed={FORWARD_SPEED}, steering={STRAIGHT_STEERING}). Ctrl+C to stop.",
        flush=True,
    )

    try:
        while True:
            sender.send_control_signal([FORWARD_SPEED, STRAIGHT_STEERING])

            try:
                receive_ret, packed_receive_msg = receiver.receive()
                if receive_ret:
                    unpack_ret, unpacked_receive_msg = receiver.unpack_msg(packed_receive_msg)
                    if unpack_ret and len(unpacked_receive_msg) >= 3:
                        sensor = robot_python_code.RobotSensorSignal(unpacked_receive_msg)
                        if sensor.encoder_counts != last_encoder:
                            last_encoder = sensor.encoder_counts
                            print(f"encoder_counts={sensor.encoder_counts}", flush=True)
            except socket.timeout:
                pass

            time.sleep(COMMAND_PERIOD_S)
    except KeyboardInterrupt:
        pass
    finally:
        # Send a few stop commands to increase the chance the robot receives one.
        for _ in range(3):
            sender.send_control_signal([0, 0])
            time.sleep(COMMAND_PERIOD_S)
        try:
            udp.UDPServerSocket.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
