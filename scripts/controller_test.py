#!/usr/bin/env python3
"""Simple controller test: print button/axis/hat events."""

import time

import pygame


def init_controller(index: int = 0):
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count == 0:
        print("No controller detected.")
        return None
    if index >= count:
        print(f"Requested index {index}, but only {count} controller(s) found.")
        return None
    js = pygame.joystick.Joystick(index)
    js.init()
    print(
        f"Connected: {js.get_name()} "
        f"(axes={js.get_numaxes()}, buttons={js.get_numbuttons()}, hats={js.get_numhats()})"
    )
    return js


def main():
    js = init_controller(0)
    if js is None:
        return 1

    print("Listening for controller events. Press Ctrl+C to exit.")
    last_axis = {}

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # Avoid spam by printing only when value changes significantly.
                    val = round(event.value, 3)
                    last = last_axis.get(event.axis)
                    if last != val:
                        last_axis[event.axis] = val
                        print(f"AXIS {event.axis}: {val}")
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"BUTTON {event.button}: down")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"BUTTON {event.button}: up")
                elif event.type == pygame.JOYHATMOTION:
                    print(f"HAT {event.hat}: {event.value}")
                elif event.type == pygame.JOYDEVICEADDED:
                    print("Controller added.")
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print("Controller removed.")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if js is not None:
            js.quit()
        pygame.joystick.quit()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
