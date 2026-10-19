"""Message types shared by every part of the system.

All timestamps are seconds on CLOCK_MONOTONIC (time.monotonic() on the Pi), so frames
and motor commands can be aligned exactly.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    seq: int            # frame counter from the camera thread
    t_sensor: float     # V4L2 buffer timestamp (when the driver finished receiving the frame)
    t_host: float       # when our process got it
    img: np.ndarray     # BGR, uint8


@dataclass
class MotorCmd:
    t_issue: float      # when the command was sent
    pan: float          # target angle, degrees (+ = right)
    tilt: float         # target angle, degrees (+ = up)


@dataclass
class MotorState:
    t: float            # time this state estimate refers to
    pan: float          # best estimate of actual angle, degrees
    tilt: float
    pan_target: float   # most recent command sent as of t (the efference copy: it leads the
                        # actual angle by the servo's reaction delay)
    tilt_target: float
