"""Actuators: what moves the eye.

Every actuator has the same interface, so the rest of the system never knows which one
it is talking to:

    command(pan, tilt, t)  -> send a target angle (degrees)
    state(t)               -> MotorState: best estimate of where the eye actually is at time t
    view(frame, state)     -> the image the eye sees, given the camera frame and eye state

NullActuator     never moves (phase 1: static camera)
VirtualActuator  a crop window that slides around the full camera frame, driven by a
                 simulated servo with latency, speed limit, backlash and noise (phase 2)
ServoActuator    real pan-tilt servos (phase 3, not written yet)
"""
import bisect
import collections
import math
import random
import time

import cv2
import numpy as np

from .messages import Frame, MotorCmd, MotorState

# Logitech C922: 78 deg diagonal field of view. At 1280x720 that's a focal length of
# about 907 px (half the diagonal / tan(39 deg)).
C922_FOCAL_PX_720P = 907.0


class ServoModel:
    """A cheap hobby servo, one per axis: command delay, max speed, backlash and jitter."""

    def __init__(self, latency=0.06, max_speed=300.0, backlash=0.5, noise=0.05, seed=None):
        self.latency = latency          # s between sending a command and the servo reacting
        self.max_speed = max_speed      # deg/s (an SG90 is rated ~600 deg/s unloaded; slower carrying a camera)
        self.backlash = backlash        # deg of slop in the gears
        self.noise = noise              # deg std of position jitter
        self._rng = np.random.default_rng(seed)
        self._cmds = collections.deque(maxlen=256)
        self._target = np.zeros(2)
        self._motor = np.zeros(2)       # where the motor shaft is
        self._out = np.zeros(2)         # where the camera actually points (after backlash)
        self._t = None

    def command(self, cmd: MotorCmd):
        self._cmds.append(cmd)

    def step(self, t):
        if self._t is None:
            self._t = t
        if t <= self._t:                # only move forward in time
            return self._out.copy(), self._target.copy()
        # The servo only knows about commands sent at least `latency` ago.
        while self._cmds and self._cmds[0].t_issue <= t - self.latency:
            c = self._cmds.popleft()
            self._target = np.array([c.pan, c.tilt])
        dt = t - self._t
        self._t = t
        delta = self._target - self._motor
        max_step = self.max_speed * dt
        self._motor += np.clip(delta, -max_step, max_step)
        half = self.backlash / 2
        self._out = np.clip(self._out, self._motor - half, self._motor + half)
        jitter = self._rng.normal(0, self.noise, 2) if self.noise > 0 else 0
        return self._out + jitter, self._target.copy()


class VirtualActuator:
    """A crop window inside the full frame acts as the eye. Pan/tilt angles become pixel
    offsets through the camera's focal length.

    Note: shifting a crop is a pure image translation. A real rotation also adds slight
    perspective distortion toward the edges; that is fine for now and is exactly the kind
    of difference the learned model will have to absorb when real servos arrive.
    """

    def __init__(self, full_size=(1280, 720), crop_size=(480, 360),
                 focal_px=C922_FOCAL_PX_720P, servo=None):
        self.full_w, self.full_h = full_size
        self.crop_w, self.crop_h = crop_size
        self.f = focal_px
        self.servo = servo or ServoModel()
        self.pan_limit = math.degrees(math.atan((self.full_w - self.crop_w) / 2 / self.f))
        self.tilt_limit = math.degrees(math.atan((self.full_h - self.crop_h) / 2 / self.f))
        self.commands = []              # every command ever sent, for logging and the efference copy
        self._cmd_times = []

    def command(self, pan, tilt, t=None):
        t = time.monotonic() if t is None else t
        pan = float(np.clip(pan, -self.pan_limit, self.pan_limit))
        tilt = float(np.clip(tilt, -self.tilt_limit, self.tilt_limit))
        cmd = MotorCmd(t, pan, tilt)
        self.servo.command(cmd)
        self.commands.append(cmd)
        self._cmd_times.append(t)
        return cmd

    def state(self, t=None):
        """Actual angle from the servo model; commanded angle = the most recent command *sent*
        by time t (not the delayed copy the servo is acting on), so it leads the motion."""
        t = time.monotonic() if t is None else t
        (pan, tilt), _ = self.servo.step(t)
        pan = float(np.clip(pan, -self.pan_limit, self.pan_limit))
        tilt = float(np.clip(tilt, -self.tilt_limit, self.tilt_limit))
        k = bisect.bisect_right(self._cmd_times, t) - 1
        pt, tt = (self.commands[k].pan, self.commands[k].tilt) if k >= 0 else (0.0, 0.0)
        return MotorState(t, pan, tilt, pt, tt)

    def crop_center(self, state):
        cx = self.full_w / 2 + self.f * math.tan(math.radians(state.pan))
        cy = self.full_h / 2 - self.f * math.tan(math.radians(state.tilt))   # + tilt = look up
        return cx, cy

    def crop_rect(self, state):
        cx, cy = self.crop_center(state)
        return (int(cx - self.crop_w / 2), int(cy - self.crop_h / 2), self.crop_w, self.crop_h)

    def view(self, frame: Frame, state: MotorState) -> Frame:
        cx, cy = self.crop_center(state)
        crop = cv2.getRectSubPix(frame.img, (self.crop_w, self.crop_h), (cx, cy))  # sub-pixel accurate
        return Frame(frame.seq, frame.t_sensor, frame.t_host, crop)


class NullActuator(VirtualActuator):
    """Static eye: same field of view as the virtual eye, fixed at the center."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pan_limit = self.tilt_limit = 0.0

    def state(self, t=None):
        t = time.monotonic() if t is None else t
        return MotorState(t, 0.0, 0.0, 0.0, 0.0)


class ServoActuator:
    """Real pan-tilt servos driven over PCA9685 / hardware PWM. Written in phase 3."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ServoActuator arrives with the servos (phase 3)")


class SaccadeBabbler:
    """Random eye movements for collecting self-supervised training data ("motor babbling").
    Mostly big jumps, sometimes small corrective ones, with random pauses between them."""

    def __init__(self, pan_limit, tilt_limit, pause=(0.3, 1.2), small_prob=0.3, seed=None):
        self.pan_limit, self.tilt_limit = pan_limit, tilt_limit
        self.pause = pause
        self.small_prob = small_prob
        self._rng = random.Random(seed)
        self._next_t = None
        self._pan = self._tilt = 0.0

    def step(self, t):
        """Return a new (pan, tilt) target when it's time to move, otherwise None."""
        if self._next_t is None:
            self._next_t = t + self._rng.uniform(*self.pause)
        if t < self._next_t:
            return None
        self._next_t = t + self._rng.uniform(*self.pause)
        if self._rng.random() < self.small_prob:
            self._pan += self._rng.gauss(0, 1.5)
            self._tilt += self._rng.gauss(0, 1.0)
        else:
            self._pan = self._rng.uniform(-0.9, 0.9) * self.pan_limit
            self._tilt = self._rng.uniform(-0.9, 0.9) * self.tilt_limit
        self._pan = max(-self.pan_limit, min(self.pan_limit, self._pan))
        self._tilt = max(-self.tilt_limit, min(self.tilt_limit, self._tilt))
        return self._pan, self._tilt


class SweepScan:
    """Back-and-forth horizontal pan: left to right over `sweep_s`, pause `pause_s`, right to
    left over `sweep_s`, pause `pause_s`, repeat. Tilt stays level. A new target is commanded
    every frame, so the (simulated) servo follows a smooth constant-speed path."""

    def __init__(self, pan_limit, sweep_s=10.0, pause_s=1.0, margin=0.95):
        self.amp = pan_limit * margin
        self.sweep_s, self.pause_s = sweep_s, pause_s
        self._t0 = None

    def step(self, t):
        if self._t0 is None:
            self._t0 = t
        period = 2 * (self.sweep_s + self.pause_s)
        ph = (t - self._t0) % period
        if ph < self.sweep_s:                                   # left -> right
            pan = -self.amp + 2 * self.amp * ph / self.sweep_s
        elif ph < self.sweep_s + self.pause_s:                  # pause at the right
            pan = self.amp
        elif ph < 2 * self.sweep_s + self.pause_s:              # right -> left
            pan = self.amp - 2 * self.amp * (ph - self.sweep_s - self.pause_s) / self.sweep_s
        else:                                                   # pause at the left
            pan = -self.amp
        return pan, 0.0


class RandomPan:
    """Slow, smooth, wandering pans for training data. Every 3-9 s (or on reaching the edge of
    the range) the eye picks a new random direction (pan and tilt) and a new speed around the
    10 s sweep's ~4 deg/s. Velocity eases toward each new heading and speed, and the heading
    drifts a little within a segment, so paths curve and speeds vary. Returns a target every
    frame; the servo model adds latency, speed limit, backlash and jitter on top."""

    def __init__(self, pan_limit, tilt_limit, segment_s=(3.0, 9.0), speed=(2.5, 6.5),
                 ease_s=(0.4, 1.5), drift_deg_s=15.0, margin=0.95, seed=None):
        self.lim = np.array([pan_limit, tilt_limit]) * margin
        self.segment_s, self.speed, self.ease_s, self.drift = segment_s, speed, ease_s, drift_deg_s
        self.rng = np.random.default_rng(seed)
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self._t = None

    def _new_segment(self, t, inward=None):
        heading = self.rng.uniform(0, 2 * np.pi)
        if inward is not None:                                  # bounced: head back into range
            d = np.array([np.cos(heading), np.sin(heading)])
            d[inward != 0] = -np.sign(inward[inward != 0]) * np.abs(d[inward != 0])
            heading = np.arctan2(d[1], d[0])
        self.heading = heading
        self.target_speed = self.rng.uniform(*self.speed)
        self.ease = self.rng.uniform(*self.ease_s)
        self.seg_end = t + self.rng.uniform(*self.segment_s)

    def step(self, t):
        if self._t is None:
            self._t = t
            self._new_segment(t)
        dt = max(0.0, t - self._t)
        self._t = t
        if t >= self.seg_end:
            self._new_segment(t)
        self.heading += np.radians(self.rng.normal(0, self.drift)) * np.sqrt(dt)   # gentle curving
        want = self.target_speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        self.vel += (want - self.vel) * (1 - np.exp(-dt / self.ease))             # smooth speed/turns
        nxt = self.pos + self.vel * dt
        hit = np.where(nxt > self.lim, 1, np.where(nxt < -self.lim, -1, 0))
        if hit.any():                                           # edge: clamp and pick a new direction
            nxt = np.clip(nxt, -self.lim, self.lim)
            self.vel[hit != 0] = 0
            self._new_segment(t, inward=hit)
        self.pos = nxt
        return float(self.pos[0]), float(self.pos[1])
