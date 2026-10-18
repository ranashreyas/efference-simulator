"""UVC camera capture: a background thread keeps a ring buffer of timestamped frames.

On start the camera is allowed to auto-adjust (white balance, focus) for a moment, then
everything is locked and exposure/gain are chosen once by a small search. After that
nothing in the camera changes on its own, so any pixel change comes from the world or
from our own motion, never from the camera's auto modes.
"""
import collections
import glob
import os
import subprocess
import threading
import time

import cv2
import numpy as np

from .messages import Frame


def find_uvc_device():
    """Return /dev/videoN for the first UVC capture node (index 0; index 1 is metadata)."""
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        driver = os.path.realpath(os.path.join(path, "device", "driver"))
        index_file = os.path.join(path, "index")
        index = open(index_file).read().strip() if os.path.exists(index_file) else "0"
        if driver.endswith("uvcvideo") and index == "0":
            return "/dev/" + os.path.basename(path)
    return None


def scene_brightness(img):
    """Mean brightness ignoring the brightest 2% of pixels, so a lamp in view doesn't dominate."""
    gray = cv2.cvtColor(cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    cutoff = np.percentile(gray, 98)
    return float(gray[gray <= cutoff].mean())


class Camera:
    def __init__(self, device=None, width=1280, height=720, fps=30, powerline_hz=60, ring=64):
        self.device = device or find_uvc_device()
        if self.device is None:
            raise RuntimeError("No UVC camera found. Check `lsusb` and the cable.")
        self.width, self.height, self.fps = width, height, fps
        self.powerline_hz = powerline_hz
        self.controls = {}
        self._ring = collections.deque(maxlen=ring)
        self._cond = threading.Condition()
        self._seq = 0
        self._running = False
        self._cap = None
        self._thread = None

    # ---- v4l2 controls -------------------------------------------------------------
    def set_ctrl(self, name, value):
        r = subprocess.run(["v4l2-ctl", "-d", self.device, f"--set-ctrl={name}={value}"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            self.controls[name] = value
        return r.returncode == 0

    def set_exposure(self, exposure, gain):
        self.set_ctrl("exposure_time_absolute", exposure)  # units of 100 us
        self.set_ctrl("gain", gain)

    # ---- lifecycle -----------------------------------------------------------------
    def start(self, calibrate=True, target_brightness=100):
        self.set_ctrl("power_line_frequency", 2 if self.powerline_hz == 60 else 1)
        self.set_ctrl("exposure_dynamic_framerate", 0)
        self.set_ctrl("auto_exposure", 3)                 # auto for now, so WB/focus see a sane image
        self.set_ctrl("white_balance_automatic", 1)
        self.set_ctrl("focus_automatic_continuous", 1)

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {self.device}")
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

        if calibrate:
            self._settle(2.0)                             # let auto WB and autofocus converge
        self.set_ctrl("white_balance_automatic", 0)
        self.set_ctrl("focus_automatic_continuous", 0)
        self.set_ctrl("auto_exposure", 1)                 # manual from here on
        if calibrate:
            self.calibrate_exposure(target_brightness)
        else:
            self.set_exposure(333, 64)
        return self

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()

    def _loop(self):
        while self._running:
            ok, img = self._cap.read()
            t_host = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            t_sensor = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            with self._cond:
                self._seq += 1
                self._ring.append(Frame(self._seq, t_sensor, t_host, img))
                self._cond.notify_all()

    # ---- frame access --------------------------------------------------------------
    def latest(self):
        with self._cond:
            return self._ring[-1] if self._ring else None

    def wait_next(self, after_seq=None, timeout=1.0):
        """Block until a frame newer than after_seq arrives (or the latest, if after_seq is None)."""
        with self._cond:
            last = self._ring[-1].seq if self._ring else 0
            after = last if after_seq is None else after_seq
            ok = self._cond.wait_for(lambda: self._ring and self._ring[-1].seq > after, timeout)
            return self._ring[-1] if ok else None

    def recent(self, n):
        with self._cond:
            return list(self._ring)[-n:]

    # ---- exposure calibration --------------------------------------------------------
    def _settle(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.wait_next(timeout=0.5)

    def _measure(self, settle_frames=6):
        f = None
        for _ in range(settle_frames):                    # a new setting takes a few frames to apply
            f = self.wait_next(timeout=0.5) or f
        return scene_brightness(f.img) if f is not None else 0.0

    def calibrate_exposure(self, target=100):
        """Walk a ladder from darkest to brightest and stop at the first setting bright enough.

        Order of preference: short exposure (less motion blur), then longer exposure in whole
        multiples of the light-flicker period (no banding), then more gain (adds noise).
        """
        period = 83 if self.powerline_hz == 60 else 100   # 8.3 ms / 10 ms, in 100 us units
        max_exp = int(10000 / self.fps)                    # can't expose longer than one frame
        flicker_safe = [k * period for k in range(1, 10) if k * period <= max_exp]
        ladder = [(e, 0) for e in (10, 20, 40)]
        ladder += [(e, 0) for e in flicker_safe]
        ladder += [(flicker_safe[-1], g) for g in (32, 64, 96, 128, 160, 192, 224, 255)]

        chosen, level = ladder[-1], 0.0
        for exposure, gain in ladder:
            self.set_exposure(exposure, gain)
            level = self._measure()
            if level >= 0.9 * target:
                chosen = (exposure, gain)
                break
        self.set_exposure(*chosen)
        self.brightness_at_lock = level
        print(f"[camera] locked exposure={chosen[0] / 10:.1f} ms gain={chosen[1]} "
              f"(scene brightness {level:.0f}/255, target {target})")
        return chosen
