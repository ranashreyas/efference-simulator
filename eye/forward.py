"""Runs the exported forward model live on the Pi and turns its output into surprise.

Each camera frame: keep the model input and eye state, then predict the current frame from
the frames 1/15 s, 2/15 s and 3/15 s earlier (the spacing the model was trained on; at 30 fps
that's frames t-6, t-4, t-2), so there's a fresh prediction every frame. History frames are
picked by timestamp, so a dropped frame can't silently change the spacing: if a needed frame
is missing, that prediction is skipped. Surprise and the frame-difference baseline are computed
exactly as in train/evaluate.py.
"""
import collections
import math
import time

import cv2
import numpy as np

from . import motor

N_FRAMES = 3
# Expected error is never allowed below ~2 gray levels of camera noise. Without this floor the
# model claims sub-noise precision on still background and its surprise map fills with speckle.
LOGVAR_FLOOR = math.log((2 / 255) ** 2)


def top_mean(m, frac=0.01):
    """Mean of the top `frac` of a lightly blurred map: one event score per frame."""
    b = cv2.blur(m, (5, 5)).ravel()
    k = max(1, int(len(b) * frac))
    return float(np.partition(b, -k)[-k:].mean())


class LiveForwardModel:
    def __init__(self, onnx_path, step_s=1 / 15, threads=2, actual_noise_deg=0.0):
        import onnxruntime as ort                      # only needed on the Pi
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        # models trained before the current motor encoding take a 32-wide input and never used it:
        # give those zeros, exactly as in their training
        self.motor_dim = self.session.get_inputs()[1].shape[1]
        self.uses_motor = self.motor_dim == motor.ACTION_DIM
        self.step_s = step_s
        self.history = collections.deque(maxlen=24)    # (t, model input), ~0.8 s at 30 fps
        self.states = collections.deque(maxlen=motor.STEPS)
        self.skipped = 0                               # predictions skipped for missing history
        self.actual_noise_deg = actual_noise_deg       # simulated position-sensor noise (moving eye)
        self.rng = np.random.default_rng()

    def _frame_at(self, t):
        """The stored frame closest to time t, or None if none is within a quarter step (8 ms at
        15 fps: well above timestamp jitter, well below the 33 ms between camera frames)."""
        best = min(self.history, key=lambda h: abs(h[0] - t))
        return best[1] if abs(best[0] - t) <= self.step_s / 4 else None

    def step(self, t, model_input, state):
        """t: frame timestamp (s); model_input: (96, 128) uint8; state: eye MotorState at t.
        Returns None until there's enough history, or if a needed history frame is missing."""
        self.history.append((t, model_input))
        self.states.append(state)
        if t - self.history[0][0] < N_FRAMES * self.step_s - self.step_s / 4:
            return None
        past = [self._frame_at(t - k * self.step_s) for k in range(N_FRAMES, 0, -1)]
        if any(p is None for p in past):
            self.skipped += 1
            return None
        frames = np.stack(past)[None].astype(np.float32) / 255
        target = model_input.astype(np.float32) / 255
        t0 = time.perf_counter()
        m = (motor.encode(self.states, self.actual_noise_deg, self.rng) if self.uses_motor
             else np.zeros(self.motor_dim, np.float32))
        pred, logvar = self.session.run(None, {"frames": frames, "motor": m[None]})
        infer_ms = (time.perf_counter() - t0) * 1000
        pred, logvar = pred[0, 0], logvar[0, 0]
        surprise = (target - pred) ** 2 * np.exp(-np.maximum(logvar, LOGVAR_FLOOR))
        diff = (target - frames[0, -1]) ** 2               # change over the same 1/15 s
        return dict(surprise=surprise, diff=diff, pred=pred, infer_ms=infer_ms,
                    surprise_score=top_mean(surprise), diff_score=top_mean(diff))
