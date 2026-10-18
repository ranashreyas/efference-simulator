"""Episode logger: writes frames, eye state and motor commands to disk in chunks.

Layout:
    runs/<episode>/meta.json          camera + actuator settings
    runs/<episode>/chunk_0000.npz     frames (N, H, W) uint8 grayscale full frame at LOG_SIZE,
                                      seq, t_sensor, t_host, pan, tilt, pan_target, tilt_target,
                                      cmd_t, cmd_pan, cmd_tilt (commands sent during this chunk)

Compression happens on a background thread so the capture loop never waits on the SD card.
"""
import json
import os
import queue
import threading
import time

import numpy as np

from .preprocess import LOG_SIZE, to_gray_small


class EpisodeLogger:
    def __init__(self, root="runs", name=None, meta=None, chunk_frames=300):
        name = name or time.strftime("ep_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.chunk_frames = chunk_frames
        self._meta = dict(meta or {}, log_size=LOG_SIZE, started=time.time())
        self._rows, self._frames, self._cmds = [], [], []
        self._chunk = 0
        self.frames_written = 0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="logger", daemon=True)
        self._thread.start()

    def add(self, frame, state):
        self._frames.append(to_gray_small(frame.img, LOG_SIZE))
        self._rows.append((frame.seq, frame.t_sensor, frame.t_host,
                           state.pan, state.tilt, state.pan_target, state.tilt_target))
        if len(self._frames) >= self.chunk_frames:
            self._flush()

    def add_command(self, cmd):
        self._cmds.append((cmd.t_issue, cmd.pan, cmd.tilt))

    def _flush(self):
        if not self._frames:
            return
        rows = np.array(self._rows, dtype=np.float64)
        cmds = np.array(self._cmds, dtype=np.float64).reshape(-1, 3)
        data = dict(frames=np.stack(self._frames), seq=rows[:, 0].astype(np.int64),
                    t_sensor=rows[:, 1], t_host=rows[:, 2], pan=rows[:, 3], tilt=rows[:, 4],
                    pan_target=rows[:, 5], tilt_target=rows[:, 6],
                    cmd_t=cmds[:, 0], cmd_pan=cmds[:, 1], cmd_tilt=cmds[:, 2])
        self._q.put((os.path.join(self.dir, f"chunk_{self._chunk:04d}.npz"), data))
        self.frames_written += len(self._frames)
        self._chunk += 1
        self._rows, self._frames, self._cmds = [], [], []

    def _writer(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            path, data = item
            np.savez_compressed(path, **data)

    def close(self):
        self._flush()
        self._q.put(None)
        self._thread.join()
        self._meta.update(frames=self.frames_written, chunks=self._chunk, ended=time.time())
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        print(f"[logger] wrote {self.frames_written} frames in {self._chunk} chunks to {self.dir}")
