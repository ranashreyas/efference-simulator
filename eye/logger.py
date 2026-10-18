"""Episode logger: writes frames, eye state and motor commands to disk in chunks, plus a color
preview video for labeling. File layout is described in episode.py.

    recordings/<episode>/chunk_0000.npz   frames (N, H, W) uint8 grayscale full frame at LOG_SIZE,
                                          seq, t_sensor, t_host, pan, tilt, pan_target, tilt_target,
                                          cmd_t, cmd_pan, cmd_tilt (commands sent during this chunk)
    recordings/<episode>/preview.mp4      color video at PREVIEW_SIZE, one video frame per logged frame

Compression and video encoding happen on background threads so the capture loop never waits.
"""
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

from .episode import save_labels
from .preprocess import LOG_SIZE, PREVIEW_SIZE, to_gray_small


class EpisodeLogger:
    def __init__(self, root="recordings", name=None, meta=None, fps=30, chunk_frames=300):
        name = name or time.strftime("ep_%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, name)
        os.makedirs(self.dir, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.fps = fps
        self._meta = dict(meta or {}, log_size=LOG_SIZE, preview_size=PREVIEW_SIZE,
                          started=time.time())
        self._rows, self._frames, self._cmds = [], [], []
        self._all_seqs = []             # every logged frame's seq, in order
        self._labels = []               # live labels from the viewer: [{"start_seq", "end_seq"}]
        self._chunk = 0
        self.frames_written = 0
        self.preview_frames = 0
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._writer, name="logger", daemon=True)
        self._thread.start()
        self._video_q = queue.Queue()
        self._video_thread = threading.Thread(target=self._video_writer, name="preview", daemon=True)
        self._video_thread.start()

    def add(self, frame, state):
        self._frames.append(to_gray_small(frame.img, LOG_SIZE))
        self._video_q.put(cv2.resize(frame.img, PREVIEW_SIZE, interpolation=cv2.INTER_AREA))
        self._all_seqs.append(frame.seq)
        self._rows.append((frame.seq, frame.t_sensor, frame.t_host,
                           state.pan, state.tilt, state.pan_target, state.tilt_target))
        if len(self._frames) >= self.chunk_frames:
            self._flush()

    def add_command(self, cmd):
        self._cmds.append((cmd.t_issue, cmd.pan, cmd.tilt))

    # ---- optional live labels from the viewer (offline labeling happens on the Mac) ----------
    @property
    def label_active(self):
        return bool(self._labels) and self._labels[-1]["end_seq"] is None

    @property
    def label_count(self):
        return len(self._labels)

    def add_label(self, event):
        """event: {"state": "start"|"end", "seq": frame the viewer was showing, ...}"""
        if event["state"] == "start" and not self.label_active:
            self._labels.append(dict(start_seq=event["seq"], end_seq=None))
        elif event["state"] == "end" and self.label_active:
            self._labels[-1]["end_seq"] = event["seq"]

    def _write_labels(self):
        seqs = np.array(self._all_seqs, dtype=np.int64)
        if self.label_active:                       # still open when recording stopped
            self._labels[-1]["end_seq"] = int(seqs[-1]) if len(seqs) else None
        intervals = []
        for lab in self._labels:
            if lab["end_seq"] is not None:
                a = int(np.searchsorted(seqs, lab["start_seq"]))
                b = int(np.searchsorted(seqs, lab["end_seq"], side="right")) - 1
                intervals.append((a, b))
        merged = save_labels(self.dir, intervals, seqs, source="live")
        return sum(b - a + 1 for a, b in merged)

    # ---- background writers ------------------------------------------------------------------
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

    def _video_writer(self):
        path = os.path.join(self.dir, "preview.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, PREVIEW_SIZE)
        if not writer.isOpened():
            print("[logger] could not open preview.mp4 for writing; recording without preview")
            writer = None
        while True:
            img = self._video_q.get()
            if img is None:
                break
            if writer is not None:
                writer.write(img)
                self.preview_frames += 1
        if writer is not None:
            writer.release()

    def close(self):
        self._flush()
        self._q.put(None)
        self._video_q.put(None)
        self._thread.join()
        self._video_thread.join()
        anomalous = self._write_labels()
        self._meta.update(frames=self.frames_written, preview_frames=self.preview_frames,
                          chunks=self._chunk, ended=time.time(),
                          duration_s=round(self.frames_written / self.fps, 1),
                          live_label_intervals=len(self._labels), anomalous_frames=anomalous)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self._meta, f, indent=2)
        print(f"[logger] wrote {self.frames_written} frames ({self.frames_written / self.fps:.0f} s) "
              f"in {self._chunk} chunks + preview.mp4 ({self.preview_frames} frames) to {self.dir}")
