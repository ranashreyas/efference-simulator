"""Reading and writing recorded episodes. Shared by the Pi logger, the Mac labeler and training.

An episode directory holds:
    meta.json          camera / actuator settings, scene, split, frame counts
    chunk_0000.npz     model data: grayscale full frames at LOG_SIZE plus per-frame timestamps
                       and eye state (see logger.py)
    preview.mp4        color video at PREVIEW_SIZE for a human to watch; video frame i is
                       logged frame i
    labels.json        anomaly intervals in frame indices (and camera seq numbers)
    labels.npy         per-frame 0/1 anomaly label, one entry per logged frame

Frame index i always means "the i-th logged frame", in every file above.
"""
import glob
import json
import os

import numpy as np


def chunk_paths(ep_dir):
    return sorted(glob.glob(os.path.join(ep_dir, "chunk_*.npz")))


def frame_index(ep_dir):
    """Per-frame seq and t_sensor arrays (reads only those fields, not the images)."""
    seqs, times = [], []
    for path in chunk_paths(ep_dir):
        with np.load(path) as d:
            seqs.append(d["seq"])
            times.append(d["t_sensor"])
    if not seqs:
        return np.zeros(0, np.int64), np.zeros(0)
    return np.concatenate(seqs), np.concatenate(times)


def load_meta(ep_dir):
    path = os.path.join(ep_dir, "meta.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def merge_intervals(intervals, n_frames):
    """Clip to [0, n_frames-1], sort, and merge overlapping or touching (start, end) pairs."""
    clipped = sorted((max(0, min(a, b)), min(n_frames - 1, max(a, b))) for a, b in intervals)
    merged = []
    for a, b in clipped:
        if a > b:
            continue
        if merged and a <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def load_labels(ep_dir, seqs=None):
    """Anomaly intervals as inclusive (start_frame, end_frame) pairs. Also reads the older
    live-label format that stored camera seq numbers only."""
    path = os.path.join(ep_dir, "labels.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        intervals = json.load(f).get("intervals", [])
    out = []
    for lab in intervals:
        if "start_frame" in lab:
            out.append((lab["start_frame"], lab["end_frame"]))
        elif lab.get("end_seq") is not None:
            if seqs is None:
                seqs, _ = frame_index(ep_dir)
            a = int(np.searchsorted(seqs, lab["start_seq"]))
            b = int(np.searchsorted(seqs, lab["end_seq"], side="right")) - 1
            out.append((a, b))
    return out


def save_labels(ep_dir, intervals, seqs, source):
    """Write labels.json (intervals) and labels.npy (per-frame 0/1). Returns the merged intervals."""
    n = len(seqs)
    merged = merge_intervals(intervals, n) if n else []
    per_frame = np.zeros(n, dtype=np.uint8)
    for a, b in merged:
        per_frame[a:b + 1] = 1
    np.save(os.path.join(ep_dir, "labels.npy"), per_frame)
    with open(os.path.join(ep_dir, "labels.json"), "w") as f:
        json.dump(dict(
            source=source,
            frames=n,
            anomalous_frames=int(per_frame.sum()),
            intervals=[dict(start_frame=int(a), end_frame=int(b),
                            start_seq=int(seqs[a]), end_seq=int(seqs[b])) for a, b in merged],
        ), f, indent=2)
    return merged
