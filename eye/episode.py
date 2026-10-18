"""Reading and writing recorded episodes. Shared by the Pi logger, the Mac labeler and training.

An episode directory holds:
    meta.json          camera / actuator settings, scene, frame counts
    chunk_0000.npz     model data: grayscale full frames at LOG_SIZE plus per-frame timestamps
                       and eye state (see logger.py)
    preview.mp4        color video at PREVIEW_SIZE for a human to watch; video frame i is
                       logged frame i
    labels.json        optional, written by tools/label.py: event intervals in frame indices
                       (and camera seq numbers)
    labels.npy         optional: per-frame 0/1 event label, one entry per logged frame

Frame index i always means "the i-th logged frame", in every file above. The model trains on
every frame except the held-out part (see holdout_split); labels are only used to score it.
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
    """meta.json as a dict ({} if missing). Older episodes may carry a "split" key; ignore it."""
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
    """Event intervals as inclusive (start_frame, end_frame) pairs. Also reads older files that
    stored camera seq numbers only."""
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


def save_labels(ep_dir, intervals, seqs):
    """Write labels.json (intervals) and labels.npy (per-frame 0/1). Returns the merged intervals."""
    n = len(seqs)
    merged = merge_intervals(intervals, n) if n else []
    per_frame = np.zeros(n, dtype=np.uint8)
    for a, b in merged:
        per_frame[a:b + 1] = 1
    np.save(os.path.join(ep_dir, "labels.npy"), per_frame)
    with open(os.path.join(ep_dir, "labels.json"), "w") as f:
        json.dump(dict(
            frames=n,
            event_frames=int(per_frame.sum()),
            intervals=[dict(start_frame=int(a), end_frame=int(b),
                            start_seq=int(seqs[a]), end_seq=int(seqs[b])) for a, b in merged],
        ), f, indent=2)
    return merged


def holdout_split(episode_dirs, fraction=0.15):
    """Frames held out from training for evaluation: the last `fraction` of every episode.

    Returns {ep_dir: range of held-out frame indices}; train on the frames before it and score
    surprise against labels inside it. Scoring the model on frames it trained on would hide
    whether it generalizes. A block at the end (rather than random frames) keeps held-out frames
    away from the near-identical neighbouring frames the model saw in training.
    """
    out = {}
    for ep in sorted(episode_dirs):
        n = len(frame_index(ep)[0])
        out[ep] = range(n - int(round(n * fraction)), n)
    return out
