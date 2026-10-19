"""Turn recorded episodes into model-ready frame stacks. Run on the Mac.

    .venv-train/bin/python -m train.prepare                     # every complete episode
    .venv-train/bin/python -m train.prepare recordings/kitchen_20261018_102345

For each episode writes data/cache/<episode>/:
    frames.npy   (N, 96, 128) uint8 model inputs (see eye.preprocess.static_model_input)
    full.npy     (N, 180, 320) uint8 the recorded full frames, for cutting virtual-eye crops
                 during training (train/virtual.py)
    index.npz    frame_idx (N,) which logged frame each one is, t (N,) seconds from start,
                 held_out (N,) bool: True for the last 15% (eye.episode.holdout_split)

Frames are subsampled by --stride (default 2: 30 fps -> 15 fps) so each prediction step spans
more motion. Episodes without meta.json (interrupted or partly deleted) are skipped.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

from eye.episode import chunk_paths, frame_index, holdout_split, load_meta
from eye.preprocess import LOG_SIZE, MODEL_SIZE, static_model_input

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "cache"


def complete_episodes(paths):
    out = []
    for p in paths:
        if not (p / "meta.json").exists():
            print(f"skipping {p.name}: no meta.json (interrupted or partly deleted)")
            continue
        out.append(p)
    return out


def prepare(ep, stride, held_out_range):
    out = CACHE / ep.name
    seqs, t = frame_index(ep)
    keep = np.arange(0, len(seqs), stride)
    frames = np.empty((len(keep), MODEL_SIZE[1], MODEL_SIZE[0]), np.uint8)
    full = np.empty((len(keep), LOG_SIZE[1], LOG_SIZE[0]), np.uint8)
    keep_set = {int(i): k for k, i in enumerate(keep)}
    offset = 0
    for path in chunk_paths(ep):
        with np.load(path) as d:
            chunk = d["frames"]
        for j in range(len(chunk)):
            k = keep_set.get(offset + j)
            if k is not None:
                frames[k] = static_model_input(chunk[j])
                full[k] = chunk[j]
        offset += len(chunk)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "frames.npy", frames)
    np.save(out / "full.npy", full)
    held_out = (keep >= held_out_range.start) & (keep < held_out_range.stop)
    np.savez(out / "index.npz", frame_idx=keep, t=t[keep] - t[0], held_out=held_out, stride=stride)
    fps = load_meta(ep).get("fps", 30) / stride
    print(f"{ep.name}: {len(keep)} frames at {fps:.0f} fps "
          f"({(~held_out).sum()} train, {held_out.sum()} held out) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*", help="episode folders (default: all in recordings/)")
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    paths = [Path(e).resolve() for e in args.episodes] or \
        sorted(p for p in (PROJECT / "recordings").iterdir() if p.is_dir())
    eps = complete_episodes(paths)
    if not eps:
        sys.exit("no complete episodes found")
    splits = holdout_split([str(e) for e in eps])
    for ep in eps:
        prepare(ep, args.stride, splits[str(ep)])


if __name__ == "__main__":
    main()
