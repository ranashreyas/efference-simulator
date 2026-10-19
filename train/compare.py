"""Moving-eye check: run models over the held-out footage with the virtual eye following one
fixed path, and compare their surprise. Run on the Mac.

    .venv-train/bin/python -m train.compare --models kitchen_v2 --motion sweep --video
    .venv-train/bin/python -m train.compare --models kitchen_v2 kitchen_v3 --motion random

Each model runs twice: with its motor input, and with it blanked (to see what the efference
copy contributes). Reports the mean surprise score while the eye is moving, while it's paused,
and around turnarounds (when it starts, stops or reverses), next to frame difference.

Writes data/compare/<motion>/: timeline.png, metrics.txt, and with --video compare.mp4
(the eye's view, each model's surprise, and frame difference side by side).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from eye.episode import frame_index
from eye.motor import ACTUAL_NOISE_DEG, STEPS, blank, encode_rows
from train.evaluate import top_mean
from train.model import N_FRAMES, load_model, surprise_map
from train.video import open_writer
from train.virtual import crop_input, simulate_track

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "cache"
SURPRISE_VMAX, DIFF_VMAX = 50.0, 0.25          # same color scales as the live viewer


def heat(m, vmax):
    return cv2.applyColorMap((np.sqrt(np.clip(m / vmax, 0, 1)) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--episode", default="kitchen_20261018_102345")
    ap.add_argument("--motion", choices=["sweep", "random"], default="sweep")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    full = np.load(CACHE / args.episode / "full.npy", mmap_mode="r")
    index = np.load(CACHE / args.episode / "index.npz")
    frame_idx, held = index["frame_idx"], np.flatnonzero(index["held_out"])
    pos = held[held >= held[0] + N_FRAMES]
    _, times30 = frame_index(PROJECT / "recordings" / args.episode)
    centers, mot30 = simulate_track(times30, seed=args.seed, motion=args.motion)

    models = {name: load_model(PROJECT / "data" / "models" / name / "model.pt") for name in args.models}
    runs = [(name, blind) for name in args.models for blind in (False, True)]
    label = {(n, b): n + (" (motor blanked)" if b else "") for n, b in runs}

    rng = np.random.default_rng(0)
    scores = {r: [] for r in runs}
    diff_scores, pan = [], []
    out = PROJECT / "data" / "compare" / args.motion
    out.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.video:
        writer = open_writer(out / "compare.mp4", 15, (320 * (2 + len(runs)), 240 + 40))
    with torch.no_grad():
        for k in range(0, len(pos), 64):
            p = pos[k:k + 64]
            x = np.stack([[crop_input(full[i], centers[fi]) for i, fi in zip(range(j - N_FRAMES, j + 1),
                                                                            frame_idx[j - N_FRAMES:j + 1])]
                          for j in p]).astype(np.float32) / 255
            steps = [np.clip(np.arange(frame_idx[j] - STEPS + 1, frame_idx[j] + 1), 0, None) for j in p]
            motor = encode_rows(np.stack([mot30[s] for s in steps]), ACTUAL_NOISE_DEG, rng)
            x = torch.from_numpy(x)
            inp, y = x[:, :N_FRAMES], x[:, N_FRAMES:]
            maps = {}
            for name, blind in runs:
                mot = blank(len(p)) if blind else motor
                pred, logvar, _ = models[name](inp, torch.from_numpy(mot))
                s_map = surprise_map(y, pred, logvar)
                scores[(name, blind)].append(top_mean(s_map).numpy())
                maps[(name, blind)] = s_map[:, 0].numpy()
            d_map = ((y - inp[:, -1:]) ** 2)
            diff_scores.append(top_mean(d_map).numpy())
            pan.append(mot30[frame_idx[p], 2])
            if writer is not None:
                for j in range(len(p)):
                    tiles = [cv2.cvtColor((y[j, 0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)]
                    tiles += [heat(maps[r][j], SURPRISE_VMAX) for r in runs]
                    tiles.append(heat(d_map[j, 0].numpy(), DIFF_VMAX))
                    names = ["eye view (crop)"] + [label[r] + " surprise" for r in runs] + ["frame difference"]
                    row = []
                    for img, n in zip(tiles, names):
                        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_NEAREST)
                        cv2.putText(img, n, (6, 18), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                        row.append(img)
                    bar = np.full((40, 320 * len(row), 3), 24, np.uint8)
                    cv2.putText(bar, f"pan {mot30[frame_idx[p[j]], 2]:+5.1f} deg   cmd {mot30[frame_idx[p[j]], 0]:+5.1f}",
                                (8, 26), 0, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
                    writer.write(np.vstack([np.hstack(row), bar]))
    if writer is not None:
        writer.release()

    scores = {r: np.concatenate(v) for r, v in scores.items()}
    diff_scores, pan = np.concatenate(diff_scores), np.concatenate(pan)
    t = times30[frame_idx[pos]]
    speed = np.abs(np.gradient(pan, t))
    moving, paused = speed > 1.0, speed < 0.3
    accel = np.abs(np.gradient(np.gradient(pan, t), t))
    turn = accel > np.percentile(accel, 95)                    # starts, stops and reversals
    lines = [f"{args.motion} path over held-out footage: {len(t)} frames, moving {moving.mean():.0%}, "
             f"paused {paused.mean():.0%}, turnarounds {turn.mean():.0%}",
             f"{'model':32s} {'moving':>8s} {'paused':>8s} {'turnaround':>11s} {'all':>8s}"]
    for r in runs:
        s = scores[r]
        lines.append(f"{label[r]:32s} {s[moving].mean():8.2f} {s[paused].mean():8.2f} {s[turn].mean():11.2f} {s.mean():8.2f}")
    lines.append(f"{'frame difference (x1000)':32s} {1000 * diff_scores[moving].mean():8.2f} "
                 f"{1000 * diff_scores[paused].mean():8.2f} {1000 * diff_scores[turn].mean():11.2f} "
                 f"{1000 * diff_scores.mean():8.2f}")
    (out / "metrics.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # timeline: each model's surprise (same scale) and the pan angle
    W, H = 1400, 120 * (len(runs) + 1)
    img = np.full((H, W, 3), 24, np.uint8)
    x = ((t - t[0]) / (t[-1] - t[0]) * (W - 1)).astype(int)
    vmax = max(np.percentile(s, 99.5) for s in scores.values())
    colors = [(80, 200, 255), (120, 255, 120), (200, 160, 255), (255, 200, 80)]
    for row, r in enumerate(runs):
        top, bot = row * 120 + 20, row * 120 + 112
        z = np.clip(scores[r] / vmax, 0, 1)
        cv2.polylines(img, [np.stack([x, (bot - z * (bot - top)).astype(int)], 1).astype(np.int32)], False,
                      colors[row % 4], 1, cv2.LINE_AA)
        cv2.putText(img, f"{label[r]} surprise (shared scale)", (8, top + 4), 0, 0.5, colors[row % 4], 1, cv2.LINE_AA)
    top, bot = len(runs) * 120 + 20, H - 8
    lim = max(1e-6, np.abs(pan).max())
    cv2.polylines(img, [np.stack([x, ((top + bot) / 2 - pan / lim * (bot - top) / 2).astype(int)], 1).astype(np.int32)],
                  False, (255, 200, 80), 1, cv2.LINE_AA)
    cv2.putText(img, "pan angle", (8, top + 4), 0, 0.5, (255, 200, 80), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "timeline.png"), img)
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
