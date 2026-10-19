"""Static-eye check: run a trained model over the held-out part of each episode, as the fixed
camera sees it, and compare its surprise score with a frame-differencing baseline. Scores
labeled events if there are any. Run on the Mac. (For the moving eye, see train/compare.py.)

    .venv-train/bin/python -m train.evaluate --model kitchen_v2            # scores + timeline
    .venv-train/bin/python -m train.evaluate --model kitchen_v2 --video    # + side-by-side video

Per frame, three event scores (each = mean of the top 1% of a blurred per-pixel map, then
smoothed over ~0.3 s):
    surprise    model error divided by the model's own expected error   <- the method
    error       model error alone (no uncertainty)                      <- ablation
    frame diff  change since the previous frame                         <- baseline

If the episode has labels (recordings/<ep>/labels.npy), prints frame-level AUROC for each:
the chance that a labeled event frame scores higher than an ordinary frame (0.5 = guessing).
Without labels, it lists the top moments so you can check what triggered them.

Writes data/models/<model>/eval/<episode>/: scores.npz, timeline.png, top_moments.txt, eval.mp4
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from eye.motor import still
from train.model import N_FRAMES, load_model, surprise_map
from train.video import open_writer

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "cache"
MODELS = PROJECT / "data" / "models"
RECORDINGS = PROJECT / "recordings"
METHODS = ["surprise", "error", "frame diff"]
COLORS = {"surprise": (80, 200, 255), "error": (200, 160, 90), "frame diff": (150, 150, 150)}


def top_mean(maps, frac=0.01):
    """Mean of the top `frac` of each map after a light blur (one value per frame)."""
    b, _, h, w = maps.shape
    k = max(1, int(h * w * frac))
    blurred = torch.nn.functional.avg_pool2d(maps, 5, 1, 2)
    return blurred.view(b, -1).topk(k, dim=1).values.mean(1)


def smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same") if n > 1 else x


def auroc(scores, labels):
    """Probability a positive scores above a negative (ties count half). Mann-Whitney form."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    s = np.sort(scores)
    for v in np.unique(s[1:][np.diff(s) == 0]):                  # tied scores share their mean rank
        ranks[scores == v] = ranks[scores == v].mean()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fmt(t):
    t = round(float(t), 1)
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def heat(m, vmax):
    m = np.sqrt(np.clip(m / vmax, 0, 1))              # square root so faint surprise stays visible
    return cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


@torch.no_grad()
def run_episode(model, name, out, fps, make_video, batch=64):
    frames = np.load(CACHE / name / "frames.npy", mmap_mode="r")
    index = np.load(CACHE / name / "index.npz")
    held = np.flatnonzero(index["held_out"])
    pos = held[held >= held[0] + N_FRAMES]            # need N_FRAMES held-out frames of history
    scores = {m: [] for m in METHODS}
    writer = None
    if make_video:
        writer = open_writer(out / "eval.mp4", fps, (768, 640))
    vmax_s = vmax_d = None
    for k in range(0, len(pos), batch):
        p = pos[k:k + batch]
        x = torch.from_numpy(np.stack([frames[i - N_FRAMES:i + 1] for i in p]).astype(np.float32) / 255)
        inp, y = x[:, :N_FRAMES], x[:, N_FRAMES:]
        pred, logvar, _ = model(inp, torch.from_numpy(still(len(p))))     # what the static eye sends
        s_map = surprise_map(y, pred, logvar)
        e_map = (y - pred) ** 2
        d_map = (y - inp[:, -1:]) ** 2
        for m, mp in zip(METHODS, (s_map, e_map, d_map)):
            scores[m].append(top_mean(mp).numpy())
        if writer is not None:
            if vmax_s is None:                        # fixed color scale for the whole video
                vmax_s = float(np.percentile(s_map.numpy(), 99.95)) * 2
                vmax_d = float(np.percentile(d_map.numpy(), 99.95)) * 2
            for j, i in enumerate(p):
                writer.write(video_frame(y[j, 0].numpy(), pred[j, 0].numpy(), s_map[j, 0].numpy(),
                                         d_map[j, 0].numpy(), vmax_s, vmax_d, index["t"][i]))
    if writer is not None:
        writer.release()
    n_smooth = max(1, int(round(0.3 * fps)))
    scores = {m: smooth(np.concatenate(v), n_smooth) for m, v in scores.items()}
    return pos, index["frame_idx"][pos], index["t"][pos], scores


def video_frame(actual, pred, s_map, d_map, vmax_s, vmax_d, t):
    def gray(a):
        return cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tiles = [gray(actual), gray(pred), heat(s_map, vmax_s), heat(d_map, vmax_d)]
    titles = ["actual", "model prediction", "model surprise", "baseline: frame difference"]
    big = []
    for img, title in zip(tiles, titles):
        img = cv2.resize(img, (384, 288), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        big.append(img)
    canvas = np.vstack([np.hstack(big[:2]), np.hstack(big[2:])])
    bar = np.full((64, 768, 3), 20, np.uint8)
    cv2.putText(bar, f"recording time {fmt(t)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([canvas, bar])


def top_moments(t, score, n=10, gap_s=3.0):
    """The n highest-scoring moments at least gap_s apart."""
    order, picked = np.argsort(score)[::-1], []
    for i in order:
        if all(abs(t[i] - t[j]) >= gap_s for j in picked):
            picked.append(i)
        if len(picked) == n:
            break
    return picked


def timeline(t, scores, labels, path):
    w, h = 1200, 420
    img = np.full((h, w, 3), 24, np.uint8)
    x = ((t - t[0]) / max(1e-6, t[-1] - t[0]) * (w - 1)).astype(int)
    if labels is not None:
        for i in np.flatnonzero(labels):
            cv2.line(img, (x[i], 0), (x[i], h - 1), (40, 40, 110), 1)
    rows = len(METHODS)
    for r, m in enumerate(METHODS):
        s = scores[m]
        top, bot = 30 + r * (h - 40) // rows, 20 + (r + 1) * (h - 40) // rows
        z = np.clip((s - np.median(s)) / (np.percentile(s, 99.5) - np.median(s) + 1e-12), 0, 1)
        pts = np.stack([x, (bot - z * (bot - top - 18)).astype(int)], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, COLORS[m], 1, cv2.LINE_AA)
        cv2.putText(img, m, (8, top + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[m], 1, cv2.LINE_AA)
    cv2.putText(img, f"held-out part: {fmt(t[0])} - {fmt(t[-1])}" +
                ("   (red = labeled events)" if labels is not None else "   (no labels)"),
                (w - 520, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="name under data/models/")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    mdir = MODELS / args.model
    config = json.loads((mdir / "config.json").read_text())
    model = load_model(mdir / "model.pt")
    for name in config["episodes"]:
        out = mdir / "eval" / name
        out.mkdir(parents=True, exist_ok=True)
        stride = int(np.load(CACHE / name / "index.npz")["stride"])
        meta = json.loads((RECORDINGS / name / "meta.json").read_text())
        fps = meta.get("fps", 30) / stride
        pos, frame_idx, t, scores = run_episode(model, name, out, fps, args.video)

        labels = None
        lab_path = RECORDINGS / name / "labels.npy"
        if lab_path.exists():
            labels = np.load(lab_path)[frame_idx].astype(bool)
        np.savez(out / "scores.npz", t=t, frame_idx=frame_idx, labels=labels if labels is not None else [],
                 **{m.replace(" ", "_"): s for m, s in scores.items()})
        timeline(t, scores, labels, out / "timeline.png")

        print(f"\n{name}: held-out {fmt(t[0])}-{fmt(t[-1])}, {len(t)} frames at {fps:.0f} fps")
        if labels is not None and labels.any():
            print(f"labeled event frames: {labels.sum()} of {len(labels)}")
            for m in METHODS:
                print(f"  AUROC {m:11s} {auroc(scores[m], labels):.3f}")
        else:
            print("no event labels in the held-out part, so no AUROC; see top moments below")
        lines = []
        for m in ("surprise", "frame diff"):
            lines.append(f"top moments by {m} (recording time):")
            for i in top_moments(t, scores[m]):
                lines.append(f"  {fmt(t[i])}   score {scores[m][i]:.4g}")
        (out / "top_moments.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        print(f"outputs in {out}")


if __name__ == "__main__":
    main()
