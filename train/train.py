"""Train the forward model on prepared episodes. Run on the Mac.

    .venv-train/bin/python -m train.train                       # all prepared episodes
    .venv-train/bin/python -m train.train --rounds 12 --name kitchen_v1

Learns to predict frame t+1 from frames t-2..t (no labels involved). Each round is a fixed
number of training steps followed by a check on the held-out frames. Half of every batch
comes from moments with motion, since most household video is a still room.

Writes data/models/<name>/: model.pt (best held-out NLL), last.pt (latest round), config.json,
log.csv
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from train.model import N_FRAMES, ForwardModel

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "cache"
MODELS = PROJECT / "data" / "models"


class Episodes:
    """All prepared episodes in memory, with valid sample positions for train and held-out."""

    def __init__(self, names):
        self.frames, self.train_pos, self.held_pos, self.motion = [], [], [], []
        for e, name in enumerate(names):
            f = np.load(CACHE / name / "frames.npy")
            held = np.load(CACHE / name / "index.npz")["held_out"]
            self.frames.append(f)
            # sample i predicts frame i from frames i-N..i-1; all N+1 frames must share a split
            ok = np.arange(N_FRAMES, len(f))
            same = np.array([held[i - N_FRAMES:i + 1].all() or not held[i - N_FRAMES:i + 1].any()
                             for i in ok])
            ok = ok[same]
            self.train_pos += [(e, i) for i in ok if not held[i]]
            self.held_pos += [(e, i) for i in ok if held[i]]
            diff = np.abs(f[1:].astype(np.int16) - f[:-1].astype(np.int16)).mean(axis=(1, 2))
            self.motion.append(np.concatenate([[0], diff]))
        motion = np.array([self.motion[e][i] for e, i in self.train_pos])
        self.moving = np.flatnonzero(motion > 2 * np.median(motion))   # "something is happening"

    def batch(self, positions, rng=None, augment=False):
        x = np.stack([self.frames[e][i - N_FRAMES:i + 1] for e, i in positions]).astype(np.float32)
        x /= 255.0
        if augment:
            # same random brightness/contrast and flip for all frames of a sample
            gain = rng.uniform(0.6, 1.4, (len(x), 1, 1, 1)).astype(np.float32)
            bias = rng.uniform(-0.1, 0.1, (len(x), 1, 1, 1)).astype(np.float32)
            x = np.clip(x * gain + bias, 0, 1)
            flip = rng.random(len(x)) < 0.5
            x[flip] = x[flip, :, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x[:, :N_FRAMES], x[:, N_FRAMES:]

    def sample_train(self, n, rng):
        k = n // 2
        idx = np.concatenate([rng.integers(0, len(self.train_pos), n - k),
                              rng.choice(self.moving, k)])
        return [self.train_pos[i] for i in idx]


def loss_fn(target, pred, logvar, flow, beta=0.5, smooth_w=0.01):
    """beta-NLL (Seitzer et al. 2022): Gaussian NLL weighted by var^beta so hard pixels aren't
    simply written off as 'uncertain'; plus a small flow smoothness penalty."""
    var = torch.exp(logvar)
    nll = 0.5 * ((target - pred) ** 2 / var + logvar)
    nll = (nll * var.detach() ** beta).mean()
    smooth = (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean() + \
             (flow[..., :, 1:] - flow[..., :, :-1]).abs().mean()
    return nll + smooth_w * smooth


@torch.no_grad()
def evaluate(model, data, batch=64):
    """Held-out loss and prediction error, overall and on the 10% of samples with the most
    motion (where copying the last frame does worst). A still room makes the overall numbers
    nearly identical to copy-last-frame; the motion subset is where the model has to earn it."""
    model.eval()
    nlls, mse, copy = [], [], []
    for k in range(0, len(data.held_pos), batch):
        x, y = data.batch(data.held_pos[k:k + batch])
        pred, logvar, flow = model(x)
        # plain Gaussian NLL: unlike the beta-weighted training loss, it's comparable across
        # rounds, so it's what picks the best checkpoint
        nll = 0.5 * ((y - pred) ** 2 * torch.exp(-logvar) + logvar).mean()
        nlls.append(float(nll) * len(x))
        mse.append(((y - pred) ** 2).mean(dim=(1, 2, 3)).numpy())
        copy.append(((y - x[:, -1:]) ** 2).mean(dim=(1, 2, 3)).numpy())
    model.train()
    mse, copy = np.concatenate(mse), np.concatenate(copy)
    moving = copy >= np.percentile(copy, 90)
    return dict(loss=sum(nlls) / len(mse), mse=mse.mean(), copy_mse=copy.mean(),
                mse_moving=mse[moving].mean(), copy_moving=copy[moving].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", help="prepared episode names (default: all in data/cache)")
    ap.add_argument("--name", default=time.strftime("model_%Y%m%d_%H%M"))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--steps", type=int, default=400, help="training steps per round")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = args.episodes or sorted(p.name for p in CACHE.iterdir() if (p / "frames.npy").exists())
    data = Episodes(names)
    print(f"episodes: {names}\ntrain samples {len(data.train_pos)} "
          f"({len(data.moving)} with motion), held-out samples {len(data.held_pos)}")

    out = MODELS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(vars(args), episodes=names), indent=2))
    model = ForwardModel()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.rounds * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    base = evaluate(model, data)
    print(f"before training: error vs copying the last frame: all {base['mse'] / base['copy_mse']:.0%}, "
          f"moving {base['mse_moving'] / base['copy_moving']:.0%} (lower is better)")
    best = float("inf")
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        log.writerow(["round", "train_loss", "held_nll", "held_mse", "copy_mse",
                      "held_mse_moving", "copy_mse_moving", "minutes"])
        t0 = time.time()
        for r in range(1, args.rounds + 1):
            running = 0.0
            for s in range(args.steps):
                x, y = data.batch(data.sample_train(args.batch, rng), rng, augment=True)
                pred, logvar, flow = model(x)
                loss = loss_fn(y, pred, logvar, flow)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                running += float(loss)
                if (s + 1) % 100 == 0:
                    print(f"  round {r} step {s + 1}/{args.steps}  loss {running / (s + 1):.4f}", flush=True)
            ev = evaluate(model, data)
            mins = (time.time() - t0) / 60
            improved = ev["loss"] < best
            torch.save(model.state_dict(), out / "last.pt")
            if improved:
                best = ev["loss"]
                torch.save(model.state_dict(), out / "model.pt")
            log.writerow([r, running / args.steps, ev["loss"], ev["mse"], ev["copy_mse"],
                          ev["mse_moving"], ev["copy_moving"], round(mins, 1)])
            f.flush()
            print(f"round {r}/{args.rounds}: held-out NLL {ev['loss']:.4f} | error vs copying the "
                  f"last frame: all {ev['mse'] / ev['copy_mse']:.0%}, moving "
                  f"{ev['mse_moving'] / ev['copy_moving']:.0%}{'  * saved' if improved else ''}"
                  f"  [{mins:.0f} min]", flush=True)
    print(f"done. best model: {out / 'model.pt'}")


if __name__ == "__main__":
    main()
