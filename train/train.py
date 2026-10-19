"""Train the forward model on prepared episodes. Run on the Mac.

    .venv-train/bin/python -m train.train --name kitchen_v2                        # from scratch
    .venv-train/bin/python -m train.train --name kitchen_v3 --init kitchen_v2      # continue from a model

Learns to predict frame t from frames 1/15, 2/15, 3/15 s earlier plus the motor history (no
labels involved). Each batch mixes two kinds of samples:
    static   the still-camera view (eye.preprocess.static_model_input), motor input "known still"
    virtual  a virtual eye panning over the same footage (train/virtual.py): only the 1/4-size
             crop goes in, plus the motor history (commanded + noisy actual angles). A fresh
             random path is simulated every round, so the same video never pans the same way twice.
Motor input is blanked (marked unavailable) on --motor-dropout of virtual samples, so the model
also works without it and we can measure how much it helps (efference gain). Half the static samples and a quarter of
the virtual ones come from moments with motion, since most household video is a still room.

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

from eye.episode import frame_index
from eye.motor import ACTUAL_NOISE_DEG, PAN_INDEX, STEPS, encode_rows, still
from train.evaluate import top_mean
from train.model import N_FRAMES, ForwardModel, surprise_map
from train.virtual import crop_input, simulate_track

PROJECT = Path(__file__).resolve().parent.parent
CACHE = PROJECT / "data" / "cache"
MODELS = PROJECT / "data" / "models"
RECORDINGS = PROJECT / "recordings"
EVAL_TRACK_SEED = 12345           # held-out virtual pans use one fixed path, so rounds compare fairly


class Episodes:
    """Prepared episodes: static model inputs and full frames (memory-mapped), sample positions
    for train and held-out, and the current virtual-eye tracks."""

    def __init__(self, names):
        self.frames, self.full, self.frame_idx, self.times30 = [], [], [], []
        self.train_pos, self.held_pos, motion = [], [], []
        for e, name in enumerate(names):
            f = np.load(CACHE / name / "frames.npy")
            index = np.load(CACHE / name / "index.npz")
            held = index["held_out"]
            self.frames.append(f)
            self.full.append(np.load(CACHE / name / "full.npy", mmap_mode="r"))
            self.frame_idx.append(index["frame_idx"])
            self.times30.append(frame_index(RECORDINGS / name)[1])
            # sample i predicts frame i from frames i-N..i-1; all N+1 frames must share a split
            ok = np.arange(N_FRAMES, len(f))
            same = np.array([held[i - N_FRAMES:i + 1].all() or not held[i - N_FRAMES:i + 1].any()
                             for i in ok])
            ok = ok[same]
            self.train_pos += [(e, i) for i in ok if not held[i]]
            self.held_pos += [(e, i) for i in ok if held[i]]
            diff = np.abs(f[1:].astype(np.int16) - f[:-1].astype(np.int16)).mean(axis=(1, 2))
            motion.append(np.concatenate([[0], diff]))
        m = np.array([motion[e][i] for e, i in self.train_pos])
        self.moving = np.flatnonzero(m > 2 * np.median(m))     # "something is happening"
        self.eval_tracks = self.make_tracks(EVAL_TRACK_SEED)
        self.tracks = None

    def make_tracks(self, seed):
        return [simulate_track(t, seed=seed + e) for e, t in enumerate(self.times30)]

    def static_batch(self, positions):
        x = np.stack([self.frames[e][i - N_FRAMES:i + 1] for e, i in positions]).astype(np.float32)
        return x / 255.0, still(len(x))

    def virtual_batch(self, positions, tracks, rng):
        xs, motor = [], []
        for e, i in positions:
            centers, mot30 = tracks[e]
            fi = self.frame_idx[e][i - N_FRAMES:i + 1]                  # logged (30 Hz) frame numbers
            xs.append([crop_input(self.full[e][k], centers[j]) for k, j in zip(range(i - N_FRAMES, i + 1), fi)])
            steps = np.clip(np.arange(fi[-1] - STEPS + 1, fi[-1] + 1), 0, None)
            motor.append(mot30[steps])
        x = np.asarray(xs, np.float32) / 255.0
        return x, encode_rows(np.asarray(motor), ACTUAL_NOISE_DEG, rng)

    def train_batch(self, n, rng, static_frac, motor_dropout):
        n_static = int(round(n * static_frac))
        n_virtual = n - n_static
        parts = []
        if n_static:
            k = n_static // 2
            idx = np.concatenate([rng.integers(0, len(self.train_pos), n_static - k), rng.choice(self.moving, k)])
            parts.append(self.static_batch([self.train_pos[i] for i in idx]))
        if n_virtual:
            k = n_virtual // 4
            idx = np.concatenate([rng.integers(0, len(self.train_pos), n_virtual - k), rng.choice(self.moving, k)])
            x, m = self.virtual_batch([self.train_pos[i] for i in idx], self.tracks, rng)
            m[rng.random(len(m)) < motor_dropout] = 0
            parts.append((x, m))
        x = np.concatenate([p[0] for p in parts])
        m = np.concatenate([p[1] for p in parts])
        # augmentation: same brightness/contrast for all frames of a sample; mirror flips also
        # flip the sign of pan in the motor history so left/right stay consistent
        gain = rng.uniform(0.6, 1.4, (len(x), 1, 1, 1)).astype(np.float32)
        bias = rng.uniform(-0.1, 0.1, (len(x), 1, 1, 1)).astype(np.float32)
        x = np.clip(x * gain + bias, 0, 1)
        flip = rng.random(len(x)) < 0.5
        x[flip] = x[flip, :, :, ::-1]
        m[np.ix_(flip, PAN_INDEX)] *= -1
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x[:, :N_FRAMES], x[:, N_FRAMES:], torch.from_numpy(m)


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
def evaluate(model, data, every=2, batch=64):
    """Held-out metrics on every `every`-th held-out sample, for the static view and for the
    virtual eye panning along a fixed path:
        nll        plain Gaussian NLL (comparable across rounds; picks the best checkpoint)
        err        prediction error relative to copying the last frame, on the 10% of samples
                   with the most change (lower is better)
        surprise   mean per-frame surprise score (top 1% of the surprise map)
        eff_gain   virtual only: error with the motor input blanked / error with it (>1 = helps)"""
    model.eval()
    pos = data.held_pos[::every]
    rng = np.random.default_rng(0)
    out = {}
    for kind in ("static", "virtual"):
        nll, mse, copy, sur, mse_blind = [], [], [], [], []
        for k in range(0, len(pos), batch):
            p = pos[k:k + batch]
            x, m = data.static_batch(p) if kind == "static" else data.virtual_batch(p, data.eval_tracks, rng)
            x, m = torch.from_numpy(x), torch.from_numpy(m)
            inp, y = x[:, :N_FRAMES], x[:, N_FRAMES:]
            pred, logvar, _ = model(inp, m)
            nll.append(float(0.5 * ((y - pred) ** 2 * torch.exp(-logvar) + logvar).mean()) * len(x))
            mse.append(((y - pred) ** 2).mean(dim=(1, 2, 3)).numpy())
            copy.append(((y - inp[:, -1:]) ** 2).mean(dim=(1, 2, 3)).numpy())
            sur.append(top_mean(surprise_map(y, pred, logvar)).numpy())
            if kind == "virtual":
                pred0, _, _ = model(inp, torch.zeros_like(m))
                mse_blind.append(((y - pred0) ** 2).mean(dim=(1, 2, 3)).numpy())
        mse, copy = np.concatenate(mse), np.concatenate(copy)
        top = copy >= np.percentile(copy, 90)
        out[kind] = dict(nll=sum(nll) / len(mse), err=mse[top].mean() / copy[top].mean(),
                         surprise=float(np.concatenate(sur).mean()))
        if kind == "virtual":
            out[kind]["eff_gain"] = float(np.concatenate(mse_blind).mean() / mse.mean())
    model.train()
    return out


def describe(ev):
    s, v = ev["static"], ev["virtual"]
    return (f"static err {s['err']:.0%} surprise {s['surprise']:.3g} | panning err {v['err']:.0%} "
            f"surprise {v['surprise']:.3g} ({v['surprise'] / s['surprise']:.1f}x static) | "
            f"efference gain {v['eff_gain']:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="*", help="prepared episode names (default: all in data/cache)")
    ap.add_argument("--name", default=time.strftime("model_%Y%m%d_%H%M"))
    ap.add_argument("--init", help="start from this model's weights (name under data/models/)")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400, help="training steps per round")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--static-frac", type=float, default=0.25,
                    help="share of still-camera samples (they keep the static eye accurate)")
    ap.add_argument("--motor-dropout", type=float, default=0.2, help="share of virtual samples with motor blanked")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = args.episodes or sorted(p.name for p in CACHE.iterdir() if (p / "full.npy").exists())
    data = Episodes(names)
    print(f"episodes: {names}\ntrain samples {len(data.train_pos)} ({len(data.moving)} with motion), "
          f"held-out samples {len(data.held_pos)}; {args.static_frac:.0%} static, "
          f"{1 - args.static_frac:.0%} virtual pans", flush=True)

    out = MODELS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(vars(args), episodes=names), indent=2))
    model = ForwardModel()
    if args.init:
        # copy every weight whose shape still matches (the motor input layer may have changed size)
        own = model.state_dict()
        init = torch.load(MODELS / args.init / "model.pt")
        keep = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(keep, strict=False)
        skipped = sorted(set(own) - set(keep))
        print(f"initialized from {args.init}" + (f"; new (not copied): {skipped}" if skipped else ""))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total = args.rounds * args.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total,
                                                pct_start=max(0.05, 3 / total))   # >= 3 warm-up steps

    base = evaluate(model, data)
    print(f"before training: {describe(base)}", flush=True)
    best = float("inf")
    with open(out / "log.csv", "w", newline="") as f:
        log = csv.writer(f)
        cols = ["round", "train_loss"] + [f"{k}_{m}" for k in ("static", "virtual")
                                          for m in ("nll", "err", "surprise")] + ["eff_gain", "minutes"]
        log.writerow(cols)
        log.writerow([0, ""] + [base[k][m] for k in ("static", "virtual") for m in ("nll", "err", "surprise")]
                     + [base["virtual"]["eff_gain"], 0])
        t0 = time.time()
        for r in range(1, args.rounds + 1):
            data.tracks = data.make_tracks(args.seed * 1000 + r)       # a new random path every round
            running = 0.0
            for s in range(args.steps):
                x, y, m = data.train_batch(args.batch, rng, args.static_frac, args.motor_dropout)
                pred, logvar, flow = model(x, m)
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
            score = (ev["static"]["nll"] + ev["virtual"]["nll"]) / 2
            improved = score < best
            torch.save(model.state_dict(), out / "last.pt")
            if improved:
                best = score
                torch.save(model.state_dict(), out / "model.pt")
            log.writerow([r, running / args.steps] + [ev[k][m] for k in ("static", "virtual")
                                                      for m in ("nll", "err", "surprise")]
                         + [ev["virtual"]["eff_gain"], round(mins, 1)])
            f.flush()
            print(f"round {r}/{args.rounds}: {describe(ev)}{'  * saved' if improved else ''}  [{mins:.0f} min]",
                  flush=True)
    print(f"done. best model: {out / 'model.pt'}")


if __name__ == "__main__":
    main()
