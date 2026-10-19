"""Virtual eye over recorded footage: random pans cut out of the saved full frames, so one
recording yields endless moving-camera training data. Run on the Mac.

    .venv-train/bin/python -m train.virtual --preview          # 30 s preview clip of the busiest stretch
    .venv-train/bin/python -m train.virtual --preview --start 1500 --seconds 30 --seed 3

The episode's full frames are stored at LOG_SIZE (320x180), so the eye works in those pixel
units: a 1/4-size crop (138x104, i.e. the live 552x414 crop scaled by 1/4) and the camera's
focal length scaled by 1/4. Pan/tilt angles therefore mean the same as on the live Pi.
Commands come from eye.actuators.RandomPan; the simulated servo (latency, speed limit,
backlash, jitter) sits between the commands and where the crop actually lands.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from eye.actuators import C922_FOCAL_PX_720P, RandomPan, ServoModel, SweepScan, VirtualActuator
from eye.episode import chunk_paths, frame_index
from eye.preprocess import LOG_SIZE, MODEL_SIZE
from train.video import open_writer

PROJECT = Path(__file__).resolve().parent.parent
LIVE_CROP = (552, 414)                                   # eye/run.py default, in 1280x720 pixels
COMMAND_DELAY_S = 0.015      # live, a frame's command is sent while it's processed, after capture
SCALE = LOG_SIZE[0] / 1280                               # 1/4
CROP = (int(round(LIVE_CROP[0] * SCALE)), int(round(LIVE_CROP[1] * SCALE)))


def load_frames(ep, start, n):
    """Logged full frames [start, start + n) as one (n, H, W) uint8 array."""
    out, offset = [], 0
    for path in chunk_paths(ep):
        with np.load(path) as d:
            f = d["frames"]
        lo, hi = max(start, offset), min(start + n, offset + len(f))
        if lo < hi:
            out.append(f[lo - offset:hi - offset])
        offset += len(f)
        if offset >= start + n:
            break
    return np.concatenate(out)


def make_eye(seed=None, servo_noise=0.05, motion="random"):
    """motion: "random" (training paths, eye.actuators.RandomPan) or "sweep" (the live demo's
    10 s left-right sweep, eye.actuators.SweepScan)."""
    act = VirtualActuator(full_size=LOG_SIZE, crop_size=CROP, focal_px=C922_FOCAL_PX_720P * SCALE,
                          servo=ServoModel(noise=servo_noise, seed=seed))
    if motion == "sweep":
        return act, SweepScan(act.pan_limit)
    return act, RandomPan(act.pan_limit, act.tilt_limit, seed=seed)


def simulate_track(times, seed=None, motion="random"):
    """Virtual eye over frame times (n,), one step per recorded frame (30 Hz). Returns crop
    centers (n, 2) in LOG_SIZE pixels and motor (n, 4) degrees as (commanded pan, commanded
    tilt, actual pan, actual tilt), with commanded = the most recent command sent. Each frame's
    state is read before that frame's command goes out, matching the live loop's timing."""
    act, mover = make_eye(seed, motion=motion)
    centers = np.empty((len(times), 2))
    motor = np.empty((len(times), 4))
    for k, t in enumerate(times):
        s = act.state(t)
        centers[k] = act.crop_center(s)
        motor[k] = (s.pan_target, s.tilt_target, s.pan, s.tilt)
        act.command(*mover.step(t), t=t + COMMAND_DELAY_S)
    return centers, motor


def crop_input(full_img, center):
    """The virtual eye's model input: the crop at `center`, nothing else, resized to MODEL_SIZE."""
    crop = cv2.getRectSubPix(full_img, CROP, (float(center[0]), float(center[1])))
    return cv2.resize(crop, MODEL_SIZE, interpolation=cv2.INTER_AREA)


def simulate(frames, times, seed=None):
    """Run the virtual eye over frames (n, H, W) taken at times (n,). Returns model inputs
    (n, 96, 128) uint8, motor (n, 4) degrees, and the crops' top-left corners (for drawing)."""
    centers, motor = simulate_track(times, seed)
    inputs = np.stack([crop_input(img, c) for img, c in zip(frames, centers)])
    rects = [(cx - CROP[0] / 2, cy - CROP[1] / 2) for cx, cy in centers]
    return inputs, motor, rects


def busiest_start(ep, seconds, fps=30):
    """First frame of the most active stretch (by frame difference) in the first 85%."""
    cache = PROJECT / "data" / "cache" / ep.name
    f = np.load(cache / "frames.npy", mmap_mode="r")
    ix = np.load(cache / "index.npz")
    act = np.abs(np.diff(f.astype(np.int16), axis=0)).mean(axis=(1, 2))
    act[ix["held_out"][1:]] = 0
    win = int(seconds * fps / int(ix["stride"]))
    k = int(np.argmax(np.convolve(act, np.ones(win), "valid")))
    return int(ix["frame_idx"][k])


def render_preview(frames, times, inputs, motor, rects, path, fps):
    W, H = 640, 360
    trail = []
    writer = open_writer(path, fps, (W + 480, H + 150))
    sx = W / LOG_SIZE[0]
    hist = []
    for i, (img, x_in, (cmd_pan, cmd_tilt, pan, tilt), (x0, y0)) in enumerate(zip(frames, inputs, motor, rects)):
        full = cv2.cvtColor(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        c = (int((x0 + CROP[0] / 2) * sx), int((y0 + CROP[1] / 2) * sx))
        trail.append(c)
        for a, b in zip(trail[-150:], trail[-149:]):
            cv2.line(full, a, b, (0, 200, 255), 1, cv2.LINE_AA)
        cv2.rectangle(full, (int(x0 * sx), int(y0 * sx)), (int((x0 + CROP[0]) * sx), int((y0 + CROP[1]) * sx)),
                      (0, 255, 0), 2)
        cv2.putText(full, "recorded full frame + virtual eye", (8, 22), 0, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        eye = cv2.cvtColor(cv2.resize(x_in, (480, 360), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
        cv2.putText(eye, f"model input {MODEL_SIZE[0]}x{MODEL_SIZE[1]} (the crop only)", (8, 22), 0, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        top = np.hstack([full, eye])

        hist.append((times[i], cmd_pan, pan, cmd_tilt, tilt))
        bar = np.full((150, W + 480, 3), 24, np.uint8)
        speed = 0.0
        if i:
            p = np.array([[h[2], h[4]] for h in hist[-6:]])
            speed = np.linalg.norm(p[-1] - p[0]) / max(times[i] - hist[-6:][0][0], 1e-3)
        cv2.putText(bar, f"t {times[i] - times[0]:5.1f} s   pan {pan:+6.2f} (cmd {cmd_pan:+6.2f})   "
                    f"tilt {tilt:+5.2f} (cmd {cmd_tilt:+5.2f})   speed {speed:4.1f} deg/s",
                    (10, 22), 0, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        left, right, top_y, bot = 10, W + 470, 34, 144
        cv2.rectangle(bar, (left, top_y), (right, bot), (60, 60, 60), 1)
        t_end, span = times[i], 10.0
        for k, (color, lim) in enumerate((((80, 170, 255), 22.0), ((255, 200, 80), 10.0))):
            pts = [(int(right - (t_end - h[0]) / span * (right - left)),
                    int((top_y + bot) / 2 - h[2 + 2 * k] / lim * (bot - top_y) / 2 * 0.95))
                   for h in hist if t_end - h[0] <= span]
            if len(pts) > 1:
                cv2.polylines(bar, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(bar, "pan", (left + 6, top_y + 16), 0, 0.45, (80, 170, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, "tilt", (left + 50, top_y + 16), 0, 0.45, (255, 200, 80), 1, cv2.LINE_AA)
        cv2.putText(bar, "last 10 s", (right - 80, top_y + 16), 0, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
        writer.write(np.vstack([top, bar]))
    writer.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="recordings/kitchen_20261018_102345")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--start", type=float, help="seconds into the recording (default: busiest stretch)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ep = (PROJECT / args.episode).resolve()
    seqs, t = frame_index(ep)
    fps = 30
    n = int(args.seconds * fps)
    start = int(args.start * fps) if args.start is not None else busiest_start(ep, args.seconds)
    frames = load_frames(ep, start, n)
    times = t[start:start + len(frames)]
    inputs, motor, rects = simulate(frames, times, seed=args.seed)
    pans = motor[:, 2:]
    v = np.linalg.norm(np.diff(pans, axis=0), axis=1) / np.diff(times)
    print(f"{ep.name}: {len(frames)} frames from {start / fps / 60:.1f} min; crop {CROP} of {LOG_SIZE}; "
          f"pan {pans[:, 0].min():+.1f}..{pans[:, 0].max():+.1f}, tilt {pans[:, 1].min():+.1f}..{pans[:, 1].max():+.1f}, "
          f"median speed {np.median(v):.1f} deg/s")
    if args.preview:
        out = PROJECT / "data" / "previews"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"random_pan_preview_seed{args.seed}.mp4"
        render_preview(frames, times, inputs, motor, rects, path, fps)
        print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
