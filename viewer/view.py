"""Live debug viewer (runs on the Mac). Subscribes to the Pi's debug stream.

    .venv/bin/python -m viewer.view                    # connects to raspberrypi.local:5555
    .venv/bin/python -m viewer.view --host 192.168.12.104
    .venv/bin/python -m viewer.view --headless --count 90   # no window; print stats and exit

Layout:
    [ full camera frame + eye window ] [ what the eye sees ] [ stats + motor trace ]
    [ model input (128x96)           ] [ prediction        ] [ surprise            ]
Keys:
    s      save a snapshot
    q      quit
"""
import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eye.stream import DebugSubscriber  # noqa: E402

PANE = (480, 360)
FULL = (640, 360)
INFO = (320, 360)
BG = (24, 24, 24)
FG = (230, 230, 230)
DIM = (130, 130, 130)
TARGET_COLOR = (80, 170, 255)   # orange (BGR)
ACTUAL_COLOR = (255, 200, 80)   # light blue (BGR)


def label(img, text, org=(10, 24), color=FG, scale=0.55):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def pane(img, size, title, interp=cv2.INTER_AREA):
    out = np.full((size[1], size[0], 3), BG, np.uint8)
    if img is not None:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = cv2.resize(img, size, interpolation=interp)
    label(out, title)
    return out


def placeholder(size, title, note):
    out = pane(None, size, title)
    label(out, note, (10, size[1] // 2), DIM, 0.5)
    return out


def draw_full(img, header):
    out = pane(img, FULL, "camera (full frame)")
    fw, fh = header["full_size"]
    x, y, w, h = header["crop_rect"]
    sx, sy = FULL[0] / fw, FULL[1] / fh
    cv2.rectangle(out, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), (0, 255, 0), 2)
    return out


def draw_info(header, trace, recv_fps):
    out = np.full((INFO[1], INFO[0], 3), BG, np.uint8)
    s = header["state"]
    lines = [
        f"actuator   {header['actuator']}",
        f"camera     {header['fps']:.1f} fps   viewer {recv_fps:.1f} fps",
        f"proc       {header['proc_ms']:.1f} ms   dropped {header['dropped']}",
        f"capture->proc {header['capture_to_proc_ms']:.1f} ms",
        f"exposure   {(header['exposure'] or 0) / 10:.1f} ms  gain {header['gain']}",
        f"pan  {s['pan']:+6.2f}  (target {s['pan_target']:+6.2f})",
        f"tilt {s['tilt']:+6.2f}  (target {s['tilt_target']:+6.2f})",
        f"recording  {header.get('scene')}" if header["logging"] else "recording  no (--log)",
    ]
    for i, text in enumerate(lines):
        label(out, text, (10, 24 + i * 22), FG, 0.45)

    # Motor trace: pan target vs actual over the last few seconds.
    top, bottom, left, right = 210, 345, 10, INFO[0] - 10
    cv2.rectangle(out, (left, top), (right, bottom), (60, 60, 60), 1)
    label(out, "pan: target vs actual (4 s)", (left + 4, top + 16), DIM, 0.4)
    if len(trace) > 1:
        t_end = trace[-1][0]
        span = max(5.0, max(abs(v) for _, a, b in trace for v in (a, b)) * 1.1)

        def pt(t, v):
            x = right - (t_end - t) / 4.0 * (right - left)
            y = (top + bottom) / 2 - v / span * (bottom - top) / 2
            return int(x), int(y)

        for (t0, tgt0, act0), (t1, tgt1, act1) in zip(trace, list(trace)[1:]):
            if t_end - t0 > 4.0:
                continue
            cv2.line(out, pt(t0, tgt0), pt(t1, tgt1), TARGET_COLOR, 1, cv2.LINE_AA)
            cv2.line(out, pt(t0, act0), pt(t1, act1), ACTUAL_COLOR, 2, cv2.LINE_AA)
    return out


def compose(header, images, trace, recv_fps):
    top = np.hstack([draw_full(images.get("full"), header),
                     pane(images.get("eye"), PANE, "eye view"),
                     draw_info(header, trace, recv_fps)])
    model = images.get("model")
    bottom = np.hstack([
        pane(model, PANE, f"model input {model.shape[1]}x{model.shape[0]}" if model is not None
             else "model input", interp=cv2.INTER_NEAREST),
        pane(images.get("prediction"), PANE, "prediction", cv2.INTER_NEAREST) if "prediction" in images
        else placeholder(PANE, "prediction", "phase 1b: forward model"),
        pane(images.get("surprise"), PANE, "surprise", cv2.INTER_NEAREST) if "surprise" in images
        else placeholder(PANE, "surprise", "phase 1b: precision-weighted error"),
    ])
    return np.vstack([top, bottom])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--count", type=int, default=0, help="exit after this many messages")
    args = ap.parse_args()

    sub = DebugSubscriber(args.host, args.port)
    trace = collections.deque(maxlen=200)
    arrivals = collections.deque(maxlen=30)
    received, last_seq, skipped = 0, None, 0
    print(f"connecting to tcp://{args.host}:{args.port} ...")
    if not args.headless:
        cv2.namedWindow("efference eye", cv2.WINDOW_NORMAL)

    while not args.count or received < args.count:
        msg = sub.recv_latest(timeout_ms=1000)
        if msg is None:
            print("waiting for the Pi (is `python -m eye.run` running?)")
            continue
        header, images = msg
        received += 1
        if last_seq is not None and header["seq"] > last_seq + 1:
            skipped += header["seq"] - last_seq - 1
        last_seq = header["seq"]
        arrivals.append(time.monotonic())
        recv_fps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0]) if len(arrivals) > 1 else 0.0
        s = header["state"]
        trace.append((header["t_sensor"], s["pan_target"], s["pan"]))

        if args.headless:
            if received % 30 == 0 or received == 1:
                shapes = {k: v.shape for k, v in images.items()}
                print(f"msg {received}: seq {header['seq']} camera {header['fps']} fps, "
                      f"viewer {recv_fps:.1f} fps, proc {header['proc_ms']} ms, "
                      f"pan {s['pan']:+.2f}/{s['pan_target']:+.2f}, images {shapes}")
            continue

        canvas = compose(header, images, trace, recv_fps)
        cv2.imshow("efference eye", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            path = time.strftime("snapshot_%H%M%S.png")
            cv2.imwrite(path, canvas)
            print(f"saved {path}")

    print(f"received {received} messages; {skipped} frames skipped by the viewer (network/display)")


if __name__ == "__main__":
    main()
