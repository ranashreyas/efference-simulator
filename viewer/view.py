"""Live viewer (runs on the Mac). Subscribes to one of the Pi's eye pipelines.

    .venv/bin/python -m viewer.view                 # first pipeline (port 5555)
    .venv/bin/python -m viewer.view --port 5556     # second pipeline (e.g. the panning eye)
    .venv/bin/python -m viewer.view --headless --count 90   # no window; print stats and exit

Layout:
    [ actual: what the model sees ] [ model surprise ] [ frame difference ]
    [ stats | score traces over the last 10 s (and pan angle for the panning eye)   ]
    [ surprise sensitivity slider                                                   ]
The slider only filters what the surprise pane shows (surprise below its threshold is hidden);
the model and the scores are unchanged. Drag it, or use - and = to step it.
Keys: - = sensitivity, s save a snapshot, q quit
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
BAR_H = 130
SLIDER_H = 44
SURPRISE_VMAX = 50.0              # fallback; the Pi sends the scale it used in the header
BG = (24, 24, 24)
FG = (230, 230, 230)
DIM = (130, 130, 130)
SURPRISE_COLOR = (80, 200, 255)   # orange (BGR)
DIFF_COLOR = (170, 170, 170)
PAN_COLOR = (255, 200, 80)        # light blue
TRACE_S = 10.0


def label(img, text, org=(10, 24), color=FG, scale=0.55):
    """Text on a dark box, readable over any image."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (16, 16, 16), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def actual_pane(images, header):
    out = cv2.resize(images["eye"], PANE, interpolation=cv2.INTER_AREA)
    if "full" in images:                                # panning: where in the room the eye points
        tw, th = 176, 99
        thumb = cv2.resize(images["full"], (tw, th), interpolation=cv2.INTER_AREA)
        fw, fh = header["full_size"]
        x, y, w, h = header["crop_rect"]
        cv2.rectangle(thumb, (int(x * tw / fw), int(y * th / fh)),
                      (int((x + w) * tw / fw), int((y + h) * th / fh)), (0, 255, 0), 1)
        out[PANE[1] - th - 8:PANE[1] - 8, PANE[0] - tw - 8:PANE[0] - 8] = thumb
    label(out, "actual (what the model sees)")
    return out


class SensitivitySlider:
    """Display-only filter for the surprise pane. 100 = show all surprise; lower values hide
    weaker surprise, on a log scale down to 0 = only the strongest (at the color scale's top)."""

    def __init__(self, value=100):
        self.value = value
        self.x0, self.x1 = 250, PANE[0] * 3 - 330         # slider track, in canvas pixels
        self.top = PANE[1] + BAR_H                          # where the slider row starts

    def threshold(self, vmax):
        return 0.0 if self.value >= 100 else vmax * 10 ** (-4 * self.value / 100)

    def apply(self, img, vmax):
        """img: surprise sent as 255 * sqrt(value / vmax). Hide pixels below the threshold."""
        cut = 255 * np.sqrt(self.threshold(vmax) / vmax)
        keep = img >= cut
        return np.where(keep, img, 0).astype(np.uint8), float(keep.mean())

    def nudge(self, step):
        self.value = int(np.clip(self.value + step, 0, 100))

    def on_mouse(self, event, x, y, flags, _):
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if self.top <= y < self.top + SLIDER_H and (event == cv2.EVENT_LBUTTONDOWN or dragging):
            self.value = int(round(np.clip((x - self.x0) / (self.x1 - self.x0), 0, 1) * 100))

    def draw(self, vmax, shown):
        out = np.full((SLIDER_H, PANE[0] * 3, 3), (30, 30, 30), np.uint8)
        mid = SLIDER_H // 2
        label(out, "surprise sensitivity", (10, mid + 5), FG, 0.5)
        cv2.line(out, (self.x0, mid), (self.x1, mid), (90, 90, 90), 4, cv2.LINE_AA)
        x = int(self.x0 + self.value / 100 * (self.x1 - self.x0))
        cv2.line(out, (self.x0, mid), (x, mid), SURPRISE_COLOR, 4, cv2.LINE_AA)
        cv2.circle(out, (x, mid), 9, SURPRISE_COLOR, -1, cv2.LINE_AA)
        thr = self.threshold(vmax)
        text = (f"{self.value:3d}  showing all" if thr == 0 else
                f"{self.value:3d}  showing surprise > {thr:.3g}  ({shown:.1%} of pixels)")
        label(out, text, (self.x1 + 20, mid + 5), FG, 0.45)
        return out


def heat_pane(img, title):
    if img is None:
        out = np.full((PANE[1], PANE[0], 3), BG, np.uint8)
        label(out, "warming up (needs 7 frames)" if title else "", (10, PANE[1] // 2), DIM, 0.5)
    else:
        out = cv2.applyColorMap(cv2.resize(img, PANE, interpolation=cv2.INTER_NEAREST),
                                cv2.COLORMAP_INFERNO)
    label(out, title)
    return out


def bar(header, trace, recv_fps):
    out = np.full((BAR_H, PANE[0] * 3, 3), BG, np.uint8)
    s = header["state"]
    lines = [f"{header['mode']} eye   camera {header['fps']:.1f} fps   viewer {recv_fps:.1f} fps",
             f"model {header.get('infer_ms', '-')} ms   loop {header['proc_ms']} ms   "
             f"dropped {header['dropped']}   skipped {header.get('skipped', 0)}",
             f"surprise {header.get('surprise_score', 0):.3g}   frame diff {header.get('diff_score', 0):.3g}"]
    if header["mode"] == "panning":
        lines.append(f"pan {s['pan']:+5.1f} (cmd {s['pan_target']:+5.1f})  "
                     f"tilt {s['tilt']:+5.1f} (cmd {s['tilt_target']:+5.1f})")
    for i, text in enumerate(lines):
        label(out, text, (10, 24 + i * 24), FG, 0.45)

    left, right, top, bottom = 480, PANE[0] * 3 - 10, 10, BAR_H - 10
    cv2.rectangle(out, (left, top), (right, bottom), (60, 60, 60), 1)
    if len(trace) < 2:
        return out
    t_end = trace[-1][0]
    pts = [p for p in trace if t_end - p[0] <= TRACE_S]
    x = np.array([right - (t_end - p[0]) / TRACE_S * (right - left) for p in pts], np.int32)

    def draw(values, color, lo, hi, thick=1):
        v = np.clip((np.asarray(values, float) - lo) / max(hi - lo, 1e-12), 0, 1)
        y = (bottom - 4 - v * (bottom - top - 8)).astype(np.int32)
        cv2.polylines(out, [np.stack([x, y], 1)], False, color, thick, cv2.LINE_AA)

    sur = np.sqrt([p[1] for p in pts])                  # sqrt: spikes visible without flattening the rest
    dif = np.sqrt([p[2] for p in pts])
    draw(dif, DIFF_COLOR, 0, max(dif.max(), 1e-6))
    draw(sur, SURPRISE_COLOR, 0, max(sur.max(), 1e-6), 2)
    names = [("surprise", SURPRISE_COLOR), ("frame diff", DIFF_COLOR)]
    if header["mode"] == "panning":
        lim = header["pan_limit"] or 1
        draw([p[3] for p in pts], PAN_COLOR, -lim, lim)
        names.append(("pan angle", PAN_COLOR))
    for i, (n, c) in enumerate(names):
        label(out, n, (left + 8 + i * 110, top + 16), c, 0.42)
    label(out, f"last {TRACE_S:.0f} s (each trace auto-scaled)", (right - 230, top + 16), DIM, 0.4)
    return out


def compose(header, images, trace, recv_fps, slider=None):
    slider = slider or SensitivitySlider()
    vmax = header.get("surprise_vmax", SURPRISE_VMAX)
    surprise, shown = images.get("surprise"), 1.0
    if surprise is not None:
        surprise, shown = slider.apply(surprise, vmax)
    row = np.hstack([actual_pane(images, header),
                     heat_pane(surprise, "model surprise"),
                     heat_pane(images.get("diff"), "frame difference (1/15 s)")])
    return np.vstack([row, bar(header, trace, recv_fps), slider.draw(vmax, shown)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--count", type=int, default=0, help="exit after this many messages")
    args = ap.parse_args()

    sub = DebugSubscriber(args.host, args.port)
    trace = collections.deque(maxlen=600)
    arrivals = collections.deque(maxlen=30)
    received, last_seq, skipped = 0, None, 0
    window = None
    slider = SensitivitySlider()
    print(f"connecting to tcp://{args.host}:{args.port} ...")

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
        trace.append((header["t_sensor"], header.get("surprise_score", 0.0),
                      header.get("diff_score", 0.0), header["state"]["pan"]))

        if args.headless:
            if received % 30 == 0 or received == 1:
                print(f"msg {received}: {header['mode']} seq {header['seq']} camera {header['fps']} fps, "
                      f"viewer {recv_fps:.1f} fps, model {header.get('infer_ms', '-')} ms, "
                      f"loop {header['proc_ms']} ms, surprise {header.get('surprise_score', 0):.3g}, "
                      f"diff {header.get('diff_score', 0):.3g}, pan {header['state']['pan']:+.1f}, "
                      f"images {sorted(images)}")
            continue

        if window is None:
            window = f"efference eye - {header['mode']} (port {args.port})"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window, slider.on_mouse)
        canvas = compose(header, images, trace, recv_fps, slider)
        cv2.imshow(window, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("-"):
            slider.nudge(-5)
        if key in (ord("="), ord("+")):
            slider.nudge(5)
        if key == ord("s"):
            path = time.strftime("snapshot_%H%M%S.png")
            cv2.imwrite(path, canvas)
            print(f"saved {path}")

    print(f"received {received} messages; {skipped} frames skipped by the viewer (network/display)")


if __name__ == "__main__":
    main()
