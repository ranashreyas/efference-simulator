"""Live viewer (runs on the Mac). Subscribes to one of the Pi's eye pipelines.

    .venv/bin/python -m viewer.view                 # first pipeline (port 5555)
    .venv/bin/python -m viewer.view --port 5556     # second pipeline (e.g. the panning eye)
    .venv/bin/python -m viewer.view --headless --count 90   # no window; print stats and exit

Layout:
    [ actual: what the model sees ] [ model surprise ] [ frame difference ]
    [ stats | score traces over the last 10 s (and pan angle for the panning eye)   ]
    [ "show top X% of pixels" slider                                               ]
    [ [x] remove specks of <= N pixels   N slider (1-10)                            ]
    [ [x] remote control (panning eye): arrow keys move the eye, overriding --motion ]
The slider filters what both heatmaps show: each pane keeps only its own strongest X% of pixels,
so surprise and frame difference get the same pixel budget whatever their units. Display only:
the model and the scores are unchanged. Drag it, or use - and = to step it.
With the checkbox on, any connected cluster of N or fewer pixels that survives the slider is
removed from both heatmaps (isolated specks of noise); with it off, the check doesn't run.
Remote control (panning window only): when checked, the Pi stops following its --motion pattern
and the arrow keys drive the virtual eye instead (via the Pi's control port, 5560). The checkbox
shows the Pi's actual state, so it ticks once the Pi has switched over.
Keys: - = slider, c speck removal on/off, , . change N, r remote control on/off,
      arrow keys move the eye (remote control on), s save a snapshot, e event window on/off, q quit

Event browser: the panning viewer also opens a second window listing the frames the Pi saved at
surprise events in three folders (unlabelled / safe / motion), refreshed every 5 s, where you can
open any picture and label it (viewer/events.py). It reaches the Pi by itself: an ssh tunnel to
the Pi's localhost-only API, and the API token read from the Pi over ssh. --no-events turns it
off, --events opens it from any viewer, --events-url skips the tunnel and uses that address.
"""
import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eye.stream import ControlSender, DebugSubscriber  # noqa: E402

PANE = (480, 360)
BAR_H = 130
SLIDER_H = 44
SURPRISE_VMAX = 50.0              # fallbacks; the Pi sends the scales it used in the header
DIFF_VMAX = 0.25
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
    """Display-only filter for both heatmaps: each keeps only its own strongest X% of pixels.
    Position 100 = show all; lower positions show fewer, on a log scale down to 0 = top 0.1%.
    Note a top-X% filter always shows *something*: on a quiet frame the strongest 1% is just the
    strongest noise."""

    def __init__(self, value=100):
        self.value = value
        self.x0, self.x1 = 250, PANE[0] * 3 - 400         # slider track, in canvas pixels
        self.top = PANE[1] + BAR_H                          # where the slider row starts

    def fraction(self):
        return 10 ** (-3 * (1 - self.value / 100))          # 100 -> 100%, 67 -> 10%, 33 -> 1%, 0 -> 0.1%

    def apply(self, img, vmax):
        """img: a map sent as 255 * sqrt(value / vmax) (monotonic, so ranking pixels by it is
        ranking them by value). Keeps the strongest fraction(); returns the filtered image and
        the cutoff in the map's own units."""
        f = self.fraction()
        if f >= 1:
            return img, 0.0
        flat = img.ravel()
        k = max(1, int(round(len(flat) * f)))
        cut = int(np.partition(flat, -k)[-k])
        out = np.where(img >= max(cut, 1), img, 0).astype(np.uint8)    # never "keep" pure zeros
        return out, (cut / 255) ** 2 * vmax

    def nudge(self, step):
        self.value = int(np.clip(self.value + step, 0, 100))

    def on_mouse(self, event, x, y, flags, _):
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if self.top <= y < self.top + SLIDER_H and (event == cv2.EVENT_LBUTTONDOWN or dragging):
            self.value = int(round(np.clip((x - self.x0) / (self.x1 - self.x0), 0, 1) * 100))

    def draw(self, cut_surprise, cut_diff):
        out = np.full((SLIDER_H, PANE[0] * 3, 3), (30, 30, 30), np.uint8)
        mid = SLIDER_H // 2
        label(out, "show top % of pixels", (10, mid + 5), FG, 0.5)
        cv2.line(out, (self.x0, mid), (self.x1, mid), (90, 90, 90), 4, cv2.LINE_AA)
        x = int(self.x0 + self.value / 100 * (self.x1 - self.x0))
        cv2.line(out, (self.x0, mid), (x, mid), SURPRISE_COLOR, 4, cv2.LINE_AA)
        cv2.circle(out, (x, mid), 9, SURPRISE_COLOR, -1, cv2.LINE_AA)
        f = self.fraction()
        text = ("showing all pixels" if f >= 1 else
                f"top {100 * f:.2g}% per pane (surprise >= {cut_surprise:.3g}, diff >= {cut_diff:.2g})")
        label(out, text, (self.x1 + 20, mid + 5), FG, 0.45)
        return out


class SpeckFilter:
    """Display-only: when enabled, removes connected clusters (8-connected) of at most `size`
    pixels from a filtered heatmap. Runs after the top-X% slider, on both heatmaps."""

    def __init__(self, enabled=False, size=3):
        self.enabled, self.size = enabled, size
        self.top = PANE[1] + BAR_H + SLIDER_H                 # where this control row starts
        self.box = (14, 12, 34, 32)                            # checkbox, canvas pixels
        self.x0, self.x1 = 330, 560                            # N slider track

    def apply(self, img):
        """Returns the image with small clusters removed, and (removed, total) cluster counts."""
        if not self.enabled:
            return img, None
        n, labels, stats, _ = cv2.connectedComponentsWithStats((img > 0).astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]                    # label 0 is the background
        small = np.flatnonzero(areas <= self.size) + 1
        out = img.copy()
        out[np.isin(labels, small)] = 0
        return out, (len(small), n - 1)

    def nudge(self, step):
        self.size = int(np.clip(self.size + step, 1, 10))

    def on_mouse(self, event, x, y, flags, _):
        if not self.top <= y < self.top + SLIDER_H:
            return
        bx0, by0, bx1, by1 = self.box
        if event == cv2.EVENT_LBUTTONDOWN and bx0 - 4 <= x <= 300 and by0 - 4 <= y - self.top <= by1 + 4:
            self.enabled = not self.enabled                    # click the box or its label
            return
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if (event == cv2.EVENT_LBUTTONDOWN or dragging) and self.x0 - 10 <= x <= self.x1 + 10:
            self.size = int(round(1 + np.clip((x - self.x0) / (self.x1 - self.x0), 0, 1) * 9))

    def draw(self, counts_s, counts_d):
        out = np.full((SLIDER_H, PANE[0] * 3, 3), (30, 30, 30), np.uint8)
        mid = SLIDER_H // 2
        bx0, by0, bx1, by1 = self.box
        cv2.rectangle(out, (bx0, by0), (bx1, by1), FG, 1)
        if self.enabled:
            cv2.line(out, (bx0 + 4, mid), (bx0 + 9, by1 - 5), SURPRISE_COLOR, 2, cv2.LINE_AA)
            cv2.line(out, (bx0 + 9, by1 - 5), (bx1 - 3, by0 + 4), SURPRISE_COLOR, 2, cv2.LINE_AA)
        color = FG if self.enabled else DIM
        label(out, f"remove specks of <= {self.size} px", (bx1 + 12, mid + 5), color, 0.5)
        cv2.line(out, (self.x0, mid), (self.x1, mid), (90, 90, 90), 4, cv2.LINE_AA)
        x = int(self.x0 + (self.size - 1) / 9 * (self.x1 - self.x0))
        cv2.line(out, (self.x0, mid), (x, mid), SURPRISE_COLOR if self.enabled else DIM, 4, cv2.LINE_AA)
        cv2.circle(out, (x, mid), 8, SURPRISE_COLOR if self.enabled else DIM, -1, cv2.LINE_AA)
        if self.enabled and counts_s and counts_d:
            text = (f"removed {counts_s[0]} of {counts_s[1]} clusters (surprise), "
                    f"{counts_d[0]} of {counts_d[1]} (diff)")
        else:
            text = "off: no cluster check" if not self.enabled else ""
        label(out, text, (self.x1 + 30, mid + 5), color, 0.45)
        return out


# cv2.waitKeyEx codes for arrow keys (macOS, then Linux/X11) -> (dx, dy)
ARROWS = {63234: (-1, 0), 65361: (-1, 0), 63235: (1, 0), 65363: (1, 0),
          63232: (0, 1), 65362: (0, 1), 63233: (0, -1), 65364: (0, -1)}


class RemoteToggle:
    """Checkbox row for driving the panning eye with the arrow keys. Clicking asks the Pi to
    switch; the tick follows the Pi's reported state (header["remote"])."""

    def __init__(self, sender):
        self.sender = sender
        self.top = PANE[1] + BAR_H + 2 * SLIDER_H
        self.box = (14, 12, 34, 32)
        self.header = {}

    @property
    def available(self):
        return bool(self.header.get("remote_available"))

    @property
    def on(self):
        return bool(self.header.get("remote"))

    def toggle(self):
        if self.available:
            self.sender.send({"type": "remote", "enabled": not self.on})

    def arrow(self, dx, dy):
        if self.on:
            self.sender.send({"type": "key", "dx": dx, "dy": dy})

    def on_mouse(self, event, x, y, flags, _):
        bx0, by0, bx1, by1 = self.box
        if (event == cv2.EVENT_LBUTTONDOWN and self.top <= y < self.top + SLIDER_H
                and bx0 - 4 <= x <= 600 and by0 - 4 <= y - self.top <= by1 + 4):
            self.toggle()

    def draw(self, header):
        self.header = header
        out = np.full((SLIDER_H, PANE[0] * 3, 3), (30, 30, 30), np.uint8)
        mid = SLIDER_H // 2
        bx0, by0, bx1, by1 = self.box
        color = FG if self.available else DIM
        cv2.rectangle(out, (bx0, by0), (bx1, by1), color, 1)
        if self.on:
            cv2.line(out, (bx0 + 4, mid), (bx0 + 9, by1 - 5), PAN_COLOR, 2, cv2.LINE_AA)
            cv2.line(out, (bx0 + 9, by1 - 5), (bx1 - 3, by0 + 4), PAN_COLOR, 2, cv2.LINE_AA)
        label(out, "remote control: arrow keys move the eye (overrides --motion)", (bx1 + 12, mid + 5), color, 0.5)
        if not self.available:
            text = "static eye: nothing to drive"
        elif self.on:
            st = header["state"]
            text = f"ON  pan {st['pan_target']:+5.1f}  tilt {st['tilt_target']:+5.1f}  (use the arrow keys)"
        else:
            text = "off: following --motion"
        label(out, text, (640, mid + 5), PAN_COLOR if self.on else color, 0.5)
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


def baseline_name(header, long=False):
    """The panning eye's baseline has its own motion removed (shift from the pan/tilt history)."""
    if header.get("mode") == "panning":
        return "shift-compensated diff" if long else "shift diff"
    return "frame difference" if long else "frame diff"


def bar(header, trace, recv_fps):
    out = np.full((BAR_H, PANE[0] * 3, 3), BG, np.uint8)
    s = header["state"]
    lines = [f"{header['mode']} eye   camera {header['fps']:.1f} fps   viewer {recv_fps:.1f} fps",
             f"model {header.get('infer_ms', '-')} ms   loop {header['proc_ms']} ms   "
             f"dropped {header['dropped']}   skipped {header.get('skipped', 0)}",
             f"surprise {header.get('surprise_score', 0):.3g}   {baseline_name(header)} {header.get('diff_score', 0):.3g}"]
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
    names = [("surprise", SURPRISE_COLOR), (baseline_name(header), DIFF_COLOR)]
    if header["mode"] == "panning":
        lim = header["pan_limit"] or 1
        draw([p[3] for p in pts], PAN_COLOR, -lim, lim)
        names.append(("pan angle", PAN_COLOR))
    for i, (n, c) in enumerate(names):
        label(out, n, (left + 8 + i * 110, top + 16), c, 0.42)
    label(out, f"last {TRACE_S:.0f} s (each trace auto-scaled)", (right - 230, top + 16), DIM, 0.4)
    return out


def compose(header, images, trace, recv_fps, slider=None, specks=None, remote=None):
    slider = slider or SensitivitySlider()
    specks = specks or SpeckFilter()
    remote = remote or RemoteToggle(None)
    surprise, diff, cut_s, cut_d = images.get("surprise"), images.get("diff"), 0.0, 0.0
    counts_s = counts_d = None
    if surprise is not None:
        surprise, cut_s = slider.apply(surprise, header.get("surprise_vmax", SURPRISE_VMAX))
        surprise, counts_s = specks.apply(surprise)
    if diff is not None:
        diff, cut_d = slider.apply(diff, header.get("diff_vmax", DIFF_VMAX))
        diff, counts_d = specks.apply(diff)
    row = np.hstack([actual_pane(images, header),
                     heat_pane(surprise, "model surprise"),
                     heat_pane(diff, baseline_name(header, long=True) + " (1/15 s)")])
    return np.vstack([row, bar(header, trace, recv_fps), slider.draw(cut_s, cut_d),
                      specks.draw(counts_s, counts_d), remote.draw(header)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="raspberrypi.local")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--control-port", type=int, default=5560, help="the Pi's remote-control port")
    ap.add_argument("--events", action="store_true", default=None,
                    help="open the event browser (default: only in the panning viewer)")
    ap.add_argument("--no-events", dest="events", action="store_false", help="never open the event browser")
    ap.add_argument("--pi", default=os.environ.get("PI", "ranashreyas@raspberrypi.local"),
                    help="ssh login of the Pi, for the event browser's tunnel and token (default: $PI)")
    ap.add_argument("--api-port", type=int, default=8787, help="the eye loop's --api-port on the Pi")
    ap.add_argument("--events-url", help="the API's address, if you tunnel yourself (needs a token)")
    ap.add_argument("--events-token-file", help="file holding the API token (default: $EVENTS_TOKEN, "
                                                "else read from the Pi over ssh)")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--count", type=int, default=0, help="exit after this many messages")
    args = ap.parse_args()

    events = None

    def make_events():
        from pathlib import Path
        from viewer.events import EventBrowser, Link
        token = (Path(args.events_token_file).expanduser().read_text() if args.events_token_file
                 else os.environ.get("EVENTS_TOKEN", "")).strip() or None
        if args.events_url:
            if not token:
                sys.exit("--events-url needs the API token: set $EVENTS_TOKEN or pass --events-token-file")
            return EventBrowser(lambda: (args.events_url, token))
        return EventBrowser(Link(args.pi, token, args.api_port).ensure)

    if args.events and not args.headless:
        events = make_events()

    sub = DebugSubscriber(args.host, args.port)
    trace = collections.deque(maxlen=600)
    arrivals = collections.deque(maxlen=30)
    received, last_seq, skipped = 0, None, 0
    window = None
    slider = SensitivitySlider()
    specks = SpeckFilter()
    remote = RemoteToggle(ControlSender(args.host, args.control_port))
    print(f"connecting to tcp://{args.host}:{args.port} ...")

    last_wait_msg = 0.0
    while not args.count or received < args.count:
        msg = sub.recv_latest(timeout_ms=30 if events else 1000)
        if msg is None:
            if time.monotonic() - last_wait_msg > 2:
                print("waiting for the Pi (is `python -m eye.run` running?)")
                last_wait_msg = time.monotonic()
            if events:                                  # the event browser works without the live stream
                events.show()
                raw = cv2.waitKeyEx(1)
                key = raw & 0xFF if raw != -1 else -1
                if key == ord("q"):
                    break
                if key == ord("e"):
                    events.toggle()
                elif key != -1:
                    events.key(key)
            continue
        header, images = msg
        received += 1
        if events is None and args.events is None and header["mode"] == "panning" and not args.headless:
            events = make_events()                      # the panning viewer hosts the event browser
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
            cv2.setMouseCallback(window, lambda *a: (slider.on_mouse(*a), specks.on_mouse(*a),
                                                     remote.on_mouse(*a)))
        canvas = compose(header, images, trace, recv_fps, slider, specks, remote)
        cv2.imshow(window, canvas)
        if events:
            events.show()
        raw = cv2.waitKeyEx(1)
        if raw in ARROWS:
            remote.arrow(*ARROWS[raw])
            continue
        key = raw & 0xFF if raw != -1 else -1
        if events and key == ord("e"):
            events.toggle()
            continue
        if events and key != -1 and events.key(key):
            continue                                    # the key was meant for the event browser
        if key == ord("r"):
            remote.toggle()
        if key == ord("q"):
            break
        if key == ord("-"):
            slider.nudge(-5)
        if key in (ord("="), ord("+")):
            slider.nudge(5)
        if key == ord("c"):
            specks.enabled = not specks.enabled
        if key == ord(","):
            specks.nudge(-1)
        if key == ord("."):
            specks.nudge(1)
        if key == ord("s"):
            path = time.strftime("snapshot_%H%M%S.png")
            cv2.imwrite(path, canvas)
            print(f"saved {path}")

    print(f"received {received} messages; {skipped} frames skipped by the viewer (network/display)")


if __name__ == "__main__":
    main()
