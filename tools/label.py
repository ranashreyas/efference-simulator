"""Label anomalies in a recorded episode, after the fact. Run on the Mac.

    .venv/bin/python tools/label.py recordings/kitchen_test_20261018_101500
    .venv/bin/python tools/label.py                 # opens the newest episode

Controls
    SPACE        play / pause
    , .          one frame back / forward          (also left / right arrow)
    j l          1 s back / forward                (also down / up arrow = 10 s)
    J L          10 s back / forward
    - =          slower / faster playback (0.25x ... 8x)
    [            mark anomaly START at the current frame
    ]            mark anomaly END here -> adds the interval
    x            delete the interval under the playhead
    n p          jump to the next / previous interval
    click, drag  on the timeline to seek
    s            save          q / Esc   save and quit

Saves labels.json + labels.npy into the episode folder (format in eye/episode.py).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from eye.episode import chunk_paths, frame_index, load_labels, load_meta, save_labels  # noqa: E402

VIEW = (960, 540)
TIMELINE_H = 64
HELP_H = 26
WIN = "label anomalies"
RED, ORANGE, WHITE, DIM, BG = (40, 40, 230), (40, 160, 255), (235, 235, 235), (140, 140, 140), (28, 28, 28)
SPEEDS = [0.25, 0.5, 1, 2, 4, 8]

# cv2.waitKeyEx codes for arrow keys (macOS, then Linux/X11)
LEFT, RIGHT, UP, DOWN = {63234, 65361}, {63235, 65363}, {63232, 65362}, {63233, 65364}


class PreviewVideo:
    """Frames from preview.mp4. Sequential reads are fast; jumps seek."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self._next = 0

    def get(self, i):
        if i != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = self.cap.read()
        self._next = i + 1 if ok else -1
        return img if ok else None


class ChunkFrames:
    """Fallback for episodes without preview.mp4: grayscale frames from the npz chunks."""

    def __init__(self, ep):
        self.paths = chunk_paths(ep)
        lengths = []
        for p in self.paths:
            with np.load(p) as d:
                lengths.append(len(d["seq"]))
        self.starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        self._k, self._frames = None, None

    def get(self, i):
        k = int(np.searchsorted(self.starts, i, side="right")) - 1
        if k != self._k:
            with np.load(self.paths[k]) as d:
                self._frames = d["frames"]
            self._k = k
        return cv2.cvtColor(self._frames[i - self.starts[k]], cv2.COLOR_GRAY2BGR)


def fmt(t):
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def text(img, s, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


class Labeler:
    def __init__(self, ep):
        self.ep = ep
        self.meta = load_meta(ep)
        self.seqs, t = frame_index(ep)
        self.n = len(self.seqs)
        if self.n == 0:
            sys.exit(f"{ep} has no frames")
        self.t = t - t[0]
        self.fps = self.meta.get("fps", 30)
        preview = ep / "preview.mp4"
        self.frames = PreviewVideo(preview) if preview.exists() else ChunkFrames(ep)
        self.intervals = load_labels(ep, self.seqs)
        self.idx, self.playing, self.speed_i = 0, False, SPEEDS.index(1)
        self.pending = None            # frame where "[" was pressed
        self.dirty = False
        self._img_idx, self._img = None, None

    # ---- editing ------------------------------------------------------------------------
    def seek(self, i):
        self.idx = int(np.clip(i, 0, self.n - 1))

    def interval_at(self, i):
        for k, (a, b) in enumerate(self.intervals):
            if a <= i <= b:
                return k
        return None

    def mark_start(self):
        self.pending = self.idx

    def mark_end(self):
        if self.pending is None:
            self.pending = self.idx       # "]" first: treat as start, wait for the other end
            return
        self.intervals.append((min(self.pending, self.idx), max(self.pending, self.idx)))
        self.pending = None
        self.dirty = True

    def delete_here(self):
        k = self.interval_at(self.idx)
        if k is not None:
            self.intervals.pop(k)
            self.dirty = True

    def jump_interval(self, direction):
        starts = sorted(a for a, _ in self.intervals)
        if direction > 0:
            nxt = [a for a in starts if a > self.idx]
            if nxt:
                self.seek(nxt[0])
        else:
            prv = [a for a in starts if a < self.idx]
            if prv:
                self.seek(prv[-1])

    def save(self):
        self.intervals = save_labels(self.ep, self.intervals, self.seqs, source="offline")
        self.dirty = False
        n_anom = sum(b - a + 1 for a, b in self.intervals)
        print(f"saved {len(self.intervals)} intervals, {n_anom} anomalous frames "
              f"({n_anom / self.fps:.1f} s) -> {self.ep / 'labels.json'}")

    # ---- drawing ------------------------------------------------------------------------
    def frame_img(self):
        if self._img_idx != self.idx:
            img = self.frames.get(self.idx)
            self._img = cv2.resize(img, VIEW) if img is not None else np.zeros((VIEW[1], VIEW[0], 3), np.uint8)
            self._img_idx = self.idx
        return self._img.copy()

    def x_of(self, i):
        return int(round(i / max(1, self.n - 1) * (VIEW[0] - 1)))

    def render(self):
        view = self.frame_img()
        if self.interval_at(self.idx) is not None:
            cv2.rectangle(view, (0, 0), (VIEW[0] - 1, VIEW[1] - 1), RED, 10)
            text(view, "ANOMALY", (VIEW[0] - 150, 34), RED, 0.8, 2)
        if self.pending is not None:
            text(view, f"start marked at {fmt(self.t[self.pending])} - press ] to end, [ to move it",
                 (12, VIEW[1] - 16), ORANGE, 0.55)
        state = "PLAY" if self.playing else "PAUSE"
        text(view, f"{self.ep.name}   {fmt(self.t[self.idx])} / {fmt(self.t[-1])}   "
                   f"frame {self.idx}/{self.n - 1}   {SPEEDS[self.speed_i]}x {state}   "
                   f"{len(self.intervals)} intervals{'  *unsaved' if self.dirty else ''}",
             (12, 26), WHITE, 0.55)

        tl = np.full((TIMELINE_H, VIEW[0], 3), BG, np.uint8)
        for m in range(0, int(self.t[-1]) + 1, 60):                      # minute ticks
            x = self.x_of(int(np.searchsorted(self.t, m)))
            cv2.line(tl, (x, TIMELINE_H - 10), (x, TIMELINE_H - 1), DIM, 1)
        for a, b in self.intervals:
            cv2.rectangle(tl, (self.x_of(a), 10), (max(self.x_of(b), self.x_of(a) + 2), TIMELINE_H - 14), RED, -1)
        if self.pending is not None:
            a, b = sorted((self.pending, self.idx))
            cv2.rectangle(tl, (self.x_of(a), 10), (max(self.x_of(b), self.x_of(a) + 2), TIMELINE_H - 14), ORANGE, -1)
        x = self.x_of(self.idx)
        cv2.line(tl, (x, 0), (x, TIMELINE_H - 1), WHITE, 2)

        help_bar = np.full((HELP_H, VIEW[0], 3), (18, 18, 18), np.uint8)
        text(help_bar, "SPACE play  ,/. frame  j/l 1s  J/L 10s  -/= speed  [ start  ] end  "
                       "x delete  n/p next/prev  s save  q quit", (10, 18), DIM, 0.45)
        return np.vstack([view, tl, help_bar])

    # ---- input --------------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _):
        on_timeline = VIEW[1] <= y < VIEW[1] + TIMELINE_H
        dragging = event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        if on_timeline and (event == cv2.EVENT_LBUTTONDOWN or dragging):
            self.seek(round(x / (VIEW[0] - 1) * (self.n - 1)))

    def handle_key(self, key):
        """Returns False to quit."""
        one_s = int(round(self.fps))
        ch = chr(key) if 0 <= key < 256 else None
        if key in (27,) or ch == "q":
            return False
        if ch == " ":
            self.playing = not self.playing
            if self.playing and self.idx >= self.n - 1:
                self.seek(0)
        elif ch == "," or key in LEFT:
            self.playing = False
            self.seek(self.idx - 1)
        elif ch == "." or key in RIGHT:
            self.playing = False
            self.seek(self.idx + 1)
        elif ch == "j":
            self.seek(self.idx - one_s)
        elif ch == "l":
            self.seek(self.idx + one_s)
        elif ch == "J" or key in DOWN:
            self.seek(self.idx - 10 * one_s)
        elif ch == "L" or key in UP:
            self.seek(self.idx + 10 * one_s)
        elif ch == "-":
            self.speed_i = max(0, self.speed_i - 1)
        elif ch in ("=", "+"):
            self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        elif ch == "[":
            self.mark_start()
        elif ch == "]":
            self.mark_end()
        elif ch == "x":
            self.delete_here()
        elif ch == "n":
            self.jump_interval(+1)
        elif ch == "p":
            self.jump_interval(-1)
        elif ch == "s":
            self.save()
        return True

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        print(f"{self.ep.name}: {self.n} frames ({self.t[-1]:.0f} s), "
              f"{self.meta.get('scene', '?')}/{self.meta.get('split', '?')}, "
              f"{len(self.intervals)} existing intervals")
        while True:
            cv2.imshow(WIN, self.render())
            speed = SPEEDS[self.speed_i]
            step = max(1, int(speed))
            delay = max(1, int(1000 * step / (self.fps * speed))) if self.playing else 30
            key = cv2.waitKeyEx(delay)
            if key == -1:
                if self.playing:
                    self.seek(self.idx + step)
                    if self.idx >= self.n - 1:
                        self.playing = False
            elif not self.handle_key(key):
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:     # window closed
                break
        if self.pending is not None:
            print("note: an unfinished START mark was discarded")
        self.save()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", nargs="?", help="episode folder (default: newest in recordings/)")
    args = ap.parse_args()
    if args.episode:
        ep = Path(args.episode)
    else:
        eps = sorted((p for p in (PROJECT / "recordings").glob("*") if p.is_dir()),
                     key=lambda p: p.stat().st_mtime)
        if not eps:
            sys.exit("No episodes in recordings/. Record one with tools/record.py first.")
        ep = eps[-1]
    Labeler(ep.resolve()).run()


if __name__ == "__main__":
    main()
