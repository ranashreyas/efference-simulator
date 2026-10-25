"""Event browser (runs on the Mac, inside viewer.view): the frames the Pi saved at surprise events,
sorted into three folders you can open, with a button to label each one.

    Home          three folders: Unlabelled, Safe, Motion (counts + latest thumbnails)
    Folder        a page of thumbnails, newest first; click one to open it
    Picture       the saved picture, [Safe] / [Motion] buttons, earlier-frame toggle, next/previous

Labeling a picture sends it to the Pi (eye/api.py) and moves on to the next one in the folder.
Pictures can also be deleted for good, one at a time or a whole folder at once, to free the Pi's
disk (the window shows how much the events use). Deleting asks for a second click to confirm.
The list is re-read from the Pi every 5 s, so the Pi's copy is always the truth. Network work
runs on background threads; drawing never waits for it.

Keys (when this window is focused): 1 Safe, 2 Motion, d delete (press twice), n / p next /
previous (page in a folder), b or Esc back.

The panning viewer opens this on its own. The API listens on the Pi's localhost only, so the viewer
starts an ssh tunnel to it (and reads the API token from the Pi over ssh) in the background;
see `Link` below. With --events-url it uses that address as given instead.
"""
import atexit
import itertools
import json
import os
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

W, H = 1280, 720
TOP = 56
BG = (24, 24, 24)
PANEL = (34, 34, 34)
FG = (230, 230, 230)
DIM = (130, 130, 130)
RED = (90, 90, 255)
RED_BTN = (70, 70, 230)
BUCKETS = [("pending", "Unlabelled", (190, 190, 190)),
           ("safe", "Safe", (110, 200, 110)),
           ("motion", "Motion", (60, 150, 255))]
NAME = {k: n for k, n, _ in BUCKETS}
COLOR = {k: c for k, _, c in BUCKETS}
GRID_COLS, GRID_ROWS = 5, 3
PER_PAGE = GRID_COLS * GRID_ROWS
THUMB_W, DETAIL_W = 240, 960
MISSING = object()


def put(img, text, org, color=FG, scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def fit(img, w, h):
    """img scaled to fit inside w x h, centered on a dark background."""
    out = np.full((h, w, 3), (16, 16, 16), np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                   interpolation=cv2.INTER_AREA)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class Link:
    """How the viewer reaches the Pi's API: an ssh tunnel to its localhost-only port, and the API
    token read from the Pi (~/.config/efference/events.env) over the same ssh login. Both happen on
    the client's background thread, so the live stream never waits for them, and the tunnel is
    restarted if it drops (for example when the Pi reboots)."""

    TOKEN_FILE = ".config/efference/events.env"

    def __init__(self, pi, token=None, remote_port=8787):
        self.pi, self.token, self.proc = pi, token, None
        self.REMOTE_PORT = remote_port
        self.LOCAL_PORT = 10000 + remote_port                  # 18787 for the default
        self.url = f"http://127.0.0.1:{self.LOCAL_PORT}"
        atexit.register(self.close)

    def _listening(self):
        with socket.socket() as s:
            s.settimeout(0.5)
            return s.connect_ex(("127.0.0.1", self.LOCAL_PORT)) == 0

    def ensure(self):
        """Returns (url, token); raises with a readable reason if the Pi isn't reachable yet."""
        if self.token is None:
            r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.pi,
                                f"cat ~/{self.TOKEN_FILE}"], capture_output=True, text=True, timeout=15)
            for line in r.stdout.splitlines():
                if line.startswith("EVENTS_TOKEN="):
                    self.token = line.split("=", 1)[1].strip()
            if not self.token:
                raise RuntimeError(f"no API token on {self.pi} (is the API set up? see README, Event capture)")
        if not self._listening():
            if self.proc is not None:
                self.proc.terminate()
            self.proc = subprocess.Popen(
                ["ssh", "-N", "-L", f"{self.LOCAL_PORT}:127.0.0.1:{self.REMOTE_PORT}", "-o", "BatchMode=yes",
                 "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", self.pi],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(40):
                if self._listening():
                    break
                if self.proc.poll() is not None:
                    raise RuntimeError(f"ssh tunnel to {self.pi} failed")
                time.sleep(0.25)
            else:
                raise RuntimeError(f"ssh tunnel to {self.pi} timed out")
        return self.url, self.token

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()


class EventsClient:
    """The Pi's event list, refreshed every `poll_s`, plus lazily fetched JPEGs."""

    def __init__(self, connect, poll_s=5.0):
        """connect() -> (url, token); called on the poll thread before every refresh."""
        self.connect, self.url, self.token, self.poll_s = connect, "", "", poll_s
        self.items = []                          # oldest first, as last read from the Pi
        self.updated = None
        self.size_bytes = None                   # disk used by the Pi's event images
        self.error = None                        # the list can't be refreshed (cleared by the next good refresh)
        self.notice = None                       # (text, until): a label/delete the Pi refused; stays 10 s
        self.version = 0                         # bumps whenever anything drawable changed
        self.lock = threading.Lock()
        self.images = {}                         # (id, which, width) -> array | MISSING
        self.requested = set()
        self.jobs = queue.PriorityQueue()        # full-size pictures jump ahead of thumbnails
        self.job_n = itertools.count()
        self.wake = threading.Event()
        threading.Thread(target=self._poll_loop, daemon=True).start()
        for _ in range(4):
            threading.Thread(target=self._image_loop, daemon=True).start()

    def _call(self, method, path, body=None, timeout=10):
        req = urllib.request.Request(self.url + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def _poll_loop(self):
        while True:
            try:
                self.url, self.token = self.connect()
                self.url = self.url.rstrip("/")
                items, since = [], 0
                while True:
                    page = json.loads(self._call("GET", f"/events?state=all&since_id={since}&limit=200"))
                    items += page
                    if len(page) < 200:
                        break
                    since = page[-1]["id"]
                size = json.loads(self._call("GET", "/stats")).get("bytes")
                with self.lock:
                    self.items, self.updated, self.error = items, time.time(), None
                    self.size_bytes = size
                    self.version += 1
            except urllib.error.HTTPError as e:
                self._fail("wrong or missing token" if e.code == 401 else f"API error {e.code}")
            except Exception as e:                                   # tunnel down, Pi off, ...
                self._fail(f"can't reach the Pi's events ({getattr(e, 'reason', e)})")
            self.wake.wait(self.poll_s)
            self.wake.clear()

    def _notify(self, msg):
        with self.lock:
            self.notice = (msg, time.time() + 10)
            self.version += 1

    def _explain(self, what, e):
        """Why a label/delete didn't go through, in words."""
        if isinstance(e, urllib.error.HTTPError):
            if e.code in (404, 405, 501) and what == "delete":
                return ("the Pi's eye loop doesn't support delete yet: run tools/pi_sync.sh and restart eye.run")
            if e.code == 401:
                return f"{what} refused: wrong token"
            return f"{what} failed: the Pi answered {e.code}"
        return f"{what} failed: {getattr(e, 'reason', e)}"

    def _fail(self, msg):
        with self.lock:
            self.error = msg
            self.version += 1

    def _image_loop(self):
        while True:
            key = self.jobs.get()[2]
            event_id, which, w = key
            try:
                data = self._call("GET", f"/events/{event_id}/image?which={which}&w={w}")
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                result = img if img is not None else MISSING
            except urllib.error.HTTPError as e:
                result = MISSING if e.code == 404 else None
            except Exception:
                result = None                                        # try again on the next draw
            with self.lock:
                if result is not None:
                    if len(self.images) > 400:                       # drop the oldest cached picture
                        self.images.pop(next(iter(self.images)))
                    self.images[key] = result
                self.requested.discard(key)
                self.version += 1

    def image(self, event_id, which, w):
        """The picture if loaded, MISSING if the Pi has none, else None (and it gets fetched)."""
        key = (event_id, which, w)
        with self.lock:
            if key in self.images:
                return self.images[key]
            if key not in self.requested:
                self.requested.add(key)
                self.jobs.put((0 if w >= DETAIL_W else 1, next(self.job_n), key))
        return None

    def bucket(self, name):
        """Newest first."""
        with self.lock:
            return [e for e in reversed(self.items) if e["label"] == name]

    def counts(self):
        with self.lock:
            return {k: sum(e["label"] == k for e in self.items) for k, _, _ in BUCKETS}

    def get(self, event_id):
        with self.lock:
            return next((e for e in self.items if e["id"] == event_id), None)

    def set_label(self, event_id, label):
        """Applied here at once, then sent; the next refresh from the Pi settles any disagreement."""
        with self.lock:
            for e in self.items:
                if e["id"] == event_id:
                    e["label"] = label
            self.version += 1
        threading.Thread(target=self._send_label, args=(event_id, label), daemon=True).start()

    def delete(self, event_ids):
        """Removed here at once, then deleted on the Pi (permanently)."""
        ids = set(event_ids)
        with self.lock:
            self.items = [e for e in self.items if e["id"] not in ids]
            self.version += 1
        threading.Thread(target=self._send_delete, args=(sorted(ids),), daemon=True).start()

    def delete_bucket(self, name, max_id):
        """Everything with this label up to max_id (what's on screen), in one request."""
        with self.lock:
            self.items = [e for e in self.items if not (e["label"] == name and e["id"] <= max_id)]
            self.version += 1
        threading.Thread(target=self._send_delete_bucket, args=(name, max_id), daemon=True).start()

    def _send_delete(self, ids):
        for i in ids:
            try:
                self._call("DELETE", f"/events/{i}")
            except urllib.error.HTTPError as e:
                if e.code == 404 and self._supports_delete():        # already gone is fine
                    continue
                self._notify(self._explain("delete", e))
                break
            except Exception as e:
                self._notify(self._explain("delete", e))
                break
        self.wake.set()

    def _supports_delete(self):
        """A 404 on DELETE means 'no such event' only if the Pi's API knows deletes at all."""
        try:
            self._call("POST", "/events/delete", {"state": "bogus"})
        except urllib.error.HTTPError as e:
            return e.code == 400                                     # new API rejects the bogus state
        except Exception:
            return False
        return False

    def _send_delete_bucket(self, name, max_id):
        try:
            self._call("POST", "/events/delete", {"state": name, "max_id": max_id})
        except Exception as e:
            self._notify(self._explain("delete", e))
        self.wake.set()

    def _send_label(self, event_id, label):
        try:
            self._call("POST", f"/events/{event_id}/label", {"label": label})
        except Exception as e:
            self._notify(self._explain("label", e))
        self.wake.set()                                              # re-read the list right away


class EventBrowser:
    def __init__(self, connect, poll_s=5.0, window="efference events"):
        self.client = EventsClient(connect, poll_s)
        self.window = window
        self.view, self.bucket, self.page, self.sel, self.which = "home", "pending", 0, None, "peak"
        self.hits = []
        self.confirm = None                       # (what, when): a delete waiting for its second click
        self._arming = False
        self.open, self._created_at = True, None
        self._drawn = None
        self._canvas = None

    # ---- window plumbing -----------------------------------------------------------------
    def show(self):
        """Redraw if anything changed and put it on screen. Call from the main loop."""
        if not self.open:
            return
        if self._created_at is None:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window, W, H)
            cv2.setMouseCallback(self.window, self._on_mouse)
            self._created_at = time.monotonic()
        elif time.monotonic() - self._created_at > 2:
            try:
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    self.open, self._created_at = False, None        # closed with the window's X
                    return
            except cv2.error:
                pass
        state = (self.client.version, self.view, self.bucket, self.page, self.sel, self.which, int(time.time()))
        if state != self._drawn:
            self._drawn = state
            self._canvas = self.render()
        cv2.imshow(self.window, self._canvas)

    def toggle(self):
        self.open = not self.open
        if not self.open and self._created_at is not None:
            cv2.destroyWindow(self.window)
            self._created_at = None

    def _on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (x0, y0, x1, y1), action in reversed(self.hits):
            if x0 <= x < x1 and y0 <= y < y1:
                self._arming = False
                action()
                if not self._arming:
                    self.confirm = None               # any other click cancels a pending delete
                self._drawn = None
                return
        self.confirm = None

    def key(self, k):
        """k: a cv2.waitKey code (low byte). Returns True if it was used here."""
        if not self.open:
            return False
        c = chr(k) if 32 <= k < 127 else k
        if self.view == "detail":
            if c == "1":
                self.label_selected("safe")
            elif c == "2":
                self.label_selected("motion")
            elif c == "d":
                self.delete_selected()
            elif c in ("n", "p"):
                self.step(1 if c == "n" else -1)
            elif c in ("b", 27, 8, 127):
                self.view = "grid"
            else:
                return False
        elif self.view == "grid":
            if c in ("n", "p"):
                self.turn(1 if c == "n" else -1)
            elif c in ("b", 27, 8, 127):
                self.view = "home"
            else:
                return False
        else:
            return False
        if c != "d":
            self.confirm = None
        self._drawn = None
        return True

    # ---- navigation ----------------------------------------------------------------------
    def go(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def turn(self, d):
        pages = max(1, -(-len(self.client.bucket(self.bucket)) // PER_PAGE))
        self.page = int(np.clip(self.page + d, 0, pages - 1))

    def step(self, d):
        items = self.client.bucket(self.bucket)
        ids = [e["id"] for e in items]
        if self.sel in ids and items:
            self.sel = ids[int(np.clip(ids.index(self.sel) + d, 0, len(ids) - 1))]
            self.which = "peak"

    def label_selected(self, label):
        items = self.client.bucket(self.bucket)
        ids = [e["id"] for e in items]
        if self.sel not in ids or label == self.bucket:
            return
        i = ids.index(self.sel)
        self.client.set_label(self.sel, label)
        rest = ids[:i] + ids[i + 1:]                                  # the folder without this picture
        if rest:
            self.sel, self.which = rest[min(i, len(rest) - 1)], "peak"
        else:
            self.view, self.sel = "grid", None

    def armed(self, what):
        return self.confirm is not None and self.confirm[0] == what and time.time() - self.confirm[1] < 4.0

    def ask(self, what):
        """First call arms a delete; a second call within 4 s returns True (go ahead)."""
        if self.armed(what):
            self.confirm = None
            return True
        self.confirm, self._arming = (what, time.time()), True
        return False

    def delete_selected(self):
        items = self.client.bucket(self.bucket)
        ids = [e["id"] for e in items]
        if self.sel not in ids or not self.ask(("one", self.sel)):
            return
        i = ids.index(self.sel)
        self.client.delete([self.sel])
        rest = ids[:i] + ids[i + 1:]
        if rest:
            self.sel, self.which = rest[min(i, len(rest) - 1)], "peak"
        else:
            self.view, self.sel = "grid", None

    def delete_folder(self):
        items = self.client.bucket(self.bucket)
        if items and self.ask(("all", self.bucket)):
            self.client.delete_bucket(self.bucket, max(e["id"] for e in items))
            self.page = 0

    # ---- drawing -------------------------------------------------------------------------
    def button(self, canvas, rect, text, color, action, active=False, scale=0.6):
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color if active else (60, 60, 60), -1 if active else 1)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
        put(canvas, text, (x0 + (x1 - x0 - tw) // 2, (y0 + y1) // 2 + 6), (16, 16, 16) if active else color, scale)
        self.hits.append((rect, action))

    def header(self, canvas, crumbs):
        cv2.rectangle(canvas, (0, 0), (W, TOP), PANEL, -1)
        put(canvas, crumbs, (20, 36), FG, 0.7)
        c = self.client
        if c.notice and time.time() < c.notice[1]:
            status, color = c.notice[0], RED
        elif c.error:
            status, color = c.error, RED
        elif c.updated is None:
            status, color = "connecting...", DIM
        else:
            status, color = f"refreshed {int(time.time() - c.updated)} s ago (every {c.poll_s:g} s)", DIM
        if c.size_bytes is not None and not c.error and color != RED:
            status += f"   |   events use {c.size_bytes / 1e6:.0f} MB"
        tw = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        put(canvas, status, (W - tw - 20, 34), color, 0.5)

    def render(self):
        canvas = np.full((H, W, 3), BG, np.uint8)
        self.hits = []
        if self.view == "detail" and self.client.get(self.sel) is None:
            self.view = "grid"
        {"home": self.draw_home, "grid": self.draw_grid, "detail": self.draw_detail}[self.view](canvas)
        return canvas

    def thumb(self, canvas, event, x, y, w, h):
        img = self.client.image(event["id"], "peak", THUMB_W)
        if isinstance(img, np.ndarray):
            canvas[y:y + h, x:x + w] = fit(img, w, h)
        else:
            canvas[y:y + h, x:x + w] = (16, 16, 16)
            put(canvas, "no image" if img is MISSING else "loading...", (x + 10, y + h // 2), DIM, 0.5)

    def draw_home(self, canvas):
        self.header(canvas, "Events saved by the Pi")
        counts = self.client.counts()
        tile_w, gap = 380, 30
        x0 = (W - 3 * tile_w - 2 * gap) // 2
        for i, (key, name, color) in enumerate(BUCKETS):
            x, y = x0 + i * (tile_w + gap), 110
            cv2.rectangle(canvas, (x, y), (x + tile_w, y + 480), PANEL, -1)
            cv2.rectangle(canvas, (x, y), (x + tile_w, y + 6), color, -1)
            put(canvas, name, (x + 20, y + 52), color, 0.95, 2)
            put(canvas, str(counts[key]), (x + 20, y + 112), FG, 1.6, 2)
            put(canvas, "picture" if counts[key] == 1 else "pictures", (x + 20 + 40 * len(str(counts[key])) + 24, y + 112),
                DIM, 0.55)
            for j, e in enumerate(self.client.bucket(key)[:4]):
                self.thumb(canvas, e, x + 20 + (j % 2) * 175, y + 150 + (j // 2) * 110, 165, 96)
            if not counts[key]:
                put(canvas, "empty", (x + 20, y + 200), DIM, 0.55)
            put(canvas, "click to open", (x + 20, y + 460), DIM, 0.5)
            self.hits.append(((x, y, x + tile_w, y + 480),
                              lambda key=key: self.go(view="grid", bucket=key, page=0)))

    def draw_grid(self, canvas):
        items = self.client.bucket(self.bucket)
        pages = max(1, -(-len(items) // PER_PAGE))
        self.page = int(np.clip(self.page, 0, pages - 1))
        self.header(canvas, f"Events  >  {NAME[self.bucket]}  ({len(items)})")
        cw, ch, x0, y0 = 240, 190, 20, TOP + 16
        gap = (W - 2 * x0 - GRID_COLS * cw) // (GRID_COLS - 1)
        for n, e in enumerate(items[self.page * PER_PAGE:(self.page + 1) * PER_PAGE]):
            x, y = x0 + (n % GRID_COLS) * (cw + gap), y0 + (n // GRID_COLS) * ch
            self.thumb(canvas, e, x, y, cw, 135)
            when = e["time"][5:16].replace("-", "/")
            put(canvas, f"#{e['id']}  {when}", (x, y + 154), FG, 0.45)
            put(canvas, f"score {e['score']:.3g}" if e["score"] is not None else "", (x, y + 172), DIM, 0.45)
            self.hits.append(((x, y, x + cw, y + ch - 10), lambda e=e: self.go(view="detail", sel=e["id"], which="peak")))
        if not items:
            put(canvas, "nothing here", (W // 2 - 70, 300), DIM, 0.7)
        by = H - 56
        self.button(canvas, (20, by, 140, by + 38), "< Back", FG, lambda: self.go(view="home"))
        self.button(canvas, (W // 2 - 190, by, W // 2 - 80, by + 38), "< Prev", FG, lambda: self.turn(-1))
        put(canvas, f"page {self.page + 1} / {pages}", (W // 2 - 55, by + 26), DIM, 0.55)
        self.button(canvas, (W // 2 + 80, by, W // 2 + 190, by + 38), "Next >", FG, lambda: self.turn(1))
        if items:
            armed = self.armed(("all", self.bucket))
            self.button(canvas, (W - 360, by, W - 20, by + 38),
                        f"Click again: delete all {len(items)}" if armed else f"Delete all {len(items)} here",
                        RED_BTN, self.delete_folder, active=armed, scale=0.55)

    def draw_detail(self, canvas):
        e = self.client.get(self.sel)
        items = self.client.bucket(self.bucket)
        ids = [x["id"] for x in items]
        pos = ids.index(self.sel) + 1 if self.sel in ids else 0
        self.header(canvas, f"Events  >  {NAME[self.bucket]}  >  #{e['id']}   ({pos} of {len(ids)})")
        which = self.which if e["has_before"] else "peak"
        img = self.client.image(e["id"], which, DETAIL_W)
        area = (20, TOP + 14, DETAIL_W, 540)
        if isinstance(img, np.ndarray):
            canvas[area[1]:area[1] + area[3], area[0]:area[0] + area[2]] = fit(img, area[2], area[3])
        else:
            canvas[area[1]:area[1] + area[3], area[0]:area[0] + area[2]] = (16, 16, 16)
            put(canvas, "no image" if img is MISSING else "loading...", (area[0] + 20, area[1] + 40), DIM, 0.6)
        if self.sel in ids and pos < len(ids):                          # warm the next picture
            self.client.image(ids[pos], "peak", DETAIL_W)
        put(canvas, "earlier frame (~1 s before)" if which == "before" else "the moment of highest surprise",
            (20, TOP + 14 + 540 + 28), DIM, 0.5)
        x0 = 1000
        put(canvas, e["time"], (x0, TOP + 40), FG, 0.6)
        put(canvas, f"surprise score {e['score']:.3g}" if e["score"] is not None else "", (x0, TOP + 66), DIM, 0.5)
        put(canvas, f"currently: {NAME[e['label']]}", (x0, TOP + 96), COLOR[e["label"]], 0.55)
        self.button(canvas, (x0, 190, x0 + 260, 260), "Safe  [1]", COLOR["safe"],
                    lambda: self.label_selected("safe"), active=e["label"] == "safe", scale=0.75)
        self.button(canvas, (x0, 280, x0 + 260, 350), "Motion  [2]", COLOR["motion"],
                    lambda: self.label_selected("motion"), active=e["label"] == "motion", scale=0.75)
        if e["has_before"]:
            self.button(canvas, (x0, 380, x0 + 260, 420), "show latest frame" if which == "before" else "show earlier frame",
                        FG, lambda: self.go(which="peak" if which == "before" else "before"))
        self.button(canvas, (x0, 450, x0 + 125, 490), "< Prev [p]", FG, lambda: self.step(-1))
        self.button(canvas, (x0 + 135, 450, x0 + 260, 490), "Next [n] >", FG, lambda: self.step(1))
        self.button(canvas, (x0, 520, x0 + 260, 560), "Back to list [b]", FG, lambda: self.go(view="grid"))
        armed = self.armed(("one", e["id"]))
        self.button(canvas, (x0, 590, x0 + 260, 630), "Click again: delete for good" if armed else "Delete  [d]",
                    RED_BTN, self.delete_selected, active=armed, scale=0.55)
        put(canvas, "labeling or deleting moves on to the next picture", (x0, 660), DIM, 0.42)
