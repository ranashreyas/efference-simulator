"""HTTP API over the event store, for labeling from elsewhere (the viewer's event window, a script,
curl). It runs inside the eye loop (`python -m eye.run --events`) as a background thread, so there
is one process and one command; the events themselves live on disk, so they survive restarts.

Every request except /health needs `Authorization: Bearer <token>`. The token is created on first
start and kept in ~/.config/efference/events.env (EVENTS_TOKEN=...). It listens on localhost only;
reach it through an authenticated ssh tunnel, since these are pictures of your home (the viewer
does this for you).

    GET  /health
    GET  /stats                         counts by label, and how many were never fetched
    GET  /events?state=&since_id=&limit=   metadata only, read-only. state: all|pending|safe|motion
    POST /events/fetch                  the events never handed out before, oldest first, with
                                        images inline (base64 JPEG, max_w pixels wide); marks them
                                        fetched. body (all optional): {"limit": 20, "max_w": 800}
    GET  /events/<id>/image?which=peak|before&w=800     the JPEG itself
    POST /events/<id>/label             body {"label": "safe" | "motion"}; relabeling overwrites
    DELETE /events/<id>                 permanently delete one event and its images
    POST /events/delete                 body {"state": "pending|safe|motion", "max_id": N}: delete every
                                        event with that label (only ids <= max_id if given)
    GET  /export                        every event with its label and times (including control
                                        frames' `kind`), for evaluation on the Mac
"""
import base64
import hmac
import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .events import STATES, EventStore

MAX_BODY = 4096


def resized_jpeg(data, max_w):
    if not max_w:
        return data
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.shape[1] <= max_w:
        return data
    h = int(img.shape[0] * max_w / img.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (max_w, h), interpolation=cv2.INTER_AREA),
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else data


class ImageCache:
    """Recently served resized JPEGs, so a page of thumbnails isn't re-decoded every 5 s on the Pi."""

    def __init__(self, size=400):
        self.size, self.items, self.lock = size, {}, threading.Lock()

    def get(self, key, make):
        with self.lock:
            if key in self.items:
                return self.items[key]
        data = make()
        with self.lock:
            self.items[key] = data
            if len(self.items) > self.size:
                self.items.pop(next(iter(self.items)))
        return data


def public(ev):
    """What a reviewer sees. `kind` (flagged vs control frame) is deliberately hidden."""
    return dict(id=ev["id"], time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["t_wall"])),
                score=round(ev["score"], 2) if ev["score"] is not None else None, label=ev["label"],
                has_before=bool(ev["image_before"]))


def make_handler(store, token, default_w):
    cache = ImageCache()

    class Handler(BaseHTTPRequestHandler):
        server_version = "efference-events"

        def log_message(self, fmt, *args):
            print(f"[api] {self.address_string()} {fmt % args}", flush=True)

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            given = self.headers.get("Authorization", "")
            if token is None or hmac.compare_digest(given, f"Bearer {token}"):
                return True
            self._send(401, {"error": "missing or wrong bearer token"})
            return False

        def _json_body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            if url.path == "/health":
                return self._send(200, {"ok": True})
            if not self._authed():
                return
            try:
                if url.path == "/stats":
                    return self._send(200, dict(store.counts(), bytes=store.size_bytes()))
                if url.path == "/events":
                    state = q.get("state", "all")
                    if state != "all" and state not in STATES:
                        return self._send(400, {"error": f"state must be all or one of {STATES}"})
                    rows = store.list(state, int(q.get("since_id", 0)), min(int(q.get("limit", 50)), 200))
                    return self._send(200, [public(r) for r in rows])
                if url.path == "/export":
                    return self._send(200, store.list("all", 0, 10 ** 9))
                m = re.fullmatch(r"/events/(\d+)/image", url.path)
                if m:
                    ev = store.get(int(m.group(1)))
                    which = q.get("which", "peak")
                    rel = ev and (ev["image_before"] if which == "before" else ev["image"])
                    data = rel and store.read_image(rel)
                    if not data:
                        return self._send(404, {"error": "no such image"})
                    w = int(q.get("w", default_w))
                    return self._send(200, cache.get((rel, w), lambda: resized_jpeg(data, w)), "image/jpeg")
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._authed():
                return
            url = urlparse(self.path)
            try:
                body = self._json_body()
                if url.path == "/events/fetch":
                    rows = store.list("all", 0, min(int(body.get("limit", 20)), 100), unfetched=True)
                    w = int(body.get("max_w", default_w))
                    out = []
                    for r in rows:
                        data = store.read_image(r["image"])
                        if data is None:
                            continue
                        e = public(r)
                        e["image_b64"] = base64.b64encode(resized_jpeg(data, w)).decode()
                        out.append(e)
                    store.mark_fetched([e["id"] for e in out])      # only what was actually serialized
                    return self._send(200, out)
                if url.path == "/events/delete":
                    state = body.get("state")
                    if state not in STATES:
                        return self._send(400, {"error": f"state must be one of {STATES}"})
                    max_id = body.get("max_id")
                    n = store.delete(state=state, max_id=int(max_id) if max_id is not None else None)
                    print(f"[api] deleted {n} {state} events", flush=True)
                    return self._send(200, {"deleted": n})
                m = re.fullmatch(r"/events/(\d+)/label", url.path)
                if m:
                    label = body.get("label")
                    if isinstance(label, str):
                        label = label.strip().lower()
                    eid = int(m.group(1))
                    if label not in ("safe", "motion"):
                        return self._send(400, {"error": 'label must be "safe" or "motion"'})
                    if not store.set_label(eid, label):
                        return self._send(404, {"error": "no such event"})
                    return self._send(200, public(store.get(eid)))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            self._send(404, {"error": "not found"})

        def do_DELETE(self):
            if not self._authed():
                return
            m = re.fullmatch(r"/events/(\d+)", urlparse(self.path).path)
            if not m:
                return self._send(404, {"error": "not found"})
            n = store.delete(ids=[int(m.group(1))])
            if not n:
                return self._send(404, {"error": "no such event"})
            print(f"[api] deleted event {m.group(1)}", flush=True)
            self._send(200, {"deleted": n})

    return Handler


TOKEN_FILE = Path.home() / ".config" / "efference" / "events.env"


def load_or_create_token(path=TOKEN_FILE):
    """The API token: $EVENTS_TOKEN if set, else the one in `path`, else a new random one written there."""
    token = os.environ.get("EVENTS_TOKEN", "").strip()
    if token:
        return token
    path = Path(path)
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("EVENTS_TOKEN="):
                return line.split("=", 1)[1].strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    token = base64.urlsafe_b64encode(os.urandom(24)).decode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(f"EVENTS_TOKEN={token}\n")
    print(f"[api] created an API token in {path}", flush=True)
    return token


def start_server(store, token, host="127.0.0.1", port=8787, image_width=800, prune_hours=1.0):
    """Serve `store` on a background thread; returns the server (call .shutdown() to stop). Also
    prunes old pending events every `prune_hours`."""
    srv = ThreadingHTTPServer((host, port), make_handler(store, token, image_width))
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    if prune_hours > 0:
        def prune_loop():
            while True:
                time.sleep(prune_hours * 3600)
                try:
                    print(f"[api] pruned {store.prune()} old pending events", flush=True)
                except Exception as e:
                    print(f"[api] prune failed: {e!r}", flush=True)
        threading.Thread(target=prune_loop, daemon=True).start()
    return srv
