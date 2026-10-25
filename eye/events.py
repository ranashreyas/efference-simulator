"""Surprise events: detect them in the live loop, keep them on disk, hand them out for labeling.

    eye.run  ─► EventDetector ─► EventSink (worker thread) ─► EventStore ◄─ eye.api (HTTP thread, same process)
                                                              events.db + images/*.jpg

An event is a stretch of frames whose surprise score stays above a threshold; we keep its peak
frame, as the triggering eye saw it (the panning eye's crop window, not the whole camera frame),
and a frame from ~1 s earlier for context (a panning eye will have moved since). Every event starts
`pending`; only a person's label ("safe" or "motion") moves it out of that state. Some
*control* frames (random moments, not flagged by the model) are stored the same way and
indistinguishable to whoever labels them. They show whether the detector misses things:
labeling only flagged frames can measure false alarms, never misses.

Persistence: SQLite (WAL) for metadata and labels, JPEGs as plain files. Files are written to a
temp name, fsynced and renamed before their row is inserted, so a crash leaves at most an
orphan file, which the next start removes. The store is safe to open from several processes
(SQLite WAL), so a script can read it while the eye loop runs.
"""
import collections
import contextlib
import os
import random
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

LABELS = ("safe", "motion")
STATES = ("pending",) + LABELS
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL CHECK (kind IN ('event', 'control')),
    t_wall      REAL NOT NULL,            -- unix time of the peak frame
    t_mono      REAL NOT NULL,            -- camera monotonic time (aligns with logged episodes)
    seq         INTEGER,                  -- camera frame counter
    score       REAL,                     -- surprise score at the peak frame
    eye         TEXT,                     -- which pipeline produced it (null = still, virtual = panning)
    image       TEXT NOT NULL,            -- peak frame, relative to the store root
    image_before TEXT,                    -- frame ~1 s earlier, same
    label       TEXT NOT NULL DEFAULT 'pending' CHECK (label IN ('pending', 'safe', 'motion')),
    labeled_at  REAL,
    fetched_at  REAL                      -- when the API first handed it out
);
CREATE INDEX IF NOT EXISTS events_label ON events (label);
"""


class EventStore:
    """Events on disk. Safe to open from several processes and threads at once."""

    def __init__(self, root, max_pending_days=7.0, max_gb=2.0):
        self.root = Path(root)
        self.images = self.root / "images"
        self.images.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "events.db"
        self.max_pending_days, self.max_gb = max_pending_days, max_gb
        with self._conn() as c:
            c.executescript(SCHEMA)
        self.reconcile()
        self.prune()

    @contextlib.contextmanager
    def _conn(self):
        c = sqlite3.connect(self.db_path, timeout=10)
        c.row_factory = sqlite3.Row
        try:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=FULL")          # events are rare; never lose a label
            with c:
                yield c
        finally:
            c.close()

    # ---- writing -------------------------------------------------------------------------
    def _write_jpeg(self, img, quality):
        name = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        tmp = self.images / (name + ".tmp")
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.images / name)
        return f"images/{name}"

    def add(self, kind, t_wall, t_mono, seq, score, eye, img, img_before=None, quality=85):
        """Store one event; returns its id. The row appears only after its images are safely on disk."""
        path = self._write_jpeg(img, quality)
        before = self._write_jpeg(img_before, quality) if img_before is not None else None
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO events (kind, t_wall, t_mono, seq, score, eye, image, image_before) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (kind, t_wall, t_mono, seq, score, eye, path, before))
            return cur.lastrowid

    def set_label(self, event_id, label):
        if label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")
        with self._conn() as c:
            cur = c.execute("UPDATE events SET label=?, labeled_at=? WHERE id=?", (label, time.time(), event_id))
            return cur.rowcount == 1

    def mark_fetched(self, ids):
        with self._conn() as c:
            c.executemany("UPDATE events SET fetched_at=? WHERE id=? AND fetched_at IS NULL",
                          [(time.time(), i) for i in ids])

    # ---- reading -------------------------------------------------------------------------
    def get(self, event_id):
        with self._conn() as c:
            row = c.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return dict(row) if row else None

    def list(self, state="all", since_id=0, limit=50, unfetched=False):
        """state: 'all', 'pending', 'safe' or 'motion'. Oldest first."""
        q, args = "SELECT * FROM events WHERE id > ?", [since_id]
        if state != "all":
            q += " AND label = ?"
            args.append(state)
        if unfetched:
            q += " AND fetched_at IS NULL"
        q += " ORDER BY id LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q, args)]

    def counts(self):
        with self._conn() as c:
            rows = c.execute("SELECT label, COUNT(*) n FROM events GROUP BY label").fetchall()
            unfetched = c.execute("SELECT COUNT(*) FROM events WHERE fetched_at IS NULL").fetchone()[0]
        out = {s: 0 for s in STATES}
        out.update({r["label"]: r["n"] for r in rows})
        out["unfetched"] = unfetched
        return out

    def read_image(self, rel):
        try:
            return (self.root / rel).read_bytes()
        except OSError:
            return None

    def size_bytes(self):
        return sum(f.stat().st_size for f in self.images.iterdir() if f.is_file())

    def delete(self, ids=None, state=None, max_id=None):
        """Permanently delete events (rows and their images): the listed ids, or every event with
        label `state` (optionally only those with id <= max_id, so ones that arrived after you
        looked are spared). Returns how many were deleted."""
        with self._conn() as c:
            if ids is not None:
                rows = [r for i in ids for r in c.execute("SELECT * FROM events WHERE id=?", (i,))]
            else:
                q, args = "SELECT * FROM events WHERE label=?", [state]
                if max_id is not None:
                    q += " AND id <= ?"
                    args.append(max_id)
                rows = c.execute(q, args).fetchall()
        self._delete(rows)
        return len(rows)

    # ---- housekeeping --------------------------------------------------------------------
    def reconcile(self):
        """After a crash: delete image files no row points at (and half-written temp files)."""
        with self._conn() as c:
            known = set()
            for r in c.execute("SELECT image, image_before FROM events"):
                known.update(p for p in (r["image"], r["image_before"]) if p)
        for f in self.images.iterdir():
            if f"images/{f.name}" not in known:
                f.unlink(missing_ok=True)

    def _delete(self, rows):
        with self._conn() as c:
            c.executemany("DELETE FROM events WHERE id=?", [(r["id"],) for r in rows])
        for r in rows:
            for p in (r["image"], r["image_before"]):
                if p:
                    (self.root / p).unlink(missing_ok=True)

    def prune(self):
        """Labeled events are kept forever (they're the valuable, small part). Pending ones are
        dropped after max_pending_days, and oldest-first if the images pass max_gb."""
        cutoff = time.time() - self.max_pending_days * 86400
        with self._conn() as c:
            old = c.execute("SELECT * FROM events WHERE label='pending' AND t_wall < ?", (cutoff,)).fetchall()
        self._delete(old)
        size = sum(f.stat().st_size for f in self.images.iterdir())
        if size > self.max_gb * 1e9:
            with self._conn() as c:
                pend = c.execute("SELECT * FROM events WHERE label='pending' ORDER BY id").fetchall()
            drop = []
            for r in pend:
                if size <= self.max_gb * 1e9:
                    break
                for p in (r["image"], r["image_before"]):
                    if p and (self.root / p).exists():
                        size -= (self.root / p).stat().st_size
                drop.append(r)
            self._delete(drop)
        return len(old)


class EventDetector:
    """Turns a per-frame surprise score into discrete events.

    idle    ─ score >= threshold for `min_frames` frames in a row ─►  active
    active  ─ track the highest-scoring frame; the event ends once the score has stayed below
              threshold * release for `release_frames` frames, or after `max_event_s`
    then    ─ emit the peak frame, and ignore further events for `cooldown_s` (counted, not stored)

    Also emits `control` frames at random moments (mean gap `control_mean_s`, 0 = none), never
    while an event is active or just ended.
    """

    def __init__(self, emit, threshold=60.0, min_frames=3, release=0.5, release_frames=10,
                 max_event_s=5.0, cooldown_s=30.0, control_mean_s=600.0, before_s=1.0, seed=None):
        self.emit = emit
        self.threshold, self.min_frames = threshold, min_frames
        self.release, self.release_frames = release, release_frames
        self.max_event_s, self.cooldown_s = max_event_s, cooldown_s
        self.control_mean_s, self.before_s = control_mean_s, before_s
        self.rng = random.Random(seed)
        self.sparse = collections.deque(maxlen=10)      # every 3rd frame: (t, frame), ~1 s of context
        self._n = 0
        self.state, self.above, self.below = "idle", 0, 0
        self.peak = None                                # (score, t, seq, img, before)
        self.start_t = 0.0
        self.quiet_until = 0.0
        self.next_control = None
        self.suppressed = 0

    def _before(self, t):
        """A stored frame at least before_s old (the oldest one if none is that old)."""
        old = [f for f in self.sparse if t - f[0] >= self.before_s]
        return (old[-1] if old else (self.sparse[0] if self.sparse else (None, None)))[1]

    def update(self, t, score, img, seq):
        """Call once per frame with the pipeline's surprise score (None = no score yet)."""
        self._n += 1
        if score is None:
            return
        if self.next_control is None and self.control_mean_s:
            self.next_control = t + self.rng.expovariate(1 / self.control_mean_s)
        if self.state == "idle":
            if score >= self.threshold:
                self.above += 1
                if self.peak is None or score > self.peak[0]:
                    self.peak = (score, t, seq, img, self._before(t))
                if self.above >= self.min_frames:
                    self.state, self.start_t, self.below = "active", t, 0
            else:
                self.above, self.peak = 0, None
        else:
            if score > self.peak[0]:
                self.peak = (score, t, seq, img, self.peak[4])
            self.below = self.below + 1 if score < self.threshold * self.release else 0
            if self.below >= self.release_frames or t - self.start_t >= self.max_event_s:
                self._finish(t)
        if (self.state == "idle" and self.next_control is not None and t >= self.next_control
                and t >= self.quiet_until and self.above == 0):
            self.emit("control", t, seq, score, img, self._before(t))
            self.next_control = t + self.rng.expovariate(1 / self.control_mean_s)
        if self._n % 3 == 0:
            self.sparse.append((t, img))

    def _finish(self, t):
        score, pt, seq, img, before = self.peak
        if pt >= self.quiet_until:
            self.emit("event", pt, seq, score, img, before)
            self.quiet_until = t + self.cooldown_s
        else:
            self.suppressed += 1
        self.state, self.above, self.below, self.peak = "idle", 0, 0, None
        self.quiet_until = max(self.quiet_until, t + 5.0)


class EventSink:
    """Writes events on a worker thread, so JPEG encoding never stalls the camera loop."""

    def __init__(self, store, eye, quality=85):
        self.store, self.eye, self.quality = store, eye, quality
        self.pool = ThreadPoolExecutor(1)
        self.lock = threading.Lock()
        self.stored = 0

    def __call__(self, kind, t_mono, seq, score, img, before):
        t_wall = time.time() - (time.monotonic() - t_mono)
        self.pool.submit(self._store, kind, t_wall, t_mono, seq, score, img, before)

    def _store(self, kind, t_wall, t_mono, seq, score, img, before):
        try:
            i = self.store.add(kind, t_wall, t_mono, seq, score, self.eye, img, before, self.quality)
            with self.lock:
                self.stored += 1
            print(f"[events] stored {kind} #{i} (score {score:.3g})", flush=True)
        except Exception as e:                           # never let storage kill the camera loop
            print(f"[events] FAILED to store {kind}: {e!r}", flush=True)

    def close(self):
        self.pool.shutdown(wait=True)
