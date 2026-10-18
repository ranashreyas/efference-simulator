# Recording and labeling

The model learns to predict the next frame from ordinary household video. Frames it predicts
badly ("surprise") should be real events: someone walking into frame, fast movement.

1. **Record** ordinary life in sessions. Just live normally; nothing special to do.
2. **Label events** in some recordings, if you want to measure detection. Optional.

## Record

From the project folder on the Mac:

```bash
.venv/bin/python tools/record.py                          # record until Ctrl-C
.venv/bin/python tools/record.py --scene kitchen --view   # name the session, open the live viewer
.venv/bin/python tools/record.py --pull-only              # just copy episodes still on the Pi
```

Sessions can be any length; 10–30 minutes keeps files manageable (`--duration 1200` stops after
20 minutes). Press **Ctrl-C** to stop: the Pi finishes writing, then new episodes are copied to
`recordings/<scene>_<date>_<time>/` and checked. `--scene` is just a name (default `home`).

Each episode holds `preview.mp4` (color video to watch), `chunk_*.npz` (model data), `meta.json`,
and, once labeled, `labels.json` + `labels.npy`. Video is about 4.5 GB per hour; episodes stay on
the Pi after copying, so delete old ones there now and then.

## Label events (optional)

```bash
.venv/bin/python tools/label.py recordings/kitchen_20261018_101500
.venv/bin/python tools/label.py          # newest episode
```

| Key | Action |
|---|---|
| SPACE | play / pause |
| `,` `.` (or ← →) | one frame back / forward |
| `j` `l` | 1 s back / forward |
| `J` `L` (or ↓ ↑) | 10 s back / forward |
| `-` `=` | slower / faster (0.25×–8×) |
| `[` | mark event **start** (press again to move it) |
| `]` | mark event **end**, which adds the interval |
| `x` | delete the interval under the playhead |
| `n` `p` | jump to the next / previous interval |
| click / drag the timeline | seek |
| `s` | save |
| `q` / Esc / close window | save and quit |

Mark events that happened naturally: a person entering the frame, fast motion. Saving writes
`labels.json` (intervals) and `labels.npy` (one 0/1 per frame).

## How the data is used

The model trains on **all** recorded video except the last 15% of each recording, which is held
out automatically (`holdout_split` in `eye/episode.py`). Event labels inside that held-out part
are used to score whether surprise spikes on real events. Labels never affect training.
