# Recording and labeling household video

All training and evaluation data comes from our own recordings. The workflow:

1. **Record** on the Pi. `tools/record.py` on the Mac starts it and, when you stop, copies
   the episode to `recordings/` on the Mac.
2. **Label** afterwards on the Mac with `tools/label.py`: scrub through the video and mark
   anomaly intervals.

Same convention as standard anomaly benchmarks: **train = ordinary life only, test = contains
events you label as anomalies.** The model trains on normal video and is scored on how well its
surprise picks out the labeled frames.

## 1. Record

Run these from the project folder on the Mac:

```bash
.venv/bin/python tools/record.py --scene kitchen --split train          # ordinary life
.venv/bin/python tools/record.py --scene kitchen --split test --view    # staged events, with live viewer
```

- Press **Ctrl-C** to stop. The Pi finishes writing, then every episode not yet on the Mac is copied over.
- `--view` opens the live viewer so you can frame the shot. It's optional; recording doesn't depend on it.
- `--duration 600` stops automatically after 10 minutes.
- `--pull-only` just copies any episodes still on the Pi (for example after a WiFi drop).
- Recording happens on the Pi's SD card, so a WiFi hiccup can't drop frames; only the copy needs the network.

**Episodes** land in `recordings/<scene>_<split>_<date>_<time>/`:

| File | What it is |
|---|---|
| `preview.mp4` | color video, 640×360 at 30 fps, for you to watch. Video frame *i* = logged frame *i* |
| `chunk_*.npz` | model data: 320×180 grayscale frames (lossless) + timestamps + eye state |
| `meta.json` | scene, split, camera settings, frame counts |
| `labels.json`, `labels.npy` | written by the labeler (see below) |

**Size:** about 1.25 MB/s, so ~4.5 GB per hour. The Pi has about 100 GB free. Copying to the Mac
takes a few minutes per hour of video over WiFi. Episodes stay on the Pi after copying, so delete
old ones there once you're happy with the Mac copies.

## 2. Label

```bash
.venv/bin/python tools/label.py recordings/kitchen_test_20261018_101500
.venv/bin/python tools/label.py          # newest episode
```

| Key | Action |
|---|---|
| SPACE | play / pause |
| `,` `.` (or ← →) | one frame back / forward |
| `j` `l` | 1 s back / forward |
| `J` `L` (or ↓ ↑) | 10 s back / forward |
| `-` `=` | slower / faster (0.25×–8×) |
| `[` | mark anomaly **start** at the current frame (press again to move it) |
| `]` | mark anomaly **end** here, which adds the interval |
| `x` | delete the interval under the playhead |
| `n` `p` | jump to the next / previous interval |
| click / drag the timeline | seek |
| `s` | save |
| `q` / Esc / close window | save and quit |

Intervals show as red blocks on the timeline, and the video gets a red border while the playhead
is inside one. Overlapping intervals merge on save.

**Output:** `labels.json` (intervals in frame numbers, `"source": "offline"`) and `labels.npy`
(one 0/1 per frame). Opening and quitting an episode with no intervals still saves, which records
"reviewed, nothing anomalous". That's different from "never labeled" (no `labels.json`).

Tips:
- Mark the interval from the first frame where something is visibly off to the last. Frame-step
  with `,` `.` at the edges; offline labels don't suffer the reaction-time lag of live labeling.
- Review **train** episodes too, at 4–8×. If something odd happened by accident, label it so it
  can be kept out of training.

The live space-bar labeling in the viewer still works during a recording (`labels.json` with
`"source": "live"`), and the labeler opens those intervals for correction. Offline labeling is
the main workflow, though.

## What to record

**Train (ordinary life):** aim for a few hours in total, spread across:
- several rooms or camera positions (use `--scene` to name each), so the model doesn't learn one room
- morning, afternoon, evening and lamp-only lighting
- people coming and going normally, pets, TV on, fan running, curtains moving

**Test (events to label):** short episodes, 2–10 minutes each, ideally in the same scenes as
training, and also a scene that's **never** in training to check generalization. Ideas:
- someone enters an empty room, or something is dropped or thrown
- a light switches on or off, or a door slams
- an object appears or disappears, or the camera gets bumped
- unusual motion: running indoors, something falling over

Also include stretches of **predictable motion with nothing anomalous** (fan, TV, curtains)
inside test episodes. Those frames should score low, and they're where simple frame
differencing fails.

**Keep test scenes honest:** record test episodes on different days from the training episodes
of the same scene, so the model can't just memorize that afternoon's lighting.

## Evaluation (later)

Frame-level AUROC: does the surprise score rank labeled frames above unlabeled ones? Computed
over all test episodes concatenated, and reported separately for:
1. scenes seen in training
2. scenes never seen in training (generalization)
3. predictable-motion stretches only (false-alarm rate)
