# Efference Simulator

A camera "eye" with a small learned **world model**. It watches a room, predicts every next
frame, and flags what it didn't see coming: someone walking into view, fast movement. The eye
can also pan (here, a simulated servo moving a crop window across the camera image), and the
model receives an **efference copy** of its own motor commands, so its own movement doesn't
register as surprise.

```
Raspberry Pi 5 + USB webcam                          Mac
┌────────────────────────────────────────────┐       ┌───────────────────────────────────┐
│ camera ─► static eye  ─► model ─► surprise ─┼─5555─►│ viewer: actual | surprise | diff  │
│        └► panning eye ─► model ─► surprise ─┼─5556─►│ viewer (panning)                  │
│           (simulated servo, sweep/random/   │◄─5560─┤ remote control (arrow keys)       │
│            remote control)                  │       │                                   │
│ recorder ─► recordings/ ───────── rsync ────┼──────►│ recordings/ ─► label ─► train     │
│ models/ ◄────────────────────── pi_sync.sh ─┼───────┤ ◄─ export ◄─────────────────┘     │
└────────────────────────────────────────────┘       └───────────────────────────────────┘
```

This README takes you from blank hardware to the current state: a trained model running live
on the Pi, with a still eye and a panning eye side by side.

---

## 1. Hardware

| | Used here | Notes |
|---|---|---|
| Raspberry Pi | Pi 5, 8 GB | Runs capture, both eyes' models and recording at 30 fps (~17 ms per frame). A 4 GB Pi 5 should also work. |
| Power supply | Official 27 W USB-C | A weak supply causes dropouts. |
| microSD card | 128 GB | Recording uses about **4.5 GB per hour**. |
| Webcam | Logitech C922 Pro (USB) | Any UVC webcam that does 1280×720 MJPEG at 30 fps works. Pan/tilt angles assume the C922's 78° field of view (`C922_FOCAL_PX_720P` in `eye/actuators.py`). |
| Computer | MacBook Pro, Intel i5, 16 GB | Viewer, labeling and training (CPU only). Apple-silicon Macs work too (see 3.3). |
| Network | Pi and Mac on the same WiFi | The Pi is reached as `raspberrypi.local`. |

No servos are needed: the panning eye is simulated.

## 2. Set up the Raspberry Pi

1. **Flash the SD card** with [Raspberry Pi Imager](https://www.raspberrypi.com/software/): Raspberry Pi OS (64-bit).
   In the OS customisation settings, set the hostname to `raspberrypi`, choose a username and password,
   enter your WiFi, and **enable SSH**. Boot the Pi with the webcam plugged into a USB port.
2. **Check you can log in** from the Mac (replace `<user>` with your Pi username):
   ```bash
   ssh <user>@raspberrypi.local
   ```
3. **Install your Mac's SSH key** so the tools can log in without a password:
   ```bash
   ssh-copy-id <user>@raspberrypi.local
   ```
4. **On the Pi**, make sure `v4l2-ctl` is installed (used to lock the camera's exposure, white
   balance and focus) and the camera is detected:
   ```bash
   sudo apt install -y v4l-utils && v4l2-ctl --list-devices
   ```

## 3. Set up the Mac

### 3.1 Get the code
Clone or copy this repository onto the Mac. The Mac copy is the master; the Pi gets copies of it.

The tools default to the Pi user `ranashreyas`. If yours differs, set it once per terminal
(`tools/pi_sync.sh` reads `PI`; `tools/record.py` takes `--host`):
```bash
export PI=<user>@raspberrypi.local
```

### 3.2 Viewer environment
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-mac.txt
```

### 3.3 Training environment
PyTorch's last build for **Intel Macs is 2.2.2**, which needs **Python 3.12 or older**, so
training has its own environment (Python 3.11 here, from Anaconda):
```bash
python3.11 -m venv .venv-train && .venv-train/bin/pip install -r requirements-train.txt
```
On an Apple-silicon Mac you can drop the version pins in `requirements-train.txt` and use a
current PyTorch.

### 3.4 Put the code on the Pi and install its environment
```bash
tools/pi_sync.sh
ssh $PI 'cd ~/Desktop/efference-simulator && python3 -m venv .venv && .venv/bin/pip install -r requirements-pi.txt'
```
`tools/pi_sync.sh` copies the code to `~/Desktop/efference-simulator` on the Pi (override with
`PI_DIR`). Run it again after every code change. It never touches the Pi's `.venv`,
`recordings/` or `models/`.

### 3.5 Check the camera and stream
Start the eye loop on the Pi (no model yet), then the viewer on the Mac in a second terminal:
```bash
ssh -t $PI 'cd ~/Desktop/efference-simulator && .venv/bin/python -u -m eye.run'
```
```bash
.venv/bin/python -m viewer.view
```
You should see the live camera at 30 fps (the surprise panes stay empty without a model). On
startup the camera auto-adjusts for 2 s, then exposure, gain, white balance and focus are locked,
so the camera's auto modes can never look like motion. A dim room gets long exposures; more light
gives sharper frames.

## 4. Record training video

Point the camera at the space you want to watch and record ordinary life, with nothing staged.
45 minutes in one spot was enough for the current model; more time and varied lighting help.
```bash
.venv/bin/python tools/record.py --scene kitchen --duration 1800      # 30 min; Ctrl-C to stop early
```
Recording happens on the Pi's SD card, then the episode is copied to `recordings/` on the Mac and
checked. Details, and the optional event labeler, are in [docs/RECORDING.md](docs/RECORDING.md).

## 5. Train the model

All on the Mac, with `.venv-train`.

**5.1 Prepare** each recording: model inputs (128×96 grayscale at 15 fps) plus the full frames
the simulated panning eye cuts from (~2.8 GB per 45 min, in `data/cache/`):
```bash
.venv-train/bin/python -m train.prepare recordings/kitchen_20261018_102345
```

**5.2 (Optional) preview the training motion.** This renders 30 s of the simulated panning eye
over your footage: slow pans that change to a random direction every 3–9 s (or at the edge),
with smooth, varying speed and servo jitter:
```bash
.venv-train/bin/python -m train.virtual --preview        # -> data/previews/
```

**5.3 Train** (about 2 hours on a 4-core Intel i5):
```bash
.venv-train/bin/python -m train.train --name kitchen_v2
```
Each batch is 25% the still camera's view and 75% the simulated panning eye along a fresh random
path. Each round prints held-out metrics:

| Metric | Meaning |
|---|---|
| `static err`, `panning err` | Prediction error relative to copying the last frame, on high-change frames (lower is better). |
| `surprise` | Average surprise score on held-out video (lower = calmer). |
| `efference gain` | How much worse panning predictions get with the motor input blanked (>1 means the model uses its own commands). |

The best round is saved to `data/models/kitchen_v2/model.pt`. Use `--init <model>` to continue
training from an existing model (for example after recording more footage).

**5.4 Check it:**
```bash
.venv-train/bin/python -m train.compare --models kitchen_v2 --motion sweep --video   # moving eye
.venv-train/bin/python -m train.evaluate --model kitchen_v2 --video                  # still eye (+ AUROC if labeled)
```
Outputs (timelines, videos, metrics) go to `data/compare/` and `data/models/kitchen_v2/eval/`.

**5.5 Export and deploy** to the Pi:
```bash
.venv-train/bin/python -m train.export --model kitchen_v2
tools/pi_sync.sh kitchen_v2
```

## 6. Run it live

On the Pi (one camera, two eyes):
```bash
ssh -t $PI 'cd ~/Desktop/efference-simulator && .venv/bin/python -u -m eye.run --actuator null virtual --motion sweep --model models/kitchen_v2.onnx'
```
On the Mac, one viewer per eye:
```bash
.venv/bin/python -m viewer.view --port 5555      # static eye
```
```bash
.venv/bin/python -m viewer.view --port 5556      # panning eye
```
Stop with `q` in each viewer and Ctrl-C on the Pi. Only one `eye.run` can hold the camera at a time.

**Viewer:** what the model sees | model surprise | frame difference (the simple baseline), with
score traces underneath (plus the pan angle for the panning eye).

| Control | What it does |
|---|---|
| "show top % of pixels" slider (`-` `=`) | Each heatmap shows only its own strongest X% of pixels. |
| "remove specks" checkbox (`c`) + N slider (`,` `.`) | Also hides isolated clusters of ≤ N pixels. |
| "remote control" checkbox (`r`), panning window | Arrow keys drive the eye, overriding `--motion`. |
| `s` / `q` | Save a snapshot / quit. |

All viewer filters are display only; the model and its scores are unchanged.

**`eye.run` options** (on the Pi):

| Option | Default | |
|---|---|---|
| `--actuator null virtual` | `null` | Which eyes to run: still and/or panning (ports 5555, 5556, ...). |
| `--motion sweep\|random\|none` | `none` | Panning pattern: `sweep` = left→right, 1 s pause, right→left, 1 s pause; `random` = like the training data. |
| `--sweep-seconds` | `10` | Duration of each sweep pan. |
| `--model` | none | ONNX model; without it the eyes just stream video. |
| `--crop W H` | `552 414` | Panning eye's view, in camera pixels (¼ of the frame). |
| `--servo-noise` / `--servo-latency` / `--servo-speed` / `--servo-backlash` | 0.05° / 0.06 s / 300°/s / 0.5° | The simulated servo. |
| `--remote-speed` | `6` | Arrow-key speed, °/s. |
| `--log --scene NAME` | off | Record an episode (what `tools/record.py` does). |

---

## How it works

**The world model** ([train/model.py](train/model.py)) is a 585k-parameter convolutional
encoder–decoder. Each frame it gets the frames 1/15, 2/15 and 3/15 s earlier (128×96 grayscale)
plus the eye's motor history, and predicts:
- **a motion field**: where each pixel moves (the previous frame is shifted along it)
- **a correction**: what motion can't explain (something appearing, a light changing)
- **an expected error** per pixel

**Surprise** = actual error ÷ expected error. Something the model expects to be unpredictable,
like a fan or a TV, scores low; a person stepping in front of a still wall scores high. Each
frame's score is the mean of the top 1% of the surprise map.

**The efference copy** ([eye/motor.py](eye/motor.py)): the last 8 frames of commanded pan/tilt
(*as sent*, so they lead the servo by its ~60 ms delay) and actual pan/tilt (with 0.2° of
simulated sensor noise). The angles are encoded relative to the latest command, so the small
changes that move the image are the size of the input, not 1% of it. A flag separates
"motor input blanked" from "camera known to be still".

**Training data from still footage** ([train/virtual.py](train/virtual.py)): recordings are
made with the camera fixed. During training, a simulated eye pans over that footage, cutting a
¼-size crop along a new random path each round, with the simulated servo (delay, speed limit,
gear slop, jitter) between commands and crop. The model sees only the crop plus the motor history.

**On the Pi** ([eye/run.py](eye/run.py), [eye/forward.py](eye/forward.py)): the exported ONNX
model runs about 13 ms per frame per eye, two eyes in parallel. History frames are chosen by
timestamp, so a dropped camera frame skips a prediction instead of feeding the model wrongly
spaced frames.

### Results (kitchen, 45 min of training video, held-out last 15%)

| | Still-camera-only model | Current model (`kitchen_v2`) |
|---|---|---|
| Panning error vs copying the last frame | 65% | 21% |
| Efference gain | 1.00× (ignores motor input) | 1.66× |
| Mean surprise on the 10 s sweep | 18.6, spikes at every turnaround | 5.5, no turnaround spikes |
| Live panning surprise (sweeping / paused) | ~31 / ~36 | ~4 / ~5 |

Blanking the motor input barely changes the current model's *surprise* (5.7 vs 5.5), even
though its predictions get 66% less accurate: when it knows it lacks motor input, it expects
larger errors. Most of the calmer surprise comes from training on moving footage; the efference
copy mainly sharpens predictions. (`kitchen_v2` was trained in two stages: a still-camera-only
model first, then 12 rounds of the recipe above starting from it. Training from scratch as in 5.3
uses the same recipe in one go.)

## Project layout

```
eye/        runs on the Pi: camera, actuators and motion patterns, live model, stream, logger
viewer/     live viewer (Mac)
tools/      record.py (record + copy to Mac), label.py (event labeler), pi_sync.sh (deploy)
train/      prepare, virtual (simulated panning eye), model, train, compare, evaluate, export
docs/       RECORDING.md: recording and labeling in detail
recordings/ episodes (not in git)      data/  caches, models, outputs (not in git)
```

## Troubleshooting

- **`raspberrypi.local` not found / Pi unreachable:** check the Pi's power (red LED) and give it
  a minute after booting. Power-cycle it if it stays offline. `ping raspberrypi.local` should answer.
- **`Could not open /dev/video0`:** another `eye.run` already has the camera. Stop it (Ctrl-C in
  its terminal, or `ssh $PI pkill -f eye.run`).
- **Viewer says "waiting for the Pi":** the Pi loop isn't running, or the port is wrong (5555 is
  the first eye, 5556 the second).
- **Remote-control checkbox won't tick:** the Pi loop must include the panning eye
  (`--actuator ... virtual`) and be running the current code (`tools/pi_sync.sh`, then restart it).
- **Videos won't play in a browser or chat app:** they're written as H.264 when the Mac supports
  it; the Pi's `preview.mp4` recordings use MPEG-4 Part 2, which QuickTime and VLC play.
- **Frames look dark or noisy:** add light. Exposure is chosen once at startup, so restart the
  loop after the lighting changes a lot.
