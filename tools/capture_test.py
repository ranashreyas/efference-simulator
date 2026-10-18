"""Phase 0 capture test: find the UVC camera, lock its auto controls, and measure frame timing.

Usage (on the Pi, inside the venv):
    python tools/capture_test.py                 # auto-detect camera, 300 frames
    python tools/capture_test.py --frames 600 --exposure 166
"""
import argparse
import glob
import os
import subprocess
import time

import cv2
import numpy as np


def find_uvc_device():
    """Return /dev/videoN for the first UVC capture node (index 0; index 1 is metadata)."""
    for path in sorted(glob.glob("/sys/class/video4linux/video*")):
        driver = os.path.realpath(os.path.join(path, "device", "driver"))
        index_file = os.path.join(path, "index")
        index = open(index_file).read().strip() if os.path.exists(index_file) else "0"
        if driver.endswith("uvcvideo") and index == "0":
            return "/dev/" + os.path.basename(path)
    return None


def set_ctrl(dev, name, value):
    r = subprocess.run(["v4l2-ctl", "-d", dev, f"--set-ctrl={name}={value}"],
                       capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else f"skipped ({r.stderr.strip() or 'unsupported'})"
    print(f"  {name}={value}: {status}")


def lock_controls(dev, exposure, gain, powerline_hz):
    # Auto exposure / white balance / focus make the world look like it changed when it didn't.
    print("Locking camera controls:")
    set_ctrl(dev, "auto_exposure", 1)                   # 1 = manual
    set_ctrl(dev, "exposure_dynamic_framerate", 0)      # don't let exposure stretch frame time
    set_ctrl(dev, "exposure_time_absolute", exposure)   # units of 100 us
    set_ctrl(dev, "white_balance_automatic", 0)
    set_ctrl(dev, "focus_automatic_continuous", 0)
    set_ctrl(dev, "gain", gain)
    set_ctrl(dev, "power_line_frequency", 2 if powerline_hz == 60 else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--exposure", type=int, default=333,
                    help="100us units; 333 = 33.3 ms, a multiple of the 120 Hz flicker period")
    ap.add_argument("--gain", type=int, default=64)
    ap.add_argument("--powerline", type=int, default=60, choices=[50, 60])
    ap.add_argument("--out", default="runs/capture_test")
    args = ap.parse_args()

    dev = args.device or find_uvc_device()
    if dev is None:
        raise SystemExit("No UVC camera found. Check `lsusb` and the cable.")
    print(f"Camera: {dev}")
    lock_controls(dev, args.exposure, args.gain, args.powerline)

    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {dev}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Negotiated {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps")

    for _ in range(15):  # let the stream settle
        cap.read()

    t_driver, t_host, samples = [], [], []
    for i in range(args.frames):
        ok, frame = cap.read()
        t_host.append(time.monotonic())
        if not ok:
            print(f"read failed at frame {i}")
            break
        t_driver.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)  # V4L2 buffer timestamp
        if i % 100 == 0:
            samples.append(frame)
    cap.release()

    os.makedirs(args.out, exist_ok=True)
    for j, f in enumerate(samples):
        cv2.imwrite(f"{args.out}/frame_{j}.jpg", f)
        small = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (128, 96), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{args.out}/model_input_{j}.png", small)
        print(f"sample {j}: mean brightness {small.mean():.0f}/255, "
              f"{(small >= 250).mean() * 100:.1f}% clipped")
    np.savez(f"{args.out}/timestamps.npz", t_driver=np.array(t_driver), t_host=np.array(t_host))

    def report(name, t):
        dt = np.diff(np.array(t)) * 1000
        if len(dt) == 0 or not np.all(np.isfinite(dt)) or np.all(dt == 0):
            print(f"{name}: unavailable")
            return
        expected = 1000 / args.fps
        if abs(np.median(dt) - expected) > 0.2 * expected:
            print(f"{name}: camera delivers {1000 / np.median(dt):.1f} fps, not the requested {args.fps}")
            expected = np.median(dt)
        dropped = int(np.sum((np.round(dt / expected) - 1).clip(min=0)))
        print(f"{name}: mean {dt.mean():.2f} ms ({1000 / dt.mean():.1f} fps), "
              f"std {dt.std():.2f} ms, min {dt.min():.2f}, max {dt.max():.2f}, ~{dropped} dropped")

    print(f"\nCaptured {len(t_host)} frames")
    report("Driver timestamps", t_driver)
    report("Host read times  ", t_host)
    if len(t_driver) == len(t_host) and t_driver and t_driver[0] > 0:
        lag = (np.array(t_host) - np.array(t_driver)) * 1000
        print(f"Driver->host delay: median {np.median(lag):.1f} ms "
              f"(only meaningful if both are CLOCK_MONOTONIC; true glass-to-host latency needs the LED test)")
    print(f"Saved samples to {args.out}/")


if __name__ == "__main__":
    main()
