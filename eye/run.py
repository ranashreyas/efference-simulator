"""Main loop on the Pi: camera -> one or more eye pipelines -> forward model -> stream (+ log).

    python -m eye.run                                         # static eye, stream only
    python -m eye.run --model models/kitchen_v2.onnx          # ... with live surprise
    python -m eye.run --actuator null virtual --motion sweep --model models/kitchen_v2.onnx
                     # two pipelines from one camera: static on port 5555, panning on 5556

Each pipeline has its own actuator and its own copy of the model, and publishes on its own port
(--port, --port + 1, ...). What the model sees:
    null     the centered 4:3 crop at full height, through exactly the training preprocessing
    virtual  only the crop window the simulated servo points at, plus the servo's command and
             position history as the model's motor input

Recording household video: run tools/record.py on the Mac, which starts this with --log and
copies the episode to the Mac afterwards.
"""
import argparse
import collections
import dataclasses
import errno
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .actuators import (NullActuator, RandomPan, RemoteControl, ServoModel, SweepScan,
                        VirtualActuator)
from .camera import Camera
from .api import load_or_create_token, start_server
from .events import EventDetector, EventSink, EventStore
from .logger import EpisodeLogger
from .preprocess import LOG_SIZE, MODEL_SIZE, static_model_input, to_gray_small
from .stream import ControlReceiver, DebugPublisher

STREAM_EYE_SIZE = (320, 240)
STREAM_THUMB_SIZE = (320, 180)
# Fixed color scales for the heatmaps (~top 0.01% of values on held-out kitchen video);
# maps are sent as sqrt(value / vmax) so faint signals stay visible.
SURPRISE_VMAX = 50.0
DIFF_VMAX = 0.25


class Pipeline:
    """One eye: actuator (+ optional motion pattern) -> what the eye sees -> model -> stream."""

    def __init__(self, kind, args, port, model_path):
        self.kind = kind
        if kind == "null":
            self.act = NullActuator(crop_size=(960, 720))       # the 4:3 training view
        else:
            servo = ServoModel(latency=args.servo_latency, max_speed=args.servo_speed,
                               backlash=args.servo_backlash, noise=args.servo_noise)
            self.act = VirtualActuator(crop_size=tuple(args.crop), servo=servo)
        self.motion = None
        if kind == "virtual" and args.motion == "sweep":
            self.motion = SweepScan(self.act.pan_limit, sweep_s=args.sweep_seconds)
        elif kind == "virtual" and args.motion == "random":
            self.motion = RandomPan(self.act.pan_limit, self.act.tilt_limit)
        # remote control from the viewer (arrow keys); while on, it overrides --motion
        self.remote = RemoteControl(self.act.pan_limit, self.act.tilt_limit, speed=args.remote_speed)
        self.remote_on = False
        self.model = None
        if model_path:
            from .forward import LiveForwardModel
            from .actuators import C922_FOCAL_PX_720P
            from .motor import ACTUAL_NOISE_DEG
            focal = (C922_FOCAL_PX_720P * MODEL_SIZE[0] / args.crop[0],
                     C922_FOCAL_PX_720P * MODEL_SIZE[1] / args.crop[1]) if kind == "virtual" else None
            self.model = LiveForwardModel(model_path, threads=args.model_threads, focal_px=focal,
                                          actual_noise_deg=ACTUAL_NOISE_DEG if kind == "virtual" else 0.0)
        self.pub = DebugPublisher(port)
        self.port = port

    def model_input(self, frame, eye):
        if self.kind == "null":
            return static_model_input(to_gray_small(frame.img, LOG_SIZE))   # identical to training
        return to_gray_small(eye.img, MODEL_SIZE)                           # the crop, nothing else

    def control(self, msg, t_now):
        """A message from the viewer: {"type": "remote", "enabled": bool} or
        {"type": "key", "dx": -1|0|1, "dy": -1|0|1}. Only the virtual eye can be driven."""
        if self.kind != "virtual":
            return
        if msg.get("type") == "remote" and bool(msg.get("enabled")) != self.remote_on:
            self.remote_on = bool(msg.get("enabled"))
            if self.remote_on:                          # take over from wherever the eye points now
                s = self.act.state(t_now)
                self.remote.reset(s.pan_target, s.tilt_target)
            print(f"[run] remote control {'on' if self.remote_on else 'off'}")
        elif msg.get("type") == "key" and self.remote_on:
            self.remote.key(msg.get("dx", 0), msg.get("dy", 0), t_now)

    def step(self, frame, t_now):
        cmd = None
        mover = self.remote if self.remote_on else self.motion
        if mover:
            target = mover.step(t_now)
            if target:
                cmd = self.act.command(*target, t=t_now)
        state = self.act.state(frame.t_sensor)          # where the eye was when this frame was taken
        eye = self.act.view(frame, state)
        out = self.model.step(frame.t_sensor, self.model_input(frame, eye), state) if self.model else None
        return state, eye, out, cmd

    def publish(self, frame, state, eye, out, header):
        header = dict(header, mode="static" if self.kind == "null" else "panning",
                      state=dataclasses.asdict(state), crop_rect=self.act.crop_rect(state),
                      pan_limit=self.act.pan_limit, tilt_limit=self.act.tilt_limit,
                      remote=self.remote_on, remote_available=self.kind == "virtual")
        images = {"eye": cv2.resize(eye.img, STREAM_EYE_SIZE, interpolation=cv2.INTER_AREA)}
        if self.kind == "virtual":
            images["full"] = cv2.resize(frame.img, STREAM_THUMB_SIZE, interpolation=cv2.INTER_AREA)
        if out is not None:
            images["surprise"] = (np.sqrt(np.clip(out["surprise"] / SURPRISE_VMAX, 0, 1)) * 255).astype(np.uint8)
            images["diff"] = (np.sqrt(np.clip(out["diff"] / DIFF_VMAX, 0, 1)) * 255).astype(np.uint8)
            header.update(surprise_score=out["surprise_score"], diff_score=out["diff_score"],
                          infer_ms=round(out["infer_ms"], 1), surprise_vmax=SURPRISE_VMAX,
                          diff_vmax=DIFF_VMAX)
        if self.model:
            header["skipped"] = self.model.skipped
        self.pub.send(header, images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuator", nargs="+", choices=["null", "virtual"], default=["null"],
                    help="one pipeline per actuator, published on --port, --port + 1, ...")
    ap.add_argument("--motion", choices=["none", "sweep", "random"], default="none",
                    help="virtual eye movement: sweep = left->right, 1 s pause, right->left, "
                         "1 s pause, repeat; random = slow wandering pans, like the training data")
    ap.add_argument("--sweep-seconds", type=float, default=10.0, help="duration of each pan of the sweep")
    ap.add_argument("--model", help="ONNX forward model (e.g. models/kitchen_v2.onnx)")
    ap.add_argument("--model-threads", type=int, default=2, help="per pipeline; pipelines run in parallel")
    ap.add_argument("--crop", type=int, nargs=2, default=[552, 414], metavar=("W", "H"),
                    help="virtual eye crop size in camera pixels (default: 1/4 of the frame, 4:3)")
    ap.add_argument("--log", action="store_true", help="record an episode to recordings/")
    ap.add_argument("--scene", default="home", help="free-form name stored with the episode")
    ap.add_argument("--port", type=int, default=5555, help="debug stream port of the first pipeline")
    ap.add_argument("--control-port", type=int, default=5560, help="viewer -> Pi remote control")
    ap.add_argument("--remote-speed", type=float, default=6.0, help="arrow-key pan/tilt speed, deg/s")
    ap.add_argument("--events", action="store_true",
                    help="save surprise events (peak frame + context) to --events-dir and serve them for "
                         "labeling on --api-port (localhost only; the viewer's event window uses it)")
    ap.add_argument("--api-port", type=int, default=8787, help="labeling API port (localhost only)")
    ap.add_argument("--events-dir", default=str(Path(__file__).resolve().parent.parent / "events"))
    ap.add_argument("--event-eye", choices=["null", "virtual"], default="virtual",
                    help="which eye's surprise triggers events, and whose view is saved: virtual = the "
                         "panning eye's crop window (default), null = the still eye's centered view")
    ap.add_argument("--event-threshold", type=float, default=60.0,
                    help="surprise score that starts an event (default: ~99.9th percentile of the "
                         "still eye on held-out kitchen video; recalibrate for other scenes)")
    ap.add_argument("--event-min-frames", type=int, default=3, help="frames in a row above threshold to count")
    ap.add_argument("--event-cooldown", type=float, default=30.0, help="seconds between stored events")
    ap.add_argument("--event-control-mean", type=float, default=600.0,
                    help="mean seconds between random unflagged control frames saved for labeling (0 = none)")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--target-brightness", type=float, default=100)
    ap.add_argument("--servo-latency", type=float, default=0.06)
    ap.add_argument("--servo-speed", type=float, default=300.0)
    ap.add_argument("--servo-backlash", type=float, default=0.5)
    ap.add_argument("--servo-noise", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = until Ctrl-C)")
    args = ap.parse_args()

    cam = Camera().start(calibrate=not args.no_calibrate, target_brightness=args.target_brightness)
    pipes = [Pipeline(kind, args, args.port + k, args.model) for k, kind in enumerate(args.actuator)]
    if args.motion != "none" and "virtual" not in args.actuator:
        print("[run] --motion only moves a virtual actuator")
    logger = None
    if args.log:
        act = pipes[0].act
        meta = dict(scene=args.scene, actuator=args.actuator[0], device=cam.device,
                    controls=cam.controls, camera_size=[cam.width, cam.height], fps=cam.fps,
                    crop_size=[act.crop_w, act.crop_h], focal_px=act.f,
                    servo=dict(latency=act.servo.latency, max_speed=act.servo.max_speed,
                               backlash=act.servo.backlash, noise=act.servo.noise))
        name = f"{args.scene}_{time.strftime('%Y%m%d_%H%M%S')}"
        logger = EpisodeLogger(name=name, meta=meta, fps=cam.fps)
        print(f"[run] recording to {logger.dir} (Ctrl-C to stop)")
    for p in pipes:
        print(f"[run] {p.kind}: eye range ±{p.act.pan_limit:.1f}° pan, ±{p.act.tilt_limit:.1f}° tilt, "
              f"model {'on' if p.model else 'off'}, streaming on tcp://*:{p.port}")

    detector = sink = event_pipe = api = None
    if args.events:
        try:
            event_pipe = next((i for i, p in enumerate(pipes) if p.kind == args.event_eye), None)
            if event_pipe is None or not pipes[event_pipe].model:
                raise SystemExit(f"--events needs a --model and a '{args.event_eye}' actuator to score frames")
            store = EventStore(args.events_dir)
            sink = EventSink(store, pipes[event_pipe].kind)
            detector = EventDetector(sink, threshold=args.event_threshold, min_frames=args.event_min_frames,
                                     cooldown_s=args.event_cooldown, control_mean_s=args.event_control_mean)
            api = start_server(store, load_or_create_token(), port=args.api_port)
        except BaseException as e:                      # leave nothing running (camera, streams, recording)
            if logger:
                logger.close()
            if sink:
                sink.close()
            for p in pipes:
                p.pub.close()
            cam.stop()
            if isinstance(e, OSError) and e.errno == errno.EADDRINUSE:
                raise SystemExit(f"[run] port {args.api_port} is already in use on this machine: an old eye.run or "
                                 f"eye.api still running? Find it with `ss -ltnp | grep {args.api_port}`, "
                                 f"or pick another port with --api-port") from None
            raise
        print(f"[run] events on: {args.event_eye} eye, threshold {args.event_threshold:g}, "
              f"stored in {store.root} {store.counts()}; labeling API on 127.0.0.1:{args.api_port}")

    pool = ThreadPoolExecutor(len(pipes))              # ONNX Runtime releases the GIL
    control = ControlReceiver(args.control_port)
    times = collections.deque(maxlen=60)
    last_seq, dropped, use_host_clock = None, 0, False
    t_start = time.monotonic()
    try:
        while not args.duration or time.monotonic() - t_start < args.duration:
            frame = cam.wait_next(last_seq)
            if frame is None:
                continue
            t0 = time.monotonic()
            if last_seq is not None and frame.seq > last_seq + 1:
                dropped += frame.seq - last_seq - 1         # the loop fell behind the camera
            last_seq = frame.seq
            if not use_host_clock and abs(frame.t_host - frame.t_sensor) > 1.0:
                print("[run] driver timestamps aren't on the monotonic clock; using host time")
                use_host_clock = True
            if use_host_clock:
                frame = dataclasses.replace(frame, t_sensor=frame.t_host)

            for msg in control.poll():
                for p in pipes:
                    p.control(msg, t0)
            steps = list(pool.map(lambda p: p.step(frame, t0), pipes))
            results = [(state, eye, out) for state, eye, out, _ in steps]
            if logger:
                state, _, _, cmd = steps[0]
                if cmd:
                    logger.add_command(cmd)
                logger.add(frame, state)

            if detector:
                out = results[event_pipe][2]
                # save what the triggering eye saw (its crop window), not the whole camera frame
                detector.update(frame.t_sensor, out["surprise_score"] if out else None,
                                results[event_pipe][1].img, frame.seq)

            times.append(frame.t_sensor)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            header = dict(seq=frame.seq, t_sensor=frame.t_sensor, fps=round(fps, 1),
                          dropped=dropped, proc_ms=round((time.monotonic() - t0) * 1000, 1),
                          capture_to_proc_ms=round((t0 - frame.t_sensor) * 1000, 1),
                          full_size=[cam.width, cam.height],
                          exposure=cam.controls.get("exposure_time_absolute"),
                          gain=cam.controls.get("gain"),
                          logging=logger.dir if logger else None, scene=args.scene)
            for p, (state, eye, out) in zip(pipes, results):
                p.publish(frame, state, eye, out, header)
    except KeyboardInterrupt:
        pass
    finally:
        if logger:
            logger.close()
        if api:
            api.shutdown()
        if sink:
            sink.close()
            print(f"[run] events stored: {sink.stored}, suppressed by cooldown: {detector.suppressed}")
        pool.shutdown()
        control.close()
        for p in pipes:
            p.pub.close()
        cam.stop()
        print(f"[run] stopped; {dropped} frames dropped by the loop")


if __name__ == "__main__":
    main()
