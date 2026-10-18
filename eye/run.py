"""Main loop on the Pi: camera -> actuator -> eye view -> (model goes here) -> stream + log.

    python -m eye.run                              # static eye, stream to the viewer
    python -m eye.run --actuator virtual --babble  # virtual eye making random saccades
    python -m eye.run --actuator virtual --babble --log   # ...and record an episode

Recording household video: run tools/record.py on the Mac, which starts this with --log and
copies the episode to the Mac afterwards.
"""
import argparse
import collections
import dataclasses
import time

import cv2

from .actuators import NullActuator, SaccadeBabbler, ServoModel, VirtualActuator
from .camera import Camera
from .logger import EpisodeLogger
from .preprocess import MODEL_SIZE, to_gray_small
from .stream import DebugPublisher

STREAM_FULL_SIZE = (640, 360)


def make_actuator(args):
    if args.actuator == "null":
        return NullActuator()
    if args.actuator == "virtual":
        servo = ServoModel(latency=args.servo_latency, max_speed=args.servo_speed,
                           backlash=args.servo_backlash, noise=args.servo_noise)
        return VirtualActuator(servo=servo)
    raise SystemExit(f"unknown actuator {args.actuator}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actuator", choices=["null", "virtual"], default="null")
    ap.add_argument("--babble", action="store_true", help="random saccades (virtual actuator)")
    ap.add_argument("--log", action="store_true", help="record an episode to recordings/")
    ap.add_argument("--scene", default="home", help="free-form name stored with the episode")
    ap.add_argument("--port", type=int, default=5555, help="debug stream port")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--target-brightness", type=float, default=100)
    ap.add_argument("--servo-latency", type=float, default=0.06)
    ap.add_argument("--servo-speed", type=float, default=300.0)
    ap.add_argument("--servo-backlash", type=float, default=0.5)
    ap.add_argument("--servo-noise", type=float, default=0.05)
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = until Ctrl-C)")
    args = ap.parse_args()

    cam = Camera().start(calibrate=not args.no_calibrate, target_brightness=args.target_brightness)
    act = make_actuator(args)
    babbler = SaccadeBabbler(act.pan_limit, act.tilt_limit) if args.babble else None
    if babbler and args.actuator == "null":
        print("[run] --babble does nothing with the null actuator")
    pub = DebugPublisher(args.port)
    logger = None
    if args.log:
        meta = dict(scene=args.scene, actuator=args.actuator, device=cam.device,
                    controls=cam.controls, camera_size=[cam.width, cam.height], fps=cam.fps,
                    crop_size=[act.crop_w, act.crop_h], focal_px=act.f,
                    servo=dict(latency=act.servo.latency, max_speed=act.servo.max_speed,
                               backlash=act.servo.backlash, noise=act.servo.noise))
        name = f"{args.scene}_{time.strftime('%Y%m%d_%H%M%S')}"
        logger = EpisodeLogger(name=name, meta=meta, fps=cam.fps)
        print(f"[run] recording to {logger.dir} (Ctrl-C to stop)")
    print(f"[run] actuator={args.actuator} eye range ±{act.pan_limit:.1f}° pan, "
          f"±{act.tilt_limit:.1f}° tilt; streaming on tcp://*:{args.port}")

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

            if babbler:
                target = babbler.step(t0)
                if target:
                    cmd = act.command(*target, t=t0)
                    if logger:
                        logger.add_command(cmd)

            state = act.state(frame.t_sensor)              # where the eye was when this frame was taken
            eye = act.view(frame, state)
            model_in = to_gray_small(eye.img, MODEL_SIZE)
            # TODO phase 1b: forward model -> prediction, uncertainty, surprise map

            if logger:
                logger.add(frame, state)

            times.append(frame.t_sensor)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0
            header = dict(seq=frame.seq, t_sensor=frame.t_sensor, fps=round(fps, 1),
                          dropped=dropped, proc_ms=round((time.monotonic() - t0) * 1000, 1),
                          capture_to_proc_ms=round((t0 - frame.t_sensor) * 1000, 1),
                          actuator=args.actuator, state=dataclasses.asdict(state),
                          crop_rect=act.crop_rect(state), full_size=[cam.width, cam.height],
                          exposure=cam.controls.get("exposure_time_absolute"),
                          gain=cam.controls.get("gain"),
                          logging=logger.dir if logger else None,
                          scene=args.scene)
            pub.send(header, {
                "full": cv2.resize(frame.img, STREAM_FULL_SIZE, interpolation=cv2.INTER_AREA),
                "eye": eye.img,
                "model": model_in,
            })
    except KeyboardInterrupt:
        pass
    finally:
        if logger:
            logger.close()
        pub.close()
        cam.stop()
        print(f"[run] stopped; {dropped} frames dropped by the loop")


if __name__ == "__main__":
    main()
