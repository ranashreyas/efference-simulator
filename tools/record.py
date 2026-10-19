"""Record household video with the Pi and bring it to the Mac. Run on the Mac.

    .venv/bin/python tools/record.py                          # record until Ctrl-C
    .venv/bin/python tools/record.py --scene kitchen --view   # name it, open the live viewer
    .venv/bin/python tools/record.py --pull-only              # just fetch episodes

Record ordinary household life in sessions of any length (10-30 min keeps files manageable).
Recording happens on the Pi's SD card (so a WiFi hiccup can't drop frames). Press Ctrl-C to
stop; the Pi finishes writing the episode, then every episode not yet on the Mac is copied
into recordings/ here. Labeling events with tools/label.py is optional.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from eye.episode import frame_index, load_meta  # noqa: E402


def local_episodes(out):
    return {p.name for p in out.iterdir() if p.is_dir()} if out.exists() else set()


def pull(host, pi_dir, out):
    """Copy every episode from the Pi that isn't complete on the Mac yet. Safe to re-run."""
    out.mkdir(parents=True, exist_ok=True)
    before = local_episodes(out)
    remote_dir = pi_dir[2:] if pi_dir.startswith("~/") else pi_dir   # rsync paths are home-relative
    print(f"Copying episodes from {host}:{remote_dir}/recordings/ to {out}/ ...")
    r = subprocess.run(["rsync", "-a", "--partial", f"{host}:{remote_dir}/recordings/", f"{out}/"])
    if r.returncode != 0:
        sys.exit(f"rsync failed ({r.returncode}). Is the Pi reachable? Try: ssh {host}")
    new = sorted(local_episodes(out) - before)
    for name in new:
        check(out / name)
    if not new:
        print("No new episodes (everything is already on the Mac).")
    return new


def check(ep):
    meta = load_meta(ep)
    seqs, _ = frame_index(ep)
    preview = ep / "preview.mp4"
    problems = []
    if not meta:
        problems.append("no meta.json (recording may not have stopped cleanly)")
    if meta and meta.get("frames") != len(seqs):
        problems.append(f"meta says {meta.get('frames')} frames but chunks hold {len(seqs)}")
    if not preview.exists():
        problems.append("no preview.mp4")
    elif meta.get("preview_frames") not in (None, len(seqs)):
        problems.append(f"preview has {meta.get('preview_frames')} frames, chunks {len(seqs)}")
    size_mb = sum(f.stat().st_size for f in ep.iterdir()) / 1e6
    status = "OK" if not problems else "CHECK: " + "; ".join(problems)
    print(f"  {ep.name}: {len(seqs)} frames ({len(seqs) / 30:.0f} s), {size_mb:.0f} MB, "
          f"scene {meta.get('scene', '?')}  {status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="home", help="optional name, e.g. kitchen, living_room")
    ap.add_argument("--duration", type=float, default=0, help="seconds (0 = until Ctrl-C)")
    ap.add_argument("--view", action="store_true", help="also open the live viewer")
    ap.add_argument("--pull-only", action="store_true", help="don't record; just copy episodes")
    ap.add_argument("--host", default="ranashreyas@raspberrypi.local")
    ap.add_argument("--pi-dir", default="~/Desktop/efference-simulator")
    ap.add_argument("--out", default=str(PROJECT / "recordings"))
    args = ap.parse_args()
    out = Path(args.out)

    if not args.pull_only:
        remote = (f"cd {args.pi_dir} && . .venv/bin/activate && python -u -m eye.run --log "
                  f"--scene {shlex.quote(args.scene)}")
        if args.duration:
            remote += f" --duration {args.duration}"
        viewer = None
        if args.view:
            host_only = args.host.split("@")[-1]
            viewer = subprocess.Popen([sys.executable, "-m", "viewer.view", "--host", host_only],
                                      cwd=PROJECT, stdout=subprocess.DEVNULL)
        print(f"Recording {args.scene} on the Pi. Press Ctrl-C to stop.", flush=True)
        try:
            # -t gives the remote process a terminal, so Ctrl-C reaches it and it can finish
            # writing the episode cleanly before we copy anything.
            subprocess.run(["ssh", "-t", args.host, remote])
        except KeyboardInterrupt:
            pass
        finally:
            if viewer:
                viewer.terminate()

    new = pull(args.host, args.pi_dir, out)
    if new:
        print("\nOptionally label events with:")
        for name in new:
            print(f"  .venv/bin/python tools/label.py recordings/{name}")


if __name__ == "__main__":
    main()
