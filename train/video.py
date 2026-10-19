import cv2


def open_writer(path, fps, size):
    """MP4 writer that browsers and chat apps can play: H.264 ('avc1', via macOS AVFoundation)
    when available, else OpenCV's default MPEG-4 Part 2 ('mp4v'), which some players reject."""
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"could not open a video writer for {path}")
