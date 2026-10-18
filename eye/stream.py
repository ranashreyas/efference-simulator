"""Debug stream: the Pi publishes a JSON header plus a few encoded images per frame over
ZeroMQ. The viewer on the Mac subscribes. Small send/receive queues mean a slow viewer
drops old frames instead of falling behind.

Wire format (multipart): [b"dbg", header_json, image_0, image_1, ...]
header["images"] lists the image names in order.
"""
import json

import cv2
import numpy as np
import zmq

TOPIC = b"dbg"


class DebugPublisher:
    def __init__(self, port=5555):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.SNDHWM, 2)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://*:{port}")

    def send(self, header, images, jpeg_quality=75):
        names, parts = [], []
        for name, img in images.items():
            if img.ndim == 2 and img.shape[0] <= 128:      # small model-sized maps: lossless
                ok, buf = cv2.imencode(".png", img)
            else:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                names.append(name)
                parts.append(buf.tobytes())
        header = dict(header, images=names)
        try:
            self.sock.send_multipart([TOPIC, json.dumps(header).encode()] + parts, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass                                            # no room: drop this frame

    def close(self):
        self.sock.close()


class DebugSubscriber:
    def __init__(self, host="raspberrypi.local", port=5555):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 2)
        self.sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self.sock.connect(f"tcp://{host}:{port}")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv_latest(self, timeout_ms=1000):
        """Wait for a message, then drain the queue and return only the newest one."""
        if not self.poller.poll(timeout_ms):
            return None
        msg = self.sock.recv_multipart()
        while True:
            try:
                msg = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
        header = json.loads(msg[1])
        images = {}
        for name, buf in zip(header["images"], msg[2:]):
            images[name] = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
        return header, images

