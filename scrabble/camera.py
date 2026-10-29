"""Photos from a camera: a USB camera (its number, e.g. 4 for /dev/video4) or a phone that streams
video (the stream's URL, e.g. http://192.168.1.20:8080/video from an IP camera app).

Usage: python -m scrabble.camera [SOURCE]

Shows the camera's live view, to aim it at the board before a game.
Keys:
  g           show / hide the grid found in the live view (slower)
  s / Space   save the current frame to dataset/photos (to annotate it later)
  q / Esc     quit
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

WINDOW = "Camera"
PHOTOS = Path("dataset/photos")


def parse_source(source):
    """A camera number ("4") or a stream URL."""
    return int(source) if str(source).isdigit() else str(source)


class Camera:
    """Reads frames all the time in a thread and keeps the newest one, so a photo taken at any
    moment shows the board as it is then, not a frame that waited in the camera's buffer."""

    def __init__(self, source):
        source = parse_source(source)
        if isinstance(source, int):
            self.capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            # MJPG gives USB cameras their full resolution at a usable frame rate; asking for
            # a huge size makes the driver pick the largest one it has.
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        else:
            self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise ValueError(f"can't open the camera {source}")
        self.source = source
        self.latest, self.time = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.latest, self.time = frame, time.monotonic()

    def frame(self, timeout=5.0, max_age=1.0):
        """The newest frame; waits up to `timeout` s for one at most `max_age` s old."""
        end = time.monotonic() + timeout
        while True:
            with self.lock:
                if self.latest is not None and time.monotonic() - self.time <= max_age:
                    return self.latest.copy()
            if time.monotonic() > end:
                raise ValueError(f"no picture from the camera {self.source}")
            time.sleep(0.02)

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.capture.release()


def next_path(folder):
    """The next free numbered photo name in `folder`: 001.jpg, 002.jpg, ..."""
    n = 1
    while (folder / f"{n:03d}.jpg").exists():
        n += 1
    return folder / f"{n:03d}.jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", nargs="?", default="0", help="camera number or stream URL (default 0)")
    parser.add_argument("--corner-model", type=Path, default=Path("models/corners.pt"))
    args = parser.parse_args()

    camera = Camera(args.source)
    corner_model = None
    show_grid = False
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    try:
        while True:
            image = camera.frame()
            view = image.copy()
            if show_grid:
                from scrabble.corners import draw, load_model, predict_corners
                corner_model = corner_model or load_model(args.corner_model)
                view = draw(image, *predict_corners(corner_model, image))
            h, w = image.shape[:2]
            text = f"{w}x{h}  g: grid {'on' if show_grid else 'off'}  s: save  q: quit"
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("g"):
                show_grid = not show_grid
            elif key in (ord("s"), ord(" ")):
                PHOTOS.mkdir(parents=True, exist_ok=True)
                path = PHOTOS / f"camera_{datetime.now():%Y-%m-%d_%H-%M-%S}.jpg"
                cv2.imwrite(str(path), image)
                print(f"Saved {path}", flush=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
