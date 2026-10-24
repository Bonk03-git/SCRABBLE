"""Manual board corner annotation.

Usage: python -m scrabble.annotate [IMAGES_DIR] [--dataset DIR]

Opens the photos in IMAGES_DIR (default: dataset/photos) one by one.
Click the 4 corners of the 15x15 cell grid IN ORDER: 1 = the board's top-left
corner, then clockwise (2 top-right, 3 bottom-right, 4 bottom-left). Points
are saved exactly in click order, so they encode the board's orientation even
when the photo is rotated. Saving is refused unless the clicks form a convex
quadrilateral going clockwise. After saving, the photo is
moved to DATASET/annotated and its corners are written to
DATASET/labels/<photo stem>.json:
  {"image": "photo.jpg", "corners": [[x, y], [x, y], [x, y], [x, y]]}
Coordinates are pixels of the original photo. Photos already in
DATASET/annotated can be reopened (IMAGES_DIR=dataset/annotated) to fix labels.

Keys:
  Enter / Space   save and go to the next photo (needs 4 corners)
  u / Backspace   undo the last corner (right click works too)
  d / a           next / previous photo without saving
  q / Esc         quit
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_W, MAX_H = 1280, 800
LOUPE_RADIUS, LOUPE_ZOOM = 30, 5
WINDOW = "annotate"
# Resolve against this file, not the current working directory.
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"


CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def is_clockwise_convex(points) -> bool:
    """True if the 4 points, in click order, go clockwise on screen around a convex shape."""
    pts = np.asarray(points, dtype=np.float64).reshape(4, 2)
    edges = np.roll(pts, -1, axis=0) - pts
    nxt = np.roll(edges, -1, axis=0)
    # With the y axis pointing down, a clockwise turn has a positive cross product.
    return bool(np.all(edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0] > 0))


def list_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def fit_scale(width: int, height: int) -> float:
    return min(1.0, MAX_W / width, MAX_H / height)


def label_path(dataset: Path, image_name: str) -> Path:
    return dataset / "labels" / f"{Path(image_name).stem}.json"


def make_label(image_name: str, points) -> dict:
    """Corners are kept in click order: 1 top-left of the board, then clockwise."""
    return {"image": image_name, "corners": [[round(float(x), 1), round(float(y), 1)] for x, y in points]}


def load_corners(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    return [tuple(c) for c in json.loads(path.read_text())["corners"]]


def save_annotation(image: Path, points, dataset: Path) -> Path:
    """Write the label and move the photo to dataset/annotated; return its new path."""
    done_dir = dataset / "annotated"
    target = done_dir / image.name
    if target.exists() and target.resolve() != image.resolve():
        raise FileExistsError(f"{target} already exists")

    labels = label_path(dataset, image.name)
    labels.parent.mkdir(parents=True, exist_ok=True)
    tmp = labels.with_suffix(".tmp")
    tmp.write_text(json.dumps(make_label(image.name, points), indent=2))
    tmp.replace(labels)

    if target.resolve() != image.resolve():
        done_dir.mkdir(parents=True, exist_ok=True)
        image.replace(target)
    return target


class Annotator:
    def __init__(self, images: list[Path], dataset: Path):
        self.images = images
        self.dataset = dataset
        self.cursor = (0, 0)
        self.message = ""
        self.saved = 0
        self.open(0)

    def open(self, index: int) -> None:
        self.index = index
        path = self.images[index]
        self.image = cv2.imread(str(path))
        if self.image is None:
            raise SystemExit(f"Cannot read image: {path}")
        h, w = self.image.shape[:2]
        self.scale = fit_scale(w, h)
        self.points = load_corners(label_path(self.dataset, path.name))

    def on_mouse(self, event, x, y, flags, param) -> None:
        self.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x / self.scale, y / self.scale))
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            self.points.pop()

    def save(self) -> bool:
        """Save the current photo; return False when no photos are left."""
        image = self.images[self.index]
        try:
            target = save_annotation(image, self.points, self.dataset)
        except FileExistsError as e:
            self.message = f"not saved: {e}"
            return True
        self.saved += 1
        if target == image:  # re-labelled a photo that was already in annotated/
            self.go(1)
            return True
        self.images.pop(self.index)
        if not self.images:
            return False
        self.open(min(self.index, len(self.images) - 1))
        self.message = f"saved {image.name}"
        return True

    def go(self, step: int) -> None:
        target = self.index + step
        if 0 <= target < len(self.images):
            self.open(target)
            self.message = ""
        else:
            self.message = "last photo" if step > 0 else "first photo"

    def loupe(self) -> np.ndarray:
        r = LOUPE_RADIUS
        cx, cy = (int(c / self.scale) for c in self.cursor)
        padded = cv2.copyMakeBorder(self.image, r, r, r, r, cv2.BORDER_CONSTANT)
        crop = padded[cy:cy + 2 * r + 1, cx:cx + 2 * r + 1]
        zoomed = cv2.resize(crop, None, fx=LOUPE_ZOOM, fy=LOUPE_ZOOM, interpolation=cv2.INTER_NEAREST)
        mid = zoomed.shape[0] // 2
        cv2.line(zoomed, (mid, 0), (mid, zoomed.shape[0]), (0, 0, 255), 1)
        cv2.line(zoomed, (0, mid), (zoomed.shape[1], mid), (0, 0, 255), 1)
        cv2.rectangle(zoomed, (0, 0), (zoomed.shape[1] - 1, zoomed.shape[0] - 1), (255, 255, 255), 2)
        return zoomed

    def render(self) -> np.ndarray:
        view = cv2.resize(self.image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        pts = [(round(x * self.scale), round(y * self.scale)) for x, y in self.points]
        if len(pts) == 4:
            color = (0, 255, 0) if is_clockwise_convex(self.points) else (0, 0, 255)
            cv2.polylines(view, [np.array(pts, np.int32)], True, color, 2)
        elif len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], False, (0, 255, 255), 1)
        for number, p in enumerate(pts, start=1):
            cv2.circle(view, p, 5, (0, 0, 255), -1)
            cv2.putText(view, str(number), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(view, str(number), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        loupe = self.loupe()
        lh, lw = loupe.shape[:2]
        if view.shape[0] > lh + 40 and view.shape[1] > lw + 10:
            # Keep the loupe on the side away from the cursor so it never hides the corner being clicked.
            cx, cy = self.cursor
            x0 = 10 if cx > view.shape[1] // 2 else view.shape[1] - lw - 10
            y0 = view.shape[0] - lh - 10 if cy < view.shape[0] // 2 else 40
            view[y0:y0 + lh, x0:x0 + lw] = loupe

        name = self.images[self.index].name
        step = f"click {len(self.points) + 1}: {CORNER_NAMES[len(self.points)]}" if len(self.points) < 4 else "Enter - save"
        status = f"{self.index + 1}/{len(self.images)} {name} | {step} | saved {self.saved}"
        if self.message:
            status += f" | {self.message}"
        cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(view, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return view

    def run(self) -> None:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (13, 10, 32):
                if len(self.points) != 4:
                    self.message = "4 corners needed"
                elif not is_clockwise_convex(self.points):
                    self.message = "wrong order: 1 top-left, then clockwise"
                elif not self.save():
                    print("All photos annotated.")
                    break
            elif key in (ord("u"), 8) and self.points:
                self.points.pop()
            elif key == ord("d"):
                self.go(1)
            elif key == ord("a"):
                self.go(-1)
        cv2.destroyAllWindows()
        print(f"Saved {self.saved} photos.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images_dir", nargs="?", default=DATASET_DIR / "photos", type=Path)
    parser.add_argument("--dataset", default=DATASET_DIR, type=Path, help="folder with annotated/ and labels/")
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        raise SystemExit(f"No such folder: {args.images_dir}")
    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No photos in {args.images_dir}")
    print(f"{len(images)} photos to annotate")
    Annotator(images, args.dataset).run()


if __name__ == "__main__":
    main()
