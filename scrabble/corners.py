"""Grid corner detector: a ResNet18 backbone with a heatmap head (SimpleBaseline style).

The network sees a 256x256 letterboxed photo and outputs 4 heatmaps of 64x64, one per
corner of the 15x15 cell grid in label order (1 = the board's top-left, then clockwise).
Each corner is the peak of its heatmap. Grid corners don't depend on the board edition
(coordinate strips, logo bar); predict_corners turns them into board corners (board.board_from_grid)
for the rest of the pipeline.

Usage:
  python -m scrabble.corners train [--dataset DIR [DIR ...]] [--epochs N] [--patience N]
  python -m scrabble.corners predict [IMAGES_DIR_OR_FILE]
  python -m scrabble.corners evaluate [--models FILE ...] [--dataset DIR ...] [--all]

Training reads the "grid" corners of DATASET/labels/*.json (python -m scrabble.check grid-labels) with photos from DATASET/annotated (of every
given DATASET; by default dataset plus the tile layout folders dataset/tiles/layout_*) and
saves the best model (lowest validation error) to models/corners.pt. Training stops early when
the validation error hasn't improved by at least --min-delta for --patience epochs (0 = never).
The learning rate schedule (one cycle) is planned for all --epochs, so a stop in its high
learning rate middle can come before the error drops at the end; keep the patience generous.

evaluate compares corner models on the validation photos of that split (the same photos for
every model; --all uses every labelled photo). A model trained on another split may have seen
some of these photos, which flatters it. It measures the network alone, without fitting the
grid to the board lines. Errors are in % of the grid side (one cell = 6.7%).
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

INPUT = 256
STRIDE = 4
HEATMAP = INPUT // STRIDE
SIGMA = 2.0                     # Gaussian radius of the target heatmaps, in heatmap pixels
PRELOAD_SIDE = 768              # photos are kept in memory with this longer side
CELLS = 15
MODEL_PATH = Path("models/corners.pt")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# --- geometry ---------------------------------------------------------------

def letterbox_matrix(w, h, size=INPUT):
    """3x3 matrix that fits a w x h image into a size x size square, centred."""
    s = size / max(w, h)
    return np.array([[s, 0, (size - s * w) / 2], [0, s, (size - s * h) / 2], [0, 0, 1]], np.float64)


def transform_points(M, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def random_view(w, h, corners, rng):
    """Random 3x3 transform from a photo to the network input that keeps the board in view."""
    if rng.random() < 0.3:
        return letterbox_matrix(w, h)
    centre = corners.mean(axis=0)
    side = np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
    scale = rng.uniform(0.25, 0.85) * INPUT / side
    angle = rng.uniform(0, 2 * math.pi)
    c, s = math.cos(angle) * scale, math.sin(angle) * scale
    # Rotate and scale around the board centre, then place it randomly.
    M = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)
    M[:2, 2] = -M[:2, :2] @ centre
    pts = transform_points(M, corners)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    margin = 4
    tx = rng.uniform(margin - lo[0], INPUT - margin - hi[0]) if hi[0] - lo[0] < INPUT - 2 * margin else INPUT / 2 - (lo[0] + hi[0]) / 2
    ty = rng.uniform(margin - lo[1], INPUT - margin - hi[1]) if hi[1] - lo[1] < INPUT - 2 * margin else INPUT / 2 - (lo[1] + hi[1]) / 2
    M[0, 2] += tx
    M[1, 2] += ty
    # Mild perspective, like a phone held at an angle.
    square = np.float32([[0, 0], [INPUT, 0], [INPUT, INPUT], [0, INPUT]])
    jitter = square + rng.uniform(-0.08, 0.08, (4, 2)).astype(np.float32) * INPUT
    P = cv2.getPerspectiveTransform(square, jitter)
    return P @ M


def color_jitter(img, rng):
    img = img.astype(np.float32)
    img = img * rng.uniform(0.6, 1.4) + rng.uniform(-30, 30)                 # contrast, brightness
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * rng.uniform(0.6, 1.3)                         # saturation
    img += rng.normal(0, rng.uniform(0, 6), img.shape)                        # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    return img


def to_tensor(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1).copy())


def target_heatmaps(pts):
    """Gaussian blobs at the corners given in input pixels."""
    ys, xs = np.mgrid[0:HEATMAP, 0:HEATMAP].astype(np.float32)
    maps = np.zeros((4, HEATMAP, HEATMAP), np.float32)
    for k, (x, y) in enumerate(pts):
        hx, hy = (x + 0.5) / STRIDE - 0.5, (y + 0.5) / STRIDE - 0.5
        maps[k] = np.exp(-((xs - hx) ** 2 + (ys - hy) ** 2) / (2 * SIGMA ** 2))
    return torch.from_numpy(maps)


def decode(heatmaps):
    """Peaks of 4 heatmaps (numpy, 4 x H x W) as input-pixel coordinates and peak values."""
    pts, conf = [], []
    for hm in heatmaps:
        y, x = np.unravel_index(int(np.argmax(hm)), hm.shape)
        fx, fy = float(x), float(y)
        # Quarter-pixel shift towards the higher neighbour.
        if 0 < x < hm.shape[1] - 1:
            fx += 0.25 * np.sign(hm[y, x + 1] - hm[y, x - 1])
        if 0 < y < hm.shape[0] - 1:
            fy += 0.25 * np.sign(hm[y + 1, x] - hm[y - 1, x])
        pts.append(((fx + 0.5) * STRIDE - 0.5, (fy + 0.5) * STRIDE - 0.5))
        conf.append(float(hm[y, x]))
    return np.array(pts), np.array(conf)


# --- data -------------------------------------------------------------------

def load_samples(dataset):
    samples = []
    for label in sorted((dataset / "labels").glob("*.json")):
        data = json.loads(label.read_text(encoding="utf-8"))
        image = cv2.imread(str(dataset / "annotated" / data["image"]))   # applies EXIF orientation
        if image is None:
            print(f"Skipping {label.name}: photo not found")
            continue
        s = PRELOAD_SIDE / max(image.shape[:2])
        image = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if "grid" not in data:
            raise SystemExit(f"{label} has no grid corners: run python -m scrabble.check grid-labels")
        samples.append((data["image"], image, np.array(data["grid"], np.float64) * s))
    return samples


class CornerDataset(torch.utils.data.Dataset):
    def __init__(self, samples, augment, repeats=1):
        self.samples, self.augment, self.repeats = samples, augment, repeats

    def __len__(self):
        return len(self.samples) * self.repeats

    def __getitem__(self, i):
        _, image, corners = self.samples[i % len(self.samples)]
        h, w = image.shape[:2]
        if self.augment:
            rng = np.random.default_rng()
            M = random_view(w, h, corners, rng)
            border = tuple(int(v) for v in rng.integers(0, 256, 3))
            img = cv2.warpPerspective(image, M, (INPUT, INPUT), borderValue=border)
            img = color_jitter(img, rng)
        else:
            M = letterbox_matrix(w, h)
            img = cv2.warpPerspective(image, M, (INPUT, INPUT))
        pts = transform_points(M, corners)
        return to_tensor(img), target_heatmaps(pts), torch.from_numpy(pts), torch.from_numpy(corners), \
            torch.from_numpy(M)


# --- model ------------------------------------------------------------------

class CornerNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])     # 512 x 8 x 8
        layers, ch = [], 512
        for _ in range(3):                                               # 8 -> 16 -> 32 -> 64
            layers += [nn.ConvTranspose2d(ch, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
            ch = 256
        self.head = nn.Sequential(*layers, nn.Conv2d(256, 4, 1))

    def forward(self, x):
        return self.head(self.backbone(x))


def load_model(path=MODEL_PATH, device="cpu"):
    model = CornerNet(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_corners(model, image, device="cpu", fit=True):
    """Return the 4 corners (label order) in photo pixels and their heatmap peak values.
    With `fit`, the corners are then moved so that the 15x15 grid lies on the board's printed
    lines (scrabble.board.fit_grid)."""
    from scrabble.board import board_from_grid, fit_grid
    h, w = image.shape[:2]
    M = letterbox_matrix(w, h)
    x = to_tensor(cv2.warpPerspective(image, M, (INPUT, INPUT)))[None].to(device)
    pts, conf = decode(model(x)[0].float().cpu().numpy())
    corners = board_from_grid(transform_points(np.linalg.inv(M), pts))
    return (fit_grid(image, corners) if fit else corners), conf


# --- training ---------------------------------------------------------------

def corner_error(pred, true):
    """Worst corner distance as a percentage of the mean side of the true quadrilateral."""
    side = np.mean([np.linalg.norm(true[i] - true[(i + 1) % 4]) for i in range(4)])
    return np.linalg.norm(pred - true, axis=1).max() / side * 100


def split(samples):
    """(training, validation) samples: a fixed 85 / 15 split."""
    samples = list(samples)
    random.Random(0).shuffle(samples)
    n_val = max(2, round(0.15 * len(samples)))
    return samples[n_val:], samples[:n_val]


def evaluate(args):
    from scrabble.board import grid_corners
    samples = [s for dataset in args.dataset for s in load_samples(Path(dataset))]
    photos = samples if args.all else split(samples)[1]
    print(f"{len(photos)} photos ({'all labelled' if args.all else 'validation split'})")
    for path in args.models:
        model = load_model(path)
        errors = np.array([corner_error(grid_corners(predict_corners(model, image, fit=False)[0]), grid)
                           for _, image, grid in photos])
        print(f"{Path(path).name:20s} corner error % of grid side: mean {errors.mean():5.1f}  "
              f"median {np.median(errors):5.1f}  max {errors.max():6.1f}  "
              f"> 1 cell (6.7%): {int((errors > 100 / CELLS).sum())}")


def train(args):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = [s for dataset in args.dataset for s in load_samples(Path(dataset))]
    if len(samples) < 5:
        raise SystemExit("Need at least 5 labelled photos.")
    tr, val = split(samples)
    print(f"{len(tr)} training / {len(val)} validation photos, device: {device}")

    repeats = max(1, 512 // len(tr))       # about 512 augmented views per epoch
    loader = torch.utils.data.DataLoader(CornerDataset(tr, True, repeats), batch_size=args.batch,
                                         shuffle=True, num_workers=args.workers, drop_last=True,
                                         persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(CornerDataset(val, False), batch_size=args.batch)

    model = CornerNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader))
    best, best_epoch = math.inf, 0
    MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, target, *_ in loader:
            x, target = x.to(device), target.to(device)
            loss = nn.functional.mse_loss(model(x), target) * 1000
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        model.eval()
        errors = []
        with torch.no_grad():
            for x, _, _, corners, M in val_loader:
                heat = model(x.to(device)).float().cpu().numpy()
                for hm, true, m in zip(heat, corners.numpy(), M.numpy()):
                    pts, _ = decode(hm)
                    errors.append(corner_error(transform_points(np.linalg.inv(m), pts), true))
        err = float(np.mean(errors))
        mark = ""
        if err < best - args.min_delta:
            best, best_epoch = err, epoch
            torch.save(model.state_dict(), MODEL_PATH)
            mark = "  saved"
        print(f"epoch {epoch:3d}  loss {total / len(loader):.3f}  val err {err:5.1f}% of grid side  "
              f"({time.time() - t0:.0f}s){mark}")
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Stopped early: no improvement for {args.patience} epochs.")
            break
    print(f"Best validation error {best:.1f}% of the grid side (epoch {best_epoch}). Model: {MODEL_PATH}")


# --- prediction -------------------------------------------------------------

def grid_lines(corners):
    """The grid's 16 vertical and 16 horizontal lines in photo pixels, as an array of shape
    (32, 2, 2): the two end points of each line. `corners` are board corners."""
    from scrabble.board import grid_corners
    square = np.float32([[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]])
    H = cv2.getPerspectiveTransform(square, grid_corners(corners))
    lines = [line for i in range(CELLS + 1) for line in (([i, 0], [i, CELLS]), ([0, i], [CELLS, i]))]
    return transform_points(H, np.float32(lines).reshape(-1, 2)).reshape(-1, 2, 2)


def draw(image, corners, conf):
    out = image.copy()
    t = max(2, image.shape[1] // 500)
    from scrabble.board import grid_corners
    grid = grid_corners(corners)
    for a, b in grid_lines(corners).astype(np.int32):
        cv2.line(out, tuple(a), tuple(b), (0, 255, 0), max(1, t // 2), cv2.LINE_AA)
    for k, (p, c) in enumerate(zip(grid.astype(np.int32), conf)):
        cv2.circle(out, tuple(p), 4 * t, (0, 0, 255), -1)
        cv2.putText(out, f"{k + 1} ({c:.2f})", tuple(p + 6 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 255), t)
    return out


def fit(image, max_side=900):
    s = max_side / max(image.shape[:2])
    return cv2.resize(image, None, fx=s, fy=s) if s < 1 else image


def predict(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)
    from scrabble.board import browse
    browse(args.path, "corners", lambda image: fit(draw(image, *predict_corners(model, image, device))))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--dataset", nargs="+", help="default: dataset and dataset/tiles/layout_*",
                   default=["dataset", *map(str, sorted(Path("dataset/tiles").glob("layout_*")))])
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--patience", type=int, default=20, help="stop after this many epochs without improvement (0 = never)")
    t.add_argument("--min-delta", type=float, default=0.05, help="smallest drop of the validation error (%% of the grid side) that counts as an improvement")
    e = sub.add_parser("evaluate")
    e.add_argument("--models", nargs="+", default=sorted(map(str, Path("models").glob("corners*.pt"))))
    e.add_argument("--dataset", nargs="+", default=t.get_default("dataset"))
    e.add_argument("--all", action="store_true", help="every labelled photo, not only the validation split")
    p = sub.add_parser("predict")
    p.add_argument("path", nargs="?", default="dataset/photos")
    args = parser.parse_args()
    {"train": train, "evaluate": evaluate, "predict": predict}[args.command](args)


if __name__ == "__main__":
    main()
