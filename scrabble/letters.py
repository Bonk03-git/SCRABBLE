"""Tile classifier: which tile (a letter, a blank, or nothing) is in a board cell.

A ResNet18 sees a 64x64 crop centred on a cell and picks one of 34 classes: the 32 letters,
_blank and _empty. It trains on the cells cut by `python -m scrabble.tiles extract`
(dataset/tiles/cells/<class>/*.png, 96x96 crops with context) and saves the best model
(highest validation accuracy) to models/letters.pt.

Usage:
  python -m scrabble.letters train [--epochs N]
  python -m scrabble.letters predict [IMAGES_DIR_OR_FILE]
  python -m scrabble.letters predict --camera

Prediction finds the board with the corner model, cuts the cells and shows the read board.
"""

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from scrabble.board import CELLS, cell_crops, rectify, side_by_side
from scrabble.corners import color_jitter, to_tensor
from scrabble.tiles import CELL_PAD, CELL_PX, EMPTY, TILE_COUNTS, TILES_DIR, class_name, draw_texts

CLASSES = [class_name(letter) for letter, _ in TILE_COUNTS] + [class_name(EMPTY)]
INPUT = CELL_PX                 # the network sees one cell; the crops' padding is room for shifts
MODEL_PATH = Path("models/letters.pt")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# --- data -------------------------------------------------------------------

def photo_of(cell_path):
    """Photo id of a cell crop: 'L01_20261024_203525_C9.png' -> 'L01_20261024_203525'."""
    return cell_path.stem.rsplit("_", 1)[0]


def load_cells(folder):
    """(crop, class index, photo id) for every saved cell."""
    samples = []
    for k, name in enumerate(CLASSES):
        for path in sorted((folder / name).glob("*.png")) if (folder / name).is_dir() else []:
            samples.append((cv2.imread(str(path)), k, photo_of(path)))
    return samples


def centre(crop):
    """The cell itself, without the padding."""
    pad = (crop.shape[0] - INPUT) // 2
    return crop[pad:pad + INPUT, pad:pad + INPUT]


def random_cell(crop, rng):
    """A 64x64 view of the cell, shifted, rotated and scaled a little, like an imperfect grid."""
    c = crop.shape[0] / 2
    dx, dy = rng.uniform(-0.12, 0.12, 2) * INPUT
    M = cv2.getRotationMatrix2D((c + dx, c + dy), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
    M[:, 2] += INPUT / 2 - np.array([c + dx, c + dy])
    return cv2.warpAffine(crop, M, (INPUT, INPUT), borderMode=cv2.BORDER_REFLECT)


class CellDataset(torch.utils.data.Dataset):
    def __init__(self, samples, augment):
        self.samples, self.augment = samples, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        crop, k, _ = self.samples[i]
        if self.augment:
            rng = np.random.default_rng()
            img = color_jitter(random_cell(crop, rng), rng)
        else:
            img = centre(crop)
        return to_tensor(img), k


# --- model ------------------------------------------------------------------

def make_model(pretrained=True):
    weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_model(path=MODEL_PATH, device="cpu"):
    model = make_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def read_board(model, image, board_corners, device="cpu"):
    """CELLS x CELLS grids of tile symbols ('.' = empty, '?' = blank) and their probabilities."""
    crops = cell_crops(image, board_corners, CELL_PX, CELL_PAD).reshape(-1, CELL_PX + 2 * CELL_PAD,
                                                                         CELL_PX + 2 * CELL_PAD, 3)
    x = torch.stack([to_tensor(centre(c)) for c in crops]).to(device)
    prob = torch.softmax(model(x).float(), 1).cpu().numpy()
    symbols = {class_name(s): s for s, _ in TILE_COUNTS} | {class_name(EMPTY): EMPTY}
    labels = [symbols[CLASSES[k]] for k in prob.argmax(1)]
    return np.array(labels).reshape(CELLS, CELLS), prob.max(1).reshape(CELLS, CELLS)


# --- training ---------------------------------------------------------------

def train(args):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = load_cells(Path(args.cells))
    photos = sorted({p for _, _, p in samples})
    if len(photos) < 3:
        raise SystemExit("Need cells from at least 3 photos (python -m scrabble.tiles extract).")
    random.Random(0).shuffle(photos)
    val_photos = set(photos[:max(1, round(0.15 * len(photos)))])
    tr = [s for s in samples if s[2] not in val_photos]
    val = [s for s in samples if s[2] in val_photos]
    print(f"{len(tr)} training / {len(val)} validation cells from {len(photos) - len(val_photos)} / "
          f"{len(val_photos)} photos, device: {device}")

    # Rare letters (one tile in the set) are drawn as often as common ones.
    counts = Counter(k for _, k, _ in tr)
    weights = [1.0 / counts[k] for _, k, _ in tr]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=args.per_epoch, replacement=True)
    loader = torch.utils.data.DataLoader(CellDataset(tr, True), batch_size=args.batch, sampler=sampler,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(CellDataset(val, False), batch_size=256)

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(loader))
    best = -math.inf
    MODEL_PATH.parent.mkdir(exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()

        model.eval()
        mistakes = defaultdict(int)
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                for t, p in zip(y.tolist(), pred.tolist()):
                    if t != p:
                        mistakes[(CLASSES[t], CLASSES[p])] += 1
        acc = correct / len(val) * 100
        mark = ""
        if acc > best:
            best = acc
            torch.save(model.state_dict(), MODEL_PATH)
            mark = "  saved"
        worst = ", ".join(f"{t}->{p} x{n}" for (t, p), n in sorted(mistakes.items(), key=lambda m: -m[1])[:5])
        print(f"epoch {epoch:3d}  loss {total / len(loader):.3f}  val acc {acc:6.2f}%  "
              f"({time.time() - t0:.0f}s){mark}  {worst}")
    print(f"Best validation accuracy {best:.2f}%. Model: {MODEL_PATH}")


# --- prediction -------------------------------------------------------------

def draw_read(board, labels, conf):
    """Rectified board with the read tiles; unsure ones (probability < 0.7) in red."""
    cell = board.shape[0] // CELLS
    out = board.copy()
    for unsure in (False, True):
        texts = [(labels[r, c], (c * cell + cell / 2, r * cell + cell / 2))
                 for r in range(CELLS) for c in range(CELLS)
                 if labels[r, c] != EMPTY and (conf[r, c] < 0.7) == unsure]
        out = draw_texts(out, texts, int(cell * 0.6), (0, 0, 255) if unsure else (0, 255, 0))
    return out


def predict(args):
    from scrabble.corners import draw, load_model as load_corner_model, predict_corners
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corner_model, model = load_corner_model(device=device), load_model(device=device)

    def view(image):
        corners, conf = predict_corners(corner_model, image, device)
        labels, prob = read_board(model, image, corners, device)
        print("\n".join(" ".join(row) for row in labels), end="\n\n")
        return side_by_side(draw(image, corners, conf), draw_read(rectify(image, corners, CELL_PX), labels, prob))

    if args.camera:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("letters", view(frame))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    else:
        target = Path(args.path)
        paths = sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS) if target.is_dir() else [target]
        print("Any key = next photo, q = quit.")
        for path in paths:
            image = cv2.imread(str(path))      # applies EXIF orientation, like the labels
            if image is None:
                continue
            cv2.imshow("letters", view(image))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--cells", default=str(TILES_DIR / "cells"))
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--per-epoch", type=int, default=4000, help="augmented cells per epoch")
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("predict")
    p.add_argument("path", nargs="?", default="test_photos")
    p.add_argument("--camera", action="store_true")
    args = parser.parse_args()
    train(args) if args.command == "train" else predict(args)


if __name__ == "__main__":
    main()
