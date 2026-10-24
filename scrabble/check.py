"""Checks of the labelled data.

Usage:
  python -m scrabble.check labels [DATASET] [--layout N] [--out FILE]
  python -m scrabble.check grid [DATASET]

labels: rectify every labelled photo of DATASET (annotated/ + labels/) by its hand-marked
corners and put them on contact sheets (100 boards each) with the 15x15 grid drawn. With
correct labels every board shows the SCRABBLE logo at the bottom and the grid lines on the
cell borders. With --layout N (for dataset/tiles/layout_NN) the expected tiles of tile
layout N are written in the cells, so misplaced tiles show up too.

grid: measure GRID_IN_BOARD (where the 15x15 grid lies inside the labelled board corners)
on the average of all rectified photos, from the centres of the triple word squares.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scrabble.board import draw_board, rectify

SHEET_COLS = 10


def labelled_photos(dataset):
    """(photo path, corners) for every label in DATASET/labels whose photo is in DATASET/annotated."""
    out = []
    for label in sorted((dataset / "labels").glob("*.json")):
        data = json.loads(label.read_text(encoding="utf-8"))
        photo = dataset / "annotated" / data["image"]
        if photo.exists():
            out.append((photo, np.array(data["corners"], np.float64)))
    return out


def contact_sheets(dataset, layout_index=None, side=200):
    """Lists of rectified boards, SHEET_COLS x 10 per sheet, numbered and named."""
    from scrabble.tiles import CELL_PX, layout, overlay
    grid = layout(layout_index) if layout_index is not None else None
    tiles, names = [], []
    for k, (photo, corners) in enumerate(labelled_photos(dataset)):
        image = cv2.imread(str(photo))          # applies EXIF orientation, like the labels
        board = rectify(image, corners, CELL_PX)
        board = overlay(board, grid) if grid is not None else draw_board(board)
        board = cv2.resize(board, (side, side))
        cv2.putText(board, str(k), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(board, str(k), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(board)
        names.append(f"{k} {photo.name}")
    sheets = []
    for start in range(0, len(tiles), SHEET_COLS * 10):
        chunk = tiles[start:start + SHEET_COLS * 10]
        chunk += [np.zeros_like(chunk[0])] * (-len(chunk) % SHEET_COLS)
        sheets.append(np.vstack([np.hstack(chunk[r:r + SHEET_COLS]) for r in range(0, len(chunk), SHEET_COLS)]))
    return sheets, names


def measure_grid(dataset, side=1000):
    """GRID_IN_BOARD (left, top, right, bottom) measured on the average rectified board."""
    square = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    total, n = np.zeros((side, side, 3), np.float64), 0
    for photo, corners in labelled_photos(dataset):
        image = cv2.imread(str(photo))
        total += cv2.warpPerspective(image, cv2.getPerspectiveTransform(np.float32(corners), square), (side, side))
        n += 1
    mean = (total / n).astype(np.float32)
    b, g, r = cv2.split(mean)
    red = ((r - g > 60) & (r - b > 60)).astype(np.uint8)
    _, _, stats, centres = cv2.connectedComponentsWithStats(red)
    # Triple word squares: big red blobs, minus the red SCRABBLE logo below the grid.
    squares = [c for c, s in zip(centres[1:], stats[1:]) if s[cv2.CC_STAT_AREA] > side * side * 3e-4
               and c[1] < 0.93 * side]
    xs, ys = np.array([c[0] for c in squares]), np.array([c[1] for c in squares])
    x0, x1 = xs[xs < side / 3].mean(), xs[xs > 2 * side / 3].mean()      # columns 1 and 15
    y0, y1 = ys[ys < side / 3].mean(), ys[ys > 2 * side / 3].mean()      # rows A and O
    cx, cy = (x1 - x0) / 14, (y1 - y0) / 14
    grid = ((x0 - cx / 2) / side, (y0 - cy / 2) / side, (x1 + cx / 2) / side, (y1 + cy / 2) / side)
    return grid, n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("labels")
    lab.add_argument("dataset", nargs="?", default="dataset", type=Path)
    lab.add_argument("--layout", type=int, help="tile layout number (1-6) of this folder")
    lab.add_argument("--out", type=Path, help="output file (default: DATASET/check_labels.jpg)")
    gr = sub.add_parser("grid")
    gr.add_argument("dataset", nargs="?", default="dataset", type=Path)
    args = parser.parse_args()

    if args.command == "labels":
        sheets, names = contact_sheets(args.dataset, args.layout - 1 if args.layout else None)
        if not sheets:
            raise SystemExit(f"No labelled photos in {args.dataset}")
        out = args.out or args.dataset / "check_labels.jpg"
        for k, sheet in enumerate(sheets):
            path = out if len(sheets) == 1 else out.with_name(f"{out.stem}_{k + 1}{out.suffix}")
            cv2.imwrite(str(path), sheet)
            print(path)
        out.with_suffix(".txt").write_text("\n".join(names) + "\n", encoding="utf-8")
        print(f"{out.with_suffix('.txt')}  (board number -> photo)")
    else:
        grid, n = measure_grid(args.dataset)
        print(f"Measured on {n} photos:")
        print(f"GRID_IN_BOARD = ({grid[0]:.4f}, {grid[1]:.4f}, {grid[2]:.4f}, {grid[3]:.4f})")


if __name__ == "__main__":
    main()
