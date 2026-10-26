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

grid-labels: add the corners of the 15x15 cell grid to every label as "grid" (same order as
"corners"): the grid that GRID_IN_BOARD puts inside the hand-marked board corners, moved onto
the board's printed lines by fit_grid. The hand-marked "corners" are kept. Writes previews to
OUT (default dataset/check_grid): sheet_N.jpg with the 4 grid corners of every photo zoomed in
(100 photos per sheet), photos/<number>_<photo>.jpg with the whole grid drawn on each photo,
and index.txt (number, photo, how far the fit moved the grid in cells). A grid that the fit
didn't move is marked "label only": check those closely.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from scrabble.board import CELLS, draw_board, fit_grid, grid_corners, rectify

SHEET_COLS = 10
UNITS = np.float32([[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]])


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


def label_grid(image, corners, max_shift=1.5):
    """Grid corners for a photo labelled with board corners, and the most a corner moved in the fit
    (cells; 0 when fit_grid kept the label). Hand labels can be a cell off, so the fit may move further
    than in reading."""
    fitted = fit_grid(image, corners, iterations=3, max_shift=max_shift)
    if np.allclose(fitted, corners, atol=1e-3):
        return grid_corners(corners), 0.0
    grid = grid_corners(fitted)
    H = cv2.getPerspectiveTransform(np.float32(grid_corners(corners)), UNITS)
    moved = cv2.perspectiveTransform(np.float32(grid).reshape(-1, 1, 2), H).reshape(-1, 2) - UNITS
    return grid, float(np.linalg.norm(moved, axis=1).max())


def draw_grid(image, grid, thickness=1, numbers=True):
    """The image with the 15x15 grid lines (green) and the grid corners (red, numbered unless `numbers` is off)."""
    out = image.copy()
    H = cv2.getPerspectiveTransform(UNITS, np.float32(grid))
    for i in range(CELLS + 1):
        for line in (((i, 0), (i, CELLS)), ((0, i), (CELLS, i))):
            a, b = cv2.perspectiveTransform(np.float32(line).reshape(-1, 1, 2), H).reshape(-1, 2)
            cv2.line(out, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (0, 255, 0), thickness,
                     cv2.LINE_AA)
    for k, (x, y) in enumerate(np.round(grid).astype(int)):
        cv2.circle(out, (x, y), 2 * thickness + 1, (0, 0, 255), -1, cv2.LINE_AA)
        if numbers:
            cv2.putText(out, str(k + 1), (x + 4 * thickness, y - 4 * thickness), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * thickness, (0, 0, 255), max(1, thickness), cv2.LINE_AA)
    return out


def corner_zooms(image, grid, side=120):
    """The 4 grid corners in label order, each zoomed to side x side pixels (about 2.4 cells across)
    with thin grid lines and the corner point drawn after zooming."""
    cell = np.mean([np.linalg.norm(grid[i] - grid[(i + 1) % 4]) for i in range(4)]) / CELLS
    r = max(8, int(1.2 * cell))
    s = side / (2 * r)
    zooms = []
    for k, (x, y) in enumerate(grid):
        zoom = cv2.resize(cv2.getRectSubPix(image, (2 * r, 2 * r), (float(x), float(y))), (side, side),
                          interpolation=cv2.INTER_CUBIC)
        local = (np.float32(grid) - [x - r, y - r]) * s     # the grid in zoom pixels
        out = draw_grid(zoom, local, numbers=False)
        cv2.putText(out, str(k + 1), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        zooms.append(out)
    return zooms


def grid_labels(datasets, out):
    """Add "grid" to every label of DATASETS and write the previews to OUT."""
    (out / "photos").mkdir(parents=True, exist_ok=True)
    tiles, lines = [], []
    for dataset in datasets:
        for label in sorted((dataset / "labels").glob("*.json")):
            data = json.loads(label.read_text(encoding="utf-8"))
            image = cv2.imread(str(dataset / "annotated" / data["image"]))   # applies EXIF orientation
            if image is None:
                print(f"Skipping {label}: photo not found")
                continue
            grid, moved = label_grid(image, np.array(data["corners"], np.float64))
            data["grid"] = [[round(float(x), 1), round(float(y), 1)] for x, y in grid]
            label.write_text(json.dumps(data, indent=2), encoding="utf-8")

            k = len(tiles)
            cell = np.linalg.norm(grid[1] - grid[0]) / CELLS
            drawn = draw_grid(image, grid, max(1, round(cell / 40)))
            zooms = corner_zooms(image, grid)
            note = f"{moved:.2f}" if moved > 0 else "label only"
            tile = np.vstack([np.hstack(zooms[:2]), np.hstack(zooms[2:][::-1])])    # 1 2 / 4 3, as on the board
            cv2.putText(tile, f"{k} {note}", (4, tile.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(tile, f"{k} {note}", (4, tile.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255) if moved > 0 else (0, 0, 255), 1)
            tiles.append(tile)
            s = 1600 / max(drawn.shape[:2])
            cv2.imwrite(str(out / "photos" / f"{k:03d}_{Path(data['image']).stem}.jpg"),
                        cv2.resize(drawn, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
            lines.append(f"{k}\t{dataset / 'annotated' / data['image']}\t{note}")
    for n, start in enumerate(range(0, len(tiles), SHEET_COLS * 10)):
        chunk = tiles[start:start + SHEET_COLS * 10]
        chunk += [np.zeros_like(chunk[0])] * (-len(chunk) % SHEET_COLS)
        sheet = np.vstack([np.hstack(chunk[r:r + SHEET_COLS]) for r in range(0, len(chunk), SHEET_COLS)])
        cv2.imwrite(str(out / f"sheet_{n + 1}.jpg"), sheet)
    (out / "index.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lab = sub.add_parser("labels")
    lab.add_argument("dataset", nargs="?", default="dataset", type=Path)
    lab.add_argument("--layout", type=int, help="tile layout number (1-6) of this folder")
    lab.add_argument("--out", type=Path, help="output file (default: DATASET/check_labels.jpg)")
    gr = sub.add_parser("grid")
    gr.add_argument("dataset", nargs="?", default="dataset", type=Path)
    gl = sub.add_parser("grid-labels")
    gl.add_argument("datasets", nargs="*", type=Path, help="default: dataset and dataset/tiles/layout_*")
    gl.add_argument("--out", type=Path, default=Path("dataset/check_grid"))
    args = parser.parse_args()

    if args.command == "grid-labels":
        datasets = args.datasets or [Path("dataset"), *sorted(Path("dataset/tiles").glob("layout_*"))]
        lines = grid_labels(datasets, args.out)
        unmoved = [line for line in lines if line.endswith("label only")]
        print(f"Added the grid to {len(lines)} labels ({len(unmoved)} not moved by the fit). Previews: {args.out}")
        for line in unmoved:
            print("  label only:", line.split("\t")[1])
    elif args.command == "labels":
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
