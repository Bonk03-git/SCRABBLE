"""Top-down view of the board: warp a photo by its 4 board corners and cut it into 15x15 cells.

The labels (and so the corner model) mark the outer corners of the whole printed board, which
includes the coordinate strips and the SCRABBLE bar at the bottom. The 15x15 cell grid sits at a
fixed place inside it, GRID_IN_BOARD, measured on the average of all 200 rectified labelled photos.

Usage:
  python -m scrabble.board [IMAGES_DIR_OR_FILE]
  python -m scrabble.board --camera

Corners come from the corner model (models/corners.pt). The preview shows the photo
with the detected grid next to the rectified board.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

CELLS = 15
CELL = 40                       # side of one cell in the rectified board, in pixels
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
# Grid edges (left, top, right, bottom) as fractions of the board side, board corner 1 at (0, 0).
GRID_IN_BOARD = (0.0677, 0.0374, 0.9319, 0.9003)


def grid_corners(board_corners):
    """Corners of the 15x15 cell grid in photo pixels, in the same order as the board corners."""
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H = cv2.getPerspectiveTransform(unit, np.float32(board_corners))
    left, top, right, bottom = GRID_IN_BOARD
    grid = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    return cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(-1, 2)


def board_from_grid(grid):
    """Board corners whose grid (by GRID_IN_BOARD) has these corners; the inverse of grid_corners."""
    left, top, right, bottom = GRID_IN_BOARD
    fractions = np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])
    H = cv2.getPerspectiveTransform(fractions, np.float32(grid))
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    return cv2.perspectiveTransform(unit.reshape(-1, 1, 2), H).reshape(-1, 2)


def line_points(response, cell, margin, vertical):
    """Where each grid line really is, band by band: rows of (x, y, line index, strength).

    `response` (a warped image) shows thin bright lines; the expected grid lines are at
    `margin` + multiples of `cell`. Each line is searched within 0.45 cell of its expected place,
    in every cell-high band (cell-wide for horizontal lines)."""
    r, edge = int(0.45 * cell), max(2, cell // 8)
    image = response if vertical else response.T
    points = []
    for i in range(CELLS + 1):
        x0 = margin + i * cell
        for j in range(CELLS):
            y0, y1 = margin + j * cell + edge, margin + (j + 1) * cell - edge
            profile = image[y0:y1, x0 - r:x0 + r + 1].mean(axis=0)
            k = int(profile.argmax())
            strength = profile[k] - np.median(profile)
            if 0 < k < len(profile) - 1:         # sub-pixel peak from a parabola through 3 samples
                a, b, c = profile[k - 1:k + 2]
                k += 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0
            x, y = x0 - r + k, (y0 + y1) / 2
            points.append((x, y, i, strength) if vertical else (y, x, i, strength))
    return np.array(points)


def fit_line_homography(vertical, horizontal, weights_v, weights_h):
    """3x3 G from warped pixels to grid units such that G(p).x = i for points p on vertical line i
    and G(p).y = i on horizontal line i: weighted linear least squares in 8 unknowns (g33 = 1)."""
    rows, rhs = [], []
    for x, y, i, _ in vertical:
        rows.append([x, y, 1, 0, 0, 0, -i * x, -i * y])
        rhs.append(i)
    for x, y, i, _ in horizontal:
        rows.append([0, 0, 0, x, y, 1, -i * x, -i * y])
        rhs.append(i)
    w = np.sqrt(np.concatenate([weights_v, weights_h]))
    g = np.linalg.lstsq(np.array(rows) * w[:, None], np.array(rhs) * w, rcond=None)[0]
    return np.append(g, 1).reshape(3, 3)


def fit_grid(image, board_corners, cell=CELL, iterations=2, max_shift=0.6):
    """Board corners moved so that the 15x15 grid lies on the board's printed lines.

    The corner model can be a few pixels off, which shifts cells near the board's edges by up to
    half a cell. Each round warps the photo to a top-down view with a one-cell margin, finds the
    thin bright grid lines in every row and column band and fits the homography that puts the
    16 + 16 grid lines on them, dropping points far from the fit. The original corners are kept
    when too few lines are found, or when a grid corner would move more than `max_shift` cells
    (the lines were probably matched to their neighbours)."""
    original = np.float32(board_corners)
    corners = original
    side = CELLS * cell
    target = np.float32([[0, 0], [side, 0], [side, side], [0, side]]) + cell
    units = np.float32([[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]])
    kernel = max(3, cell // 4) | 1
    first = cv2.getPerspectiveTransform(grid_corners(original), target)
    for _ in range(iterations):
        H = cv2.getPerspectiveTransform(grid_corners(corners), target)
        gray = cv2.cvtColor(cv2.warpPerspective(image, H, (side + 2 * cell,) * 2), cv2.COLOR_BGR2GRAY)
        points = [line_points(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones(shape, np.uint8)), cell, cell, v)
                  for v, shape in ((True, (1, kernel)), (False, (kernel, 1)))]
        points = [p[p[:, 3] > max(2.0, np.percentile(p[:, 3], 30))] for p in points]
        weights = [np.ones(len(p)) for p in points]
        for _ in range(4):                       # refit without the points far from the fit
            if min(w.sum() for w in weights) < 2 * CELLS:
                return original
            G = fit_line_homography(*points, *weights)
            for k, p in enumerate(points):
                q = np.c_[p[:, :2], np.ones(len(p))] @ G.T
                weights[k] = (np.abs(q[:, k] / q[:, 2] - p[:, 2]) < 0.15).astype(float)
        grid_warped = cv2.perspectiveTransform(units.reshape(-1, 1, 2), np.linalg.inv(G))
        grid = cv2.perspectiveTransform(grid_warped, np.linalg.inv(H)).reshape(-1, 2)
        if np.abs(cv2.perspectiveTransform(grid.reshape(-1, 1, 2), first).reshape(-1, 2) - target).max() \
                > max_shift * cell:
            return original
        corners = board_from_grid(grid)
    return corners


def board_homography(board_corners, cell=CELL):
    """3x3 matrix from photo pixels to the rectified grid (board corner 1 at its top-left)."""
    side = CELLS * cell
    square = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    return cv2.getPerspectiveTransform(grid_corners(board_corners), square)


def rectify(image, board_corners, cell=CELL):
    """Square top-down image of the 15x15 grid, CELLS * cell pixels on a side."""
    side = CELLS * cell
    return cv2.warpPerspective(image, board_homography(board_corners, cell), (side, side), flags=cv2.INTER_LINEAR)


def cell_crops(image, board_corners, cell=CELL, pad=0):
    """CELLS x CELLS array of cell crops, each cell + 2 * pad pixels on a side (pad = context around the cell)."""
    side, k = CELLS * cell, cell + 2 * pad
    shift = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    big = cv2.warpPerspective(image, shift @ board_homography(board_corners, cell), (side + 2 * pad, side + 2 * pad),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.stack([np.stack([big[r * cell:r * cell + k, c * cell:c * cell + k] for c in range(CELLS)])
                     for r in range(CELLS)])


def split_cells(board):
    """Cut a rectified board into a CELLS x CELLS x cell x cell (x channels) array; [row, col] = cell."""
    cell = board.shape[0] // CELLS
    rows = board[:CELLS * cell, :CELLS * cell].reshape(CELLS, cell, CELLS, cell, *board.shape[2:])
    return rows.swapaxes(1, 2)


def draw_board(board):
    """The rectified board with cell lines."""
    out = board.copy()
    cell = board.shape[0] // CELLS
    for i in range(CELLS + 1):
        p = min(i * cell, board.shape[0] - 1)
        cv2.line(out, (p, 0), (p, board.shape[0]), (0, 255, 255), 1)
        cv2.line(out, (0, p), (board.shape[1], p), (0, 255, 255), 1)
    return out


def side_by_side(photo, board, height=600):
    """Photo and board scaled to the same height, next to each other."""
    views = [cv2.resize(v, None, fx=height / v.shape[0], fy=height / v.shape[0]) for v in (photo, board)]
    return np.hstack(views)


def main():
    from scrabble.corners import draw, load_model, predict_corners
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default="test_photos")
    parser.add_argument("--camera", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    def view(image):
        corners, conf = predict_corners(model, image, device)
        return side_by_side(draw(image, corners, conf), draw_board(rectify(image, corners)))

    if args.camera:
        cap = cv2.VideoCapture(0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("board", view(frame))
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
            cv2.imshow("board", view(image))
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
