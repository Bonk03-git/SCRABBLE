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


def board_homography(board_corners, cell=CELL):
    """3x3 matrix from photo pixels to the rectified grid (board corner 1 at its top-left)."""
    side = CELLS * cell
    square = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    return cv2.getPerspectiveTransform(grid_corners(board_corners), square)


def rectify(image, board_corners, cell=CELL):
    """Square top-down image of the 15x15 grid, CELLS * cell pixels on a side."""
    side = CELLS * cell
    return cv2.warpPerspective(image, board_homography(board_corners, cell), (side, side), flags=cv2.INTER_LINEAR)


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
