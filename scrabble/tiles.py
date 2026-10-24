"""Tile dataset: photograph the board with all tiles laid out in a known order, then cut labelled cells.

Usage:
  python -m scrabble.tiles layouts                 # draw the layout sheets to dataset/tiles/layouts/
  python -m scrabble.tiles extract                 # cut the cells of dataset/tiles/layout_NN/ photos
  python -m scrabble.tiles extract --no-review     # same, without the preview, hand-marked photos only

Layout NN puts the whole tile set in alphabetical order (blanks last), left to right and
row by row, starting at the cell given by LAYOUT_STARTS and wrapping from the last cell
(O15) to the first (A1). Each layout starts somewhere else, so every letter lands on
different premium squares and every cell is sometimes empty.

Photos of layout NN go to dataset/tiles/layout_NN/photos/. The board corners come from
dataset/tiles/layout_NN/labels/ when the photo was annotated by hand (./start
dataset/tiles/layout_NN/photos --dataset dataset/tiles/layout_NN), otherwise from the
corner model. The preview shows the expected tiles over the rectified board: Enter saves
the cells to dataset/tiles/cells/<class>/, s skips the photo, q quits. With --no-review
every photo with hand-marked corners is saved without asking; check them first with
python -m scrabble.check labels dataset/tiles/layout_NN --layout NN.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scrabble.board import CELLS, cell_crops, draw_board, rectify

# Polish Scrabble tile set, 100 tiles; "?" is a blank.
TILE_COUNTS = [
    ("A", 9), ("Ą", 1), ("B", 2), ("C", 3), ("Ć", 1), ("D", 3), ("E", 7), ("Ę", 1), ("F", 1), ("G", 2),
    ("H", 2), ("I", 8), ("J", 2), ("K", 3), ("L", 3), ("Ł", 2), ("M", 3), ("N", 5), ("Ń", 1), ("O", 6),
    ("Ó", 1), ("P", 3), ("R", 4), ("S", 4), ("Ś", 1), ("T", 3), ("U", 2), ("W", 4), ("Y", 4), ("Z", 5),
    ("Ź", 1), ("Ż", 1), ("?", 2),
]
EMPTY, BLANK = ".", "?"
LAYOUT_STARTS = (0, 38, 76, 114, 152, 190)     # first tile's cell index, row-major
ROW_NAMES = "ABCDEFGHIJKLMNO"                   # as printed on the board's left strip
TILES_DIR = Path("dataset/tiles")
CELL_PX = 64                                    # side of one cell in the saved crops
CELL_PAD = 16                                   # context saved around each cell, so crops are 96 x 96
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "arial.ttf"]


def tile_sequence():
    return [letter for letter, count in TILE_COUNTS for _ in range(count)]


def layout(index):
    """CELLS x CELLS grid of tile symbols for layout `index` (0-based)."""
    grid = [[EMPTY] * CELLS for _ in range(CELLS)]
    start = LAYOUT_STARTS[index]
    for i, tile in enumerate(tile_sequence()):
        k = (start + i) % (CELLS * CELLS)
        grid[k // CELLS][k % CELLS] = tile
    return grid


def cell_name(index):
    """Board coordinate of a row-major cell index, e.g. 0 -> A1, 38 -> C9."""
    return f"{ROW_NAMES[index // CELLS]}{index % CELLS + 1}"


def class_name(symbol):
    """Folder name of a cell class."""
    return {EMPTY: "_empty", BLANK: "_blank"}.get(symbol, symbol)


def font(size):
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def draw_texts(image, texts, size, color):
    """Draw (text, (x, y) centre) pairs on a BGR image with a font that has Polish letters."""
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw, f = ImageDraw.Draw(pil), font(size)
    for text, (x, y) in texts:
        draw.text((x, y), text, font=f, fill=color[::-1], anchor="mm", stroke_width=max(1, size // 12),
                  stroke_fill=(0, 0, 0) if sum(color) > 300 else (255, 255, 255))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


# --- layout sheets ------------------------------------------------------------

def layout_sheet(index, cell=56):
    """Printable picture of a layout, with the board's own coordinates (rows A-O, columns 1-15)."""
    grid = layout(index)
    pad, top = cell, 2 * cell
    w, h = pad + CELLS * cell + cell // 2, top + CELLS * cell + cell // 2
    sheet = np.full((h, w, 3), 255, np.uint8)
    for r in range(CELLS):
        for c in range(CELLS):
            x, y = pad + c * cell, top + r * cell
            if grid[r][c] != EMPTY:
                cv2.rectangle(sheet, (x, y), (x + cell, y + cell), (190, 225, 245), -1)
            cv2.rectangle(sheet, (x, y), (x + cell, y + cell), (120, 120, 120), 1)
    start = LAYOUT_STARTS[index]
    sr, sc = divmod(start, CELLS)
    cv2.rectangle(sheet, (pad + sc * cell, top + sr * cell), (pad + (sc + 1) * cell, top + (sr + 1) * cell),
                  (0, 0, 220), 3)
    texts = [(grid[r][c], (pad + c * cell + cell / 2, top + r * cell + cell / 2))
             for r in range(CELLS) for c in range(CELLS) if grid[r][c] != EMPTY]
    sheet = draw_texts(sheet, texts, cell // 2, (40, 40, 40))
    labels = [(str(c + 1), (pad + c * cell + cell / 2, top - cell / 3)) for c in range(CELLS)]
    labels += [(ROW_NAMES[r], (pad / 2, top + r * cell + cell / 2)) for r in range(CELLS)]
    sheet = draw_texts(sheet, labels, cell // 3, (120, 120, 120))
    title = (f"Layout {index + 1:02d}: tiles A..Z, blanks last, row by row from {cell_name(start)} "
             f"(red), wrapping O15 -> A1. SCRABBLE logo at the bottom.")
    return draw_texts(sheet, [(title, (w / 2, cell / 2))], cell // 4, (0, 0, 0))


def write_layouts():
    out = TILES_DIR / "layouts"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(len(LAYOUT_STARTS)):
        path = out / f"layout_{i + 1:02d}.png"
        cv2.imwrite(str(path), layout_sheet(i))
        (TILES_DIR / f"layout_{i + 1:02d}" / "photos").mkdir(parents=True, exist_ok=True)
        print(f"{path}  (photos go to {TILES_DIR / f'layout_{i + 1:02d}' / 'photos'})")
    print(f"{len(tile_sequence())} tiles per layout.")


# --- cell extraction ------------------------------------------------------------

def overlay(board, grid):
    """Rectified board with cell lines and the expected tile in the corner of each cell."""
    cell = board.shape[0] // CELLS
    texts = [(grid[r][c], (c * cell + cell * 0.22, r * cell + cell * 0.25))
             for r in range(CELLS) for c in range(CELLS) if grid[r][c] != EMPTY]
    return draw_texts(draw_board(board), texts, cell // 3, (0, 255, 255))


def photo_corners(layout_dir, photo, model):
    label = layout_dir / "labels" / f"{photo.stem}.json"
    if label.exists():
        return np.array(json.loads(label.read_text(encoding="utf-8"))["corners"], np.float64), "label"
    from scrabble.corners import predict_corners
    return predict_corners(model, cv2.imread(str(photo)))[0], "model"


def save_cells(cells, grid, prefix):
    """Write cell crops (from board.cell_crops) to cells/<class>/<prefix>_<cell>.png."""
    for r in range(CELLS):
        for c in range(CELLS):
            folder = TILES_DIR / "cells" / class_name(grid[r][c])
            folder.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(folder / f"{prefix}_{ROW_NAMES[r]}{c + 1}.png"), cells[r, c])


def extract(review=True):
    model = None
    if review:
        from scrabble.corners import load_model
        model = load_model()
        print("Enter = save cells, s = skip, q = quit.")
    saved = 0
    for i in range(len(LAYOUT_STARTS)):
        layout_dir = TILES_DIR / f"layout_{i + 1:02d}"
        photos = [p for sub in ("photos", "annotated") if (layout_dir / sub).is_dir()
                  for p in sorted((layout_dir / sub).iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        grid = layout(i)
        for photo in photos:
            image = cv2.imread(str(photo))      # applies EXIF orientation, like the labels
            if image is None:
                continue
            if not review:
                if (layout_dir / "labels" / f"{photo.stem}.json").exists():
                    corners, _ = photo_corners(layout_dir, photo, model)
                    save_cells(cell_crops(image, corners, CELL_PX, CELL_PAD), grid, f"L{i + 1:02d}_{photo.stem}")
                    saved += 1
                else:
                    print(f"Skipped {photo.name}: no hand-marked corners")
                continue
            corners, source = photo_corners(layout_dir, photo, model)
            board = rectify(image, corners, CELL_PX)
            view = overlay(board, grid)
            cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(view, f"layout {i + 1:02d} | {photo.name} | corners: {source} | Enter save, s skip, q quit",
                        (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("tiles", view)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Saved {saved} photos.")
                return
            if key in (13, 10, 32):
                save_cells(cell_crops(image, corners, CELL_PX, CELL_PAD), grid, f"L{i + 1:02d}_{photo.stem}")
                saved += 1
            else:
                print(f"Skipped {photo.name}. If the grid was off, mark its corners by hand: "
                      f"./start {layout_dir / 'photos'} --dataset {layout_dir}")
    cv2.destroyAllWindows()
    print(f"Saved {saved} photos. Cells: {TILES_DIR / 'cells'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["layouts", "extract"])
    parser.add_argument("--no-review", action="store_true", help="extract: save hand-marked photos without the preview")
    args = parser.parse_args()
    write_layouts() if args.command == "layouts" else extract(review=not args.no_review)


if __name__ == "__main__":
    main()
