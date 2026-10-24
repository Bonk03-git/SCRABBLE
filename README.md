# SCRABBLE
A system for analyzing the progress of a Scrabble game in real time; it reads the board, the tiles on it, and the scores for each player.
It works similarly to the feature offered by chess.com during chess tournaments.

## Workflow

The first model finds the 15×15 grid on an **empty board** (calibration before the game starts).

1. **Take photos.** Aim for 200–300 photos of the empty board (500+ is better). Variety matters more than the count:
   - different backgrounds (floor, table, carpet, bedspread) and lighting (daylight, lamp, shadows, glare),
   - different distances (board small or filling the frame) and camera angles (from straight above to steep),
   - the board rotated in every direction, so each side of it ends up at the top of the photo,
   - sometimes a hand or other objects in the frame.

   No tiles on the board.
2. **Label them.** Copy the photos to `dataset/photos/` and mark the corners (see below).
3. **Train the model** (see "Training the corner model").
4. **Check the results** on new photos or live from the camera.

## Creating labels (marking the board on photos)

The labeling tool builds the photo dataset. On each photo you mark the 4 corners of the board by hand. The tool saves their coordinates and moves the photo to the annotated folder.

### Running it

1. Put the board photos (JPG or PNG) in `dataset/photos/`.
2. From the project folder, run:

   ```bash
   ./start
   ```

   On the first run, the script creates a Python 3.11 environment and installs the packages.

### Point order: THE MOST IMPORTANT RULE

> **Always click the points in the same order:
> 1 = TOP-LEFT corner, then CLOCKWISE.**

```
   1 ───────────── 2
   │               │
   │     BOARD     │
   │               │
   4 ───────────── 3
```

| Point | Corner |
|:---:|---|
| **1** | top-left |
| **2** | top-right |
| **3** | bottom-right |
| **4** | bottom-left |

Why this matters:
- **The tool saves the points exactly in the order you click them** and never reorders them. The order in the label file is the order you clicked.
- This way a label records the **orientation of the board**. If the board is rotated in the photo, point 1 is still the top-left corner **of the board**, even if it lands on, say, the right side of the photo.
- Clicking in the wrong order produces a wrong label. A future model would learn the wrong board orientation from it, and that mistake can't be detected automatically later.

What the tool checks for you:
- Next to each point it shows its number, and the bar at the top tells you which corner to click next, e.g. `click 2: top-right`.
- After 4 clicks, the outline is **green** if the points go clockwise. It is **red** if they go counter-clockwise or the sides cross, and saving is then blocked.
- The tool **can't tell** which corner of the board really is the top-left. Green only means the direction is right. You have to make sure you started from the correct corner.

### What to mark

Click the **outer corners of the whole printed board**, including the strips with the coordinates and the bar with the SCRABBLE logo. Don't click the corners of the box. The 15×15 cell grid sits at a fixed place inside the board, so the code computes it from these corners (`GRID_IN_BOARD` in `scrabble/board.py`). For precise clicks, use the loupe: a 5× zoom around the cursor with a red crosshair. It always sits on the opposite side of the window from the cursor.

### Keys

| Key | Action |
|---|---|
| left click | add the next point (1 → 2 → 3 → 4) |
| right click / `u` / Backspace | undo the last point |
| Enter / Space | save and go to the next photo |
| `d` / `a` | next / previous photo without saving |
| `q` / Esc | quit |

### What happens after saving

```
dataset/
├── photos/      photos waiting to be annotated
├── annotated/   annotated photos (the tool moves them here after saving)
└── labels/      one .json file per annotated photo
```

A label has the same name as its photo, e.g. `annotated/abc.jpeg` → `labels/abc.json`:

```json
{
  "image": "abc.jpeg",
  "corners": [[294.4, 455.7], [1231.4, 437.8], [1254.4, 1425.9], [271.4, 1431.0]]
}
```

- `corners` holds points 1, 2, 3, 4 in that order: top-left, top-right, bottom-right, bottom-left.
- The `[x, y]` coordinates are in pixels of the original photo: `x` from the left, `y` from the top. Large photos are scaled down for display only.
- If `annotated/` already has a photo with the same name, the tool won't overwrite it. It shows a message and leaves the photo in `photos/`.

### Fixing labels

```bash
./start dataset/annotated
```

The tool loads the existing points. Undo them with right click and click them again. After saving, the label is overwritten and the photo stays in `annotated/`.

## Training the corner model

`scrabble/corners.py` trains a network (ResNet18 + heatmaps, no YOLO) that finds the 4 board corners in label order. It uses the photos in `dataset/annotated/` with their labels in `dataset/labels/`.

On Windows, set up the environment once and activate it:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train (the best model is saved to `models/corners.pt`):

```powershell
python -m scrabble.corners train                 # 150 epochs by default
python -m scrabble.corners train --epochs 300    # longer training
```

Each epoch prints the validation error: the worst corner's distance as a percentage of the board side. One cell is about 5.8%.

Check the results (any key = next photo, `q` = quit):

```powershell
python -m scrabble.corners predict                 # photos in dataset/photos
python -m scrabble.corners predict path\to\photos   # a folder or a single file
python -m scrabble.corners predict --camera         # live from the camera
```

The preview shows corners 1–4, the detected 15×15 grid and each corner's confidence (0–1).

## Top-down board view

`scrabble/board.py` warps a photo by the detected board corners into a square top-down view of the 15×15 grid and cuts it into cells (`rectify`, `split_cells`). Preview the photo next to the rectified board:

```bash
python -m scrabble.board test_photos     # a folder or a single file
python -m scrabble.board --camera        # live from the camera
```

## Tile dataset (cells with known letters)

Instead of labelling tiles by hand, lay the whole tile set out on the board in a known order and let the program cut and label the cells.

1. Draw the layout sheets:

   ```bash
   python -m scrabble.tiles layouts
   ```

   This writes `dataset/tiles/layouts/layout_01.png` … `layout_06.png` and creates a `photos/` folder for each layout. Every layout is the same rule: all 100 tiles in alphabetical order (A, Ą, B, …, Ż, then the 2 blanks), left to right and row by row, starting at the red cell and wrapping from O15 back to A1. Rows (A–O) and columns (1–15) match the coordinates printed on the board.
2. Lay out the tiles of one layout. Keep the SCRABBLE logo at the bottom and the letters upright, readable from the logo side.
3. Take 2–3 photos of it from above (different light or angle is good) and copy them to `dataset/tiles/layout_NN/photos/`.
4. Repeat for the other layouts.
5. Cut the cells:

   ```bash
   python -m scrabble.tiles extract
   ```

   Each photo is shown rectified, with the expected tile written in the corner of every cell. Check that the letters match the tiles. Enter saves the cells to `dataset/tiles/cells/<letter>/` (`_empty`, `_blank` for empty cells and blanks), `s` skips the photo, `q` quits. If the grid is off, mark the board corners by hand (`./start dataset/tiles/layout_NN/photos --dataset dataset/tiles/layout_NN`) and run `extract` again: hand-marked corners are used instead of the model's.

To skip the preview, mark the corners of all photos by hand, check them on contact sheets (see "Checking labels") and run `python -m scrabble.tiles extract --no-review`: it saves every hand-marked photo without asking.

If a tile is missing from your set, everything after it shifts by one cell. Fix `TILE_COUNTS` in `scrabble/tiles.py` to match your set before taking photos.

## Checking labels

```bash
python -m scrabble.check labels                                     # corner dataset -> dataset/check_labels.jpg
python -m scrabble.check labels dataset/tiles/layout_01 --layout 1  # tile photos, with the expected tiles
```

Every labelled photo is rectified by its hand-marked corners and put on a contact sheet with the 15×15 grid (the `.txt` next to it maps board numbers to photos). With correct labels every board has the SCRABBLE logo at the bottom and grid lines on the cell borders; with `--layout` the expected tile is written in each cell.

`python -m scrabble.check grid` measures `GRID_IN_BOARD` again from the average of all rectified photos. Run it if the corner labels change, and copy the printed values into `scrabble/board.py`.

## Training the tile classifier

`scrabble/letters.py` trains a ResNet18 (no YOLO) that tells which tile is in a cell: one of the 32 letters, a blank, or nothing. It uses the cells cut by `python -m scrabble.tiles extract`. Validation uses whole photos the network never saw.

```bash
python -m scrabble.letters train                  # 15 epochs by default; the best model goes to models/letters.pt
python -m scrabble.letters predict test_photos    # read boards in photos (a folder or a single file)
python -m scrabble.letters predict --camera       # live from the camera
```

Each epoch prints the validation accuracy and the most common mistakes, e.g. `W->M x2`. The preview shows the read tiles on the rectified board, with unsure ones in red, and prints the board as text.
