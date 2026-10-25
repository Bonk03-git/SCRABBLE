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

`scrabble/corners.py` trains a network (ResNet18 + heatmaps, no YOLO) that finds the 4 board corners in label order. It uses the photos in `dataset/annotated/` with their labels in `dataset/labels/`, plus the labelled tile photos in `dataset/tiles/layout_*/` (`--dataset DIR ...` picks other folders). While training, every photo is shown at a new random rotation (any angle), scale, position and slight perspective, with its corner labels transformed the same way, so the labelled photos don't need rotated copies.

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

Compare corner models on the same labelled photos (the validation split of the training; `--all` for every labelled photo):

```powershell
python -m scrabble.corners evaluate                                   # every models/corners*.pt
python -m scrabble.corners evaluate --models models/corners.pt        # one model
```

Check the results (any key = next photo, `q` = quit):

```powershell
python -m scrabble.corners predict                 # photos in dataset/photos
python -m scrabble.corners predict path\to\photos   # a folder or a single file
python -m scrabble.corners predict --camera         # live from the camera
```

The preview shows corners 1–4, the detected 15×15 grid and each corner's confidence (0–1).

## Top-down board view

`scrabble/board.py` warps a photo by the detected board corners into a square top-down view of the 15×15 grid and cuts it into cells (`rectify`, `split_cells`).

The corner model is often a few pixels off, which shifts the cells near the board's edges by up to half a cell. So after the network, `fit_grid` moves the grid onto the board's printed lines. It warps the photo to a top-down view, finds where each of the 16 vertical and 16 horizontal grid lines really is in every row and column (thin bright lines; tiles and premium squares are simply skipped), and fits the perspective that puts the grid lines there, ignoring points that don't agree. If too few lines are found, or the grid would have to move more than 0.6 cell, it keeps the network's corners. Every command that reads boards does this (`predict_corners(..., fit=True)`).

Preview the photo next to the rectified board:

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

## Scoring moves

`scrabble/game.py` compares the board before and after a move and scores it like a referee would: the new tiles must lie in one row or column without gaps and touch the tiles already on the board (the first move must cover the centre square). Each word made counts, cross words included; premium squares count only under the tiles placed in this move, and playing all 7 tiles adds 50. Words are not checked against a dictionary yet, and a blank always scores 0.

```bash
python -m scrabble.game path/to/photos --players Ala Ola
```

The photos, sorted by name (`2.jpg` before `10.jpg`), must show the board after each move, starting with the first one. Every photo prints who played, the words with their scores and the move's total; a photo without new tiles counts as a pass or an exchange, and a photo that is not a legal move is skipped with the reason. The totals are printed at the end.

Tiles never move once played, so the game remembers the tiles it has accepted and looks for new tiles only on empty cells. If a later photo reads an old tile differently (a misread, or a grid that slid a little), the old tile is kept and the cell is reported. `--corner-model models/corners_200.pt` picks another corner model (default: `models/corners.pt`). `--accept-refused` puts the tiles of a refused move on the board for 0 points instead of skipping the photo (like the button in the test app), and `--no-fit` turns off fitting the grid to the board lines. The test app below shows the same game with the full score breakdown.

## Test app

A simple local web app shows what the system reads and how it scores, so you can check it with your own eyes. Start it from the repository root:

```bash
./start app                                # Linux
python -m streamlit run scrabble/app.py    # Windows (in the activated .venv)
```

It opens in the browser at http://localhost:8501; stop it with Ctrl+C in the terminal. It needs both models (`models/corners*.pt` and `models/letters.pt`, see the training sections above).

**Sidebar.** It opens on the **Game** view, the `corners.pt` corner model and the `test_photos/gameplay` folder. You can change:
- the view: **Game** or **Single photo**;
- where the photos come from (**Photos**): a folder under `test_photos/`, files uploaded in the browser, or (Single photo view only) a shot from the computer's camera;
- the number of players and their names (**Players**, Game view), in playing order;
- under **Settings**: the corner model (any `models/corners*.pt`) and **Fit the grid to the board lines** (on by default; turn it off to see what the corner model alone gives).

**Moving between photos or moves.** Drag the slider, or use the buttons on either side of it: first, one step back, one step forward, last. The keyboard works too: **←** and **→** for one step, **Home** and **End** for the first and the last.

**Game view.** The photos are the board after each move, sorted by name (`2.jpg` before `10.jpg`), starting with the first move. For the selected move you see:
- a card for every player with their total and the points just scored; the player who made the move is framed, and the leader is marked with a ★;
- the digital board (tab **Board**). Premium squares are coloured and named as on the real board: **TW** triple word (red), **DW** double word (pink, the centre ★ too), **TL** triple letter (dark blue), **DL** double letter (light blue). The tiles of this move are yellow with an orange frame, and a small badge on a tile shows the premium square under it. Tiles already on the board that belong to this move's words have a yellow frame. A red frame marks an old tile that this photo read differently; the game keeps the old one. The legend under the board lists only what the board shows;
- the photo with the detected corners and grid (tab **Photo**) and the rectified board with the read tiles (tab **Top-down view**);
- on the right, who played and a card for every word made, main word and cross words. It shows each tile with its value, the premium square it used (a premium counts only under a tile placed in this move; old tiles are marked "on board"), and the sum, for example `1×2 + 2 + 2 + 3 + 3×2 + 5 + 2×2 = 24`, then `× 2 (double word) = …` when a word premium was covered. Below it is the move's total, with +50 when all 7 tiles were played;
- under it all, **All moves**: a table of every move with its status, words, points and the running totals. The shown move is highlighted; tick a row's checkbox to jump to that move.

If a move is refused (for example 8 new tiles at once, tiles not in one line, or a misread tile far from the others), the reason is shown and the turn stays with the same player. The tiles read in that photo but not counted are drawn on the board with a dashed red frame, so you can see what went wrong. **Accept this board anyway (0 points)** puts its tiles on the board for no points, so the rest of the game still counts; **Undo accepting this board** reverses it.

**Single photo view.** One photo at a time: the number of tiles read, the number of unsure tiles and the weakest corner confidence (hover over the ⓘ for all four), the digital board (hover over a tile to see how sure the network is; unsure tiles have red letters and a red dot), the photo with the detected corners and grid, the rectified board with the read tiles (red = unsure), the list of unsure tiles and the board as text.

## Current results

Measured on 2026-10-25. Rerun the commands above after every retraining and update this section.

**Corner models.** `models/corners.pt` was trained on all 259 labelled photos (200 board photos plus 59 tile photos) for 120 epochs. `models/corners_200.pt` is the earlier model, trained on the 200 board photos for 60 epochs. `python -m scrabble.corners evaluate` on the same 39 validation photos:

| | mean corner error | median | worst | photos off by more than one cell |
|---|---|---|---|---|
| `corners.pt` | **2.4%** of the board side | 2.1% | 6.3% | 1 |
| `corners_200.pt` | 4.5% | 1.6% | 101% | 1 |

`corners_200.pt` may have seen some of these photos in training, which flatters it.

**Tile classifier** (59 photos of the 6 tile layouts, 15 epochs): 100% of cells correct on the 9 held-out layout photos. With either corner model, no tiles are found on the 5 empty boards in `test_photos/`.

**Whole pipeline on photos no model was trained on**, with and without fitting the grid to the board lines:

| | `corners.pt` + fit | `corners_200.pt` + fit | `corners.pt` alone | `corners_200.pt` alone |
|---|---|---|---|---|
| `test_photos/words/`: tiles read correctly | **348 of 351** | **348 of 351** | 347 of 351 | 339 of 351 |
| `test_photos/words/`: tiles marked unsure | 0 | 0 | 4 | 16 |
| `test_photos/gameplay/`: moves scored correctly | **13 of 13** | **13 of 13** | 11 of 13 | 13 of 13 |
| `test_photos/gameplay/`: old tiles read differently later | 0 | 0 | 15 | 6 |

In the game, the first move is not counted: it has 8 tiles by mistake, so it is accepted anyway (`--accept-refused`). The 3 tiles still missed in the word photos are one tile lying loose across cell borders, in 3 photos of the same board; every tile on a cell is read correctly.

Without the fit, almost every mistake came from the grid sliding about half a cell near one edge of the board, because a corner was found a few pixels off. Examples: SIW read as BIW, WRZOSY as WRZ?BY, and A read as Ą (the neighbouring cell got into the crop). The hand labels are only accurate to about 0.3 cell themselves (the fitted grid of a labelled photo moves that much from the label's grid, and it lies on the printed lines), so `evaluate` measures the network alone and can't score the fitted grid.

**How to improve:**
- Photograph real games with more varied light, angles and distances. With the fit in place, the open questions are boards seen from far away or at a steep angle, and glare on the tiles.
- Test tiles placed upside down, and decide what to do with tiles lying across cell borders.
- Add labelled photos of random words to the tile training data.
