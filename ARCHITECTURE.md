# Architecture

How the system works, step by step: the datasets, the models, reading a board, scoring, and the clock with the camera. [README.md](README.md) describes the project as a whole.

## Pipeline overview

```
photo ──► corner model ──► grid fit ──► top-down board ──► 15×15 cells ──► tile classifier ──► board
        (scrabble/corners.py)  (scrabble/board.py)                        (scrabble/letters.py)

board before + board after ──► move and score (scrabble/game.py)

clock press ──► camera page photo ──► read the board ──► score the move ──► clock + game preview
(scrabble/clock.py)  (scrabble/camera_page.py)                              (scrabble/app.py)
```

| Module | What it does |
|---|---|
| `scrabble/annotate.py` | OpenCV tool for marking the board corners on photos (the labels) |
| `scrabble/check.py` | contact sheets for checking labels, grid corners in the labels, `GRID_IN_BOARD` |
| `scrabble/corners.py` | the corner model: training, evaluation, `predict_corners` |
| `scrabble/board.py` | board corners → 15×15 grid, fitting it to the printed lines, top-down view and cells |
| `scrabble/tiles.py` | tile layouts and the labelled cell dataset cut from their photos |
| `scrabble/letters.py` | the tile classifier and the full photo → letters pipeline |
| `scrabble/game.py` | finding and scoring moves, the bag, racks and end of the game |
| `scrabble/clock.py` | the game clock server, scoring every turn |
| `scrabble/camera_page.py` | the browser camera page that takes the photos |
| `scrabble/camera.py` | OpenCV live view for aiming the camera and saving frames |
| `scrabble/app.py` | the Streamlit test app and the live game preview |

## Workflow

The first model finds the 15×15 grid on an **empty board** (calibration before the game starts).

1. **Take photos.** Aim for 200–300 photos of the empty board (500+ is better). Variety matters more than the count:
   - different backgrounds (floor, table, carpet, bedspread) and lighting (daylight, lamp, shadows, glare),
   - different distances (board small or filling the frame) and camera angles (from straight above to steep),
   - the board rotated in every direction, so each side of it ends up at the top of the photo,
   - sometimes a hand or other objects in the frame.

   No tiles on the board.
2. **Label them.** Copy the photos to `dataset/photos/` and mark the corners (see below), then add the grid corners to the labels with `python -m scrabble.check grid-labels` and check the previews (see "Grid corners in the labels").
3. **Train the model** (see "Training the corner model").
4. **Check the results** on new photos, or live on the clock's camera page (the grid over the live view).

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

`scrabble/corners.py` trains a network (ResNet18 + heatmaps, no YOLO) that finds the 4 corners of the 15×15 cell grid in label order (the `"grid"` points of the labels, see "Grid corners in the labels"; run `python -m scrabble.check grid-labels` first if your labels don't have them). The board corners are then computed from the grid corners, so the rest of the pipeline is unchanged. Because the network looks for the grid itself, not for the edge of the board, it also works on board editions with a different margin around the grid. It uses the photos in `dataset/annotated/` with their labels in `dataset/labels/`, plus the labelled tile photos in `dataset/tiles/layout_*/` (`--dataset DIR ...` picks other folders). While training, every photo is shown at a new random rotation (any angle), scale, position and slight perspective, with its corner labels transformed the same way, so the labelled photos don't need rotated copies.

On Windows, set up the environment once and activate it:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Train (the best model is saved to `models/corners.pt`):

```powershell
python -m scrabble.corners train                 # up to 150 epochs by default
python -m scrabble.corners train --epochs 300    # longer training
python -m scrabble.corners train --patience 0    # no early stop: always run all epochs
```

Each epoch prints the validation error: the worst corner's distance as a percentage of the grid side. One cell is about 6.7%. The best model so far is saved after every improvement, so Ctrl+C keeps it. Training stops early when the validation error hasn't dropped by at least 0.05 (`--min-delta`) for 30 epochs (`--patience`). The learning rate is planned for all epochs (high in the middle, falling to zero at the end), so keep the patience generous: the error often drops most near the end.

Compare corner models on the same labelled photos (the validation split of the training; `--all` for every labelled photo):

```powershell
python -m scrabble.corners evaluate                                   # every models/corners*.pt
python -m scrabble.corners evaluate --models models/corners.pt        # one model
```

Check the results (any key = next photo, `q` = quit):

```powershell
python -m scrabble.corners predict                 # photos in dataset/photos
python -m scrabble.corners predict path\to\photos   # a folder or a single file
```

The preview shows corners 1–4, the detected 15×15 grid and each corner's confidence (0–1).

## Top-down board view

`scrabble/board.py` warps a photo by the detected board corners into a square top-down view of the 15×15 grid and cuts it into cells (`rectify`, `split_cells`).

The corner model is often a few pixels off, which shifts the cells near the board's edges by up to half a cell. So after the network, `fit_grid` moves the grid onto the board's printed lines. It warps the photo to a top-down view, finds where each of the 16 vertical and 16 horizontal grid lines really is in every row and column (thin bright lines; tiles and premium squares are simply skipped), and fits the perspective that puts the grid lines there, ignoring points that don't agree. If too few lines are found, or the grid would have to move more than 0.6 cell, it keeps the network's corners. Every command that reads boards does this (`predict_corners(..., fit=True)`).

Preview the photo next to the rectified board:

```bash
python -m scrabble.board test_photos     # a folder or a single file
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

### Grid corners in the labels

```bash
python -m scrabble.check grid-labels      # all labels in dataset/ and dataset/tiles/layout_*/
```

Adds the 4 corners of the 15×15 cell grid to every label as `"grid"` (same order as `"corners"`; your hand-marked `"corners"` stay as they are). The grid starts where `GRID_IN_BOARD` puts it inside your board corners and is then moved onto the board's printed lines. Previews go to `dataset/check_grid/`:
- `sheet_1.jpg`, `sheet_2.jpg`, …: every photo as 4 zoomed grid corners (1 2 on top, 4 3 below, as on the board), with the photo number and how far the fit moved the grid in cells. A correct corner sits exactly on the outer corner of the grid: the outer corner of the triple word square.
- `photos/NNN_<photo>.jpg`: the whole photo with the grid drawn, for a closer look.
- `index.txt`: photo number → photo file and how far the grid moved. "label only" means no printed lines were found, so the grid comes from your corners alone.

The fit can't fix a grid more than about a cell off. `GRID_IN_BOARD` was measured mostly on one board, so on other board editions (for example the one with star-shaped premium squares) the grid can land on the wrong lines; check those photos on the sheets.

## Training the tile classifier

`scrabble/letters.py` trains a ResNet18 (no YOLO) that tells which tile is in a cell: one of the 32 letters, a blank, or nothing. It uses the cells cut by `python -m scrabble.tiles extract`. Validation uses whole photos the network never saw.

```bash
python -m scrabble.letters train                  # 15 epochs by default; the best model goes to models/letters.pt
python -m scrabble.letters predict test_photos    # read boards in photos (a folder or a single file)
```

Each epoch prints the validation accuracy and the most common mistakes, e.g. `W->M x2`. The preview shows the read tiles on the rectified board, with unsure ones in red, and prints the board as text.

## Scoring moves

`scrabble/game.py` compares the board before and after a move and scores it like a referee would: the new tiles must lie in one row or column without gaps and touch the tiles already on the board (the first move must cover the centre square). Each word made counts, cross words included; premium squares count only under the tiles placed in this move, and playing all 7 tiles adds 50. Words are not checked against a dictionary yet, and a blank always scores 0.

```bash
python -m scrabble.game path/to/photos --players Ala Ola
```

The photos, sorted by name (`2.jpg` before `10.jpg`), must show the board after each move, starting with the first one. Every photo prints who played, the words with their scores and the move's total; a photo without new tiles counts as a pass or an exchange, and a photo that is not a legal move is skipped with the reason. The totals are printed at the end.

Tiles never move once played, so the game remembers the tiles it has accepted and looks for new tiles only on empty cells. If a later photo reads an old tile differently (a misread, or a grid that slid a little), the old tile is kept and the cell is reported. On a real board every tile is linked to the centre square through tiles side by side, so a tile read apart from the others (a premium square or a tile beside the board read as a tile) is ignored and reported too. `--corner-model PATH` picks another corner model (default: `models/corners.pt`). `--accept-refused` puts the tiles of a refused move on the board for 0 points instead of skipping the photo (like the button in the test app), and `--no-fit` turns off fitting the grid to the board lines. The test app below shows the same game with the full score breakdown.

## Test app

A simple local web app shows what the system reads and how it scores, so you can check it with your own eyes. Start it from the repository root:

```bash
./start app                                # Linux
python -m streamlit run scrabble/app.py    # Windows (in the activated .venv)
```

It opens in the browser at http://localhost:8501; stop it with Ctrl+C in the terminal. It needs both models (`models/corners*.pt` and `models/letters.pt`, see the training sections above).

**Sidebar.** It opens on the **Game** view, the `corners.pt` corner model and the `test_photos/gameplay/1` folder (the first sample game; `test_photos/gameplay/2` is the second). You can change:
- the view: **Game** or **Single photo**;
- where the photos come from (**Photos**): a folder under `test_photos/` or `dataset/camera/` (games taken with the clock and the camera page, see [Camera](#camera)), or the clock's current game (**Follow the clock's game**), files uploaded in the browser, or (Single photo view only) a shot from the computer's camera;
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

**Single photo view.** One photo at a time (tiles apart from the others are left out and listed under the photo, as in the game): the number of tiles read, the number of unsure tiles and the weakest corner confidence (hover over the ⓘ for all four), the digital board (hover over a tile to see how sure the network is; unsure tiles have red letters and a red dot), the photo with the detected corners and grid, the rectified board with the read tiles (red = unsure), the list of unsure tiles and the board as text.

## Game clock

A game runs on three browser pages, all started by one command:

| Page | Address | What it does |
|---|---|---|
| **Clock** | http://localhost:8600/ | chess-style clock; every press ends a turn and asks the camera page for a photo |
| **Camera** | http://localhost:8600/camera, or on a phone https://<computer's IP>:8643/camera | the live view of a camera with the grid found in it; takes the photo when the clock is pressed |
| **Game preview** | http://localhost:8501/?game=live | the test app (see above) following this game: the board, words and scores after the newest move |

```bash
./start clock                                   # 2 players, 25 minutes each, camera page + preview
./start clock --players Ala Ola Ela --minutes 20
./start clock --photos test_photos/gameplay/2  # replay a sample game's photos instead of the camera
./start clock --clock-only                      # just the clock
```

The clock opens in the browser; the links under it open the camera page and the game preview. Stop it all with Ctrl+C in the terminal. `--port`, `--phone-port` and `--preview-port` change the ports, `--no-preview` doesn't start the preview, `--no-phone` serves the camera page only on this computer.

| Key | Action |
|---|---|
| Enter / Space, or a click on the clock | start the game; then end the turn, and the time moves to the next player |
| `p` | pause / resume |
| `r` | reset all clocks (while paused) |

The player on turn is green. Every end of a turn is printed in the terminal (who played, how long the turn took, the time left). While scoring, the bar at the bottom also shows how many tiles are left in the bag.

**End of the game** follows the PFS tournament rules (Regulamin Turniejów PFS):
- the bag is empty and a player has played the last tile from their rack (5.1.1a);
- four pauses in a row (5.1.1b): every player still playing paused twice in a row. A turn without new tiles is a pause, but an **exchange is not**: end an exchange turn with **X** instead of Enter. The photo can't tell them apart, so a forgotten X counts as a pause. A loss (see below) is not a pause either, but once there are losses among them, six pauses and losses in a row end the game (three for a player playing on alone);
- every player's time has run out (5.1.1c).

When a player's time runs out (the clock reaches 0:00), they make no more moves (6.2.1): their box goes grey and the others play on. A move they were making when the time ran out is taken back (6.2.2): take the tiles off the board before the next press. Playing on alone, two pauses in a row end the game. There are no penalty points for time.

| Key | Action |
|---|---|
| X | end the turn as an exchange (not a pause) |
| S | after a refused move: count it as a loss |
| Enter / Space | after a refused move: it was misread, take a new photo |

**A refused move** (tiles not in one line, not touching the others, the first move off the centre, ...). The clock learns it only after the board is read, when the next player's time is already running. So the clock goes back to the moment it was pressed: the next player gets back the time used meanwhile, the turn is the mover's again and the clock stops, showing why. Then:
- **S**: the move really is wrong. By PFS 3.2.3 that is a loss: the player takes the tiles back, scores 0, and the next player's time runs;
- **Enter**: the photo was misread (a hand over the board, a tile lying across a line). Put it right and press Enter: a new photo is taken and scored for the same player.

The refused photo stays in the game folder, marked in `game.json` as refused (and as a loss after S). If a turn was ended again before the answer came, the clock can't tell whose time to give back, so it only shows the reason.

The bag and every rack are counted from the tiles read on the board: each player starts with 7 tiles and draws back up to 7 while the bag lasts. A misread tile can put the count off by one. When the game ends, the clock stops and asks for the points of the tiles left on each rack. Where the board tells them, they are already filled in: once the bag is empty, a player with an empty rack has 0, and if only one player still has tiles, all the tiles not on the board are theirs. With the bag empty, the number of tiles on each rack is shown next to the name. **Final score** takes each rack's points off its owner's score (5.1.4) and gives the player who played out the points of all the others' racks. The final scores are shown on the clock; `r` starts a new game. The end goes into `game.json` too, and the game preview shows it above the last move.

**Scoring.** Every end of a turn takes a photo, reads the board in it and scores the move for the player who pressed the clock: the clock, not the order of the photos, says who played. The move (words and points, or why it was refused) is shown under the clocks and printed in the terminal, and each player's score is shown under their time. A turn with no new tiles on the board is a pass or an exchange. A refused move puts nothing on the board and waits for S or Enter (above); with `--accept-refused` its tiles go on the board for 0 points instead and the clock goes on, so one misread doesn't block every later move. Resetting the clock (`r`, or reloading the page) starts a new game. The models load once at start (`--corner-model`, `--no-fit` as in `python -m scrabble.game`).

**Replaying photos.** With `--photos DIR`, every end of a turn takes the next photo in `DIR` that has not been used yet (in name order, 2 before 10) instead of the camera page; a turn with no new photo scores nothing. A sample game folder replays the game one press at a time.

## Camera

Mount the camera above the board, looking down or slightly at an angle, so the whole board is in the picture and as large as possible. A 1920×1080 camera gives only about 60 px per cell when the board fills the frame.

**Camera page** (http://localhost:8600/camera, linked from the clock). It uses the camera of the device it is open on: pick it in the list at the top (the page remembers it; a phone starts with its back camera) and allow the browser to use it. The grid that the corner model finds is drawn over the live view about three times a second, in green, or in orange when a corner is unsure (usually the board is not all in view); **Grid** or `g` hides it. Keep the page open during the game: when the clock is pressed, it takes a full-resolution frame (the view flashes) and sends it to the server. If the page is not open, or sends nothing within 10 s, the turn is not scored and the clock says why. On a phone the page also keeps the screen on.

**Phone as the camera.** Browsers give a page the camera only on `localhost` or over https, so the clock also serves its pages over https on port 8643, open to the local network; the terminal and the clock show the address (`Phone: https://192.168.x.y:8643/camera`). Connect the phone to the same Wi-Fi and open that address. The certificate is self-signed (made with openssl in `.cert/`, one per IP address of the computer), so the phone warns the first time: choose to continue anyway (Chrome: *Advanced → Proceed*; Safari: *Show details → visit this website*). If the phone can't reach the page at all, the firewall blocks the port: `sudo ufw allow 8643/tcp`.

Each game gets its own folder, `<date and time>/` inside `--games DIR` (default `dataset/camera/`), made when the game starts; a reset of the clock starts a new folder (one without photos is removed). It holds the photos `001.jpg`, `002.jpg`, ... and `game.json`, the clock's record of the game: the players, for each photo who played it, whether its move was refused, counted as a loss or accepted for 0 points, and at the end why the game ended and the points left on each rack.

**Game preview.** The clock starts the test app with **Follow the clock's game** on: it shows the newest game in `dataset/camera/` and jumps to the newest move by itself after every move (it looks every 2 s). A game is replayed as the clock scored it: the players and who played each photo come from `game.json`, not from the sidebar or the order of the photos, so a pass without a photo doesn't shift the turns. Turn the switch off to pick any folder, e.g. an older game; in `./start app` it is off unless the address has `?game=live`. The folders are also new training data.

**Aiming without the clock** (`./start camera [SOURCE]`): an OpenCV window with the live view of a USB camera (the N of `/dev/videoN`; `ls /sys/class/video4linux/*/name` shows their names) or a stream URL. `g` shows the grid found in the live view, `s` or Space saves the current frame to `dataset/photos/` to annotate it later, `q` or Esc quits.

## Current results

Measured on 2026-10-26, after retraining both models on the CPU (no GPU; about 30 s per corner epoch, 10 s per tile epoch). Rerun the commands above after every retraining and update this section.

**Corner model.** `models/corners.pt` now finds the grid corners. It was trained on all 258 labelled photos (199 board photos plus 59 tile photos; one photo of another board edition was removed). `python -m scrabble.corners evaluate` on the 39 validation photos, as a percentage of the grid side (one cell = 6.7%):

| | mean corner error | median | worst | photos off by more than one cell |
|---|---|---|---|---|
| `corners.pt` (grid corners) | **2.4%** | **1.8%** | 15.2% | 1 |
| the earlier model (board corners, grid from `GRID_IN_BOARD`) | 2.6% | 2.5% | 5.3% | 0 |

The earlier model was trained with a different validation split, so some of these photos were among its training photos and its numbers are a bit too good. The new model's one miss (15.2%, about 2 cells) is a WhatsApp photo (`WhatsApp Image 2026-10-24 at 14.15.08 (5).jpeg`). The earlier model can't be used with the current code: `predict_corners` expects grid corners from the network.

**Tile classifier** (59 photos of the 6 tile layouts, 15 epochs): 100% of cells correct on the 9 held-out layout photos. On the 5 empty boards in `test_photos/`, 3 read empty; on the other 2 the centre star is read as K, but only with 20–24% confidence, so it is marked unsure.

**Whole pipeline on photos no model was trained on**, with and without fitting the grid to the board lines. With the fit, the new corner model gives exactly the same results as the earlier one on the word photos and in games 1 and 2:

| | `corners.pt` + fit | `corners.pt` alone |
|---|---|---|
| `test_photos/words/`: tiles read correctly | **348 of 351** | 346 of 351 (+1 tile on an empty cell) |
| `test_photos/words/`: tiles marked unsure | 0 | 7 |
| `test_photos/gameplay/1/`: moves scored correctly | **13 of 13** | 11 of 13 |
| `test_photos/gameplay/1/`: old tiles read differently later | 0 | 21 |
| `test_photos/gameplay/2/`: moves scored correctly | **18 of 18** | 16 of 18 |
| `test_photos/gameplay/2/`: old tiles read differently later | 1 | 15 |

In game 1, the first move is not counted: it has 8 tiles by mistake, so it is accepted anyway (`--accept-refused`); the final score is 111 : 90. Game 2 (18 moves, two blanks, final score 173 : 202) is legal throughout; with the fit, one old tile (A10) is read differently in one later photo and the game keeps the old one. The 3 tiles still missed in the word photos are one tile lying loose across cell borders, in 3 photos of the same board; every tile on a cell is read correctly.

Without the fit, almost every mistake came from the grid sliding about half a cell near one edge of the board, because a corner was found a few pixels off. Examples: WIGOR read as WTGOR, SIW? as ĄIW?, PALIWO as PALTWO, and Ó read as O. The hand labels are only accurate to about 0.3 cell themselves (the fitted grid of a labelled photo moves that much from the label's grid, and it lies on the printed lines), so `evaluate` measures the network alone and can't score the fitted grid.

Without the fit, the new corner model reads the word photos better than the earlier one: 4 tiles marked unsure instead of 7, and the misreads PĘTO → PFTO and Ó → O are gone.

**A game on another board** (8 photos, not kept in `test_photos/`) was played on another board edition, with star-shaped premium squares, in strong sunlight. The earlier model put the grid in the wrong place there, because `GRID_IN_BOARD` doesn't fit this board. The new model finds the grid correctly on all 8 photos (the fit moves it by at most 0.6 cell). The game still can't be scored, because the tile classifier has never seen this board and these tiles (white, not cream): it misses most tiles (of WĘDKARZ it reads only W, E, D) and reads two pink star squares as blanks. The star squares are now ignored as tiles apart from the others, but where a tile is missed, the tiles behind it are cut off from the rest and ignored as well. The tile classifier needs cells from this board and tile set in its training data.

**How to improve:**
- Photograph real games with more varied light, angles and distances. With the fit in place, the open questions are boards seen from far away or at a steep angle, and glare on the tiles.
- Test tiles placed upside down, and decide what to do with tiles lying across cell borders.
- Add labelled photos of random words to the tile training data.
