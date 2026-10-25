# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SCRABBLE is a system for analyzing a Scrabble game in real time. It reads the board, the tiles on it, and each player's score, much like chess.com's live analysis during chess tournaments. The user talks to Claude in Polish, but **everything in the repository is written in English**: code, comments, on-screen text, messages, README and folder names.

**Hard rule: YOLO must not be used anywhere in this project**, in any version or form (ultralytics, darknet, etc.).

## Commands

The stack is Python 3.11 (`/usr/bin/python3.11`) + OpenCV, with a local virtualenv in `.venv`. numpy is pinned to 2.4.x because 2.5+ requires Python 3.12.

```bash
./start [annotate] [IMAGES_DIR]                      # mark board corners (default IMAGES_DIR: dataset/photos)
./start app                                          # test app in the browser (Streamlit)
/usr/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # setup
env -u PYTHONPATH .venv/bin/pytest -q                # all tests
env -u PYTHONPATH .venv/bin/pytest tests/test_annotate.py::test_save_moves_photo_and_writes_label   # single test
```

`./start` unsets `PYTHONPATH` and creates `.venv` if it is missing. It also sets `QT_QPA_PLATFORM=xcb`, because the user runs Wayland and opencv-python's bundled Qt has only the xcb plugin. It symlinks `/usr/share/fonts/truetype/dejavu` into `cv2/qt/fonts`, because that bundled Qt ships no fonts and ignores `QT_QPA_FONTDIR`. Always unset `PYTHONPATH` when running anything by hand: the user's shell exports ROS Jazzy / MoveIt paths (Python 3.12 site-packages), which leak foreign packages into the venv.

## Architecture

The current scope: the corner dataset and model, the top-down board view, tile reading (tile dataset and classifier) and scoring (`scrabble/game.py`). `scrabble/app.py` is a simple Streamlit test app (`./start app`) for checking the pipeline and the scoring by eye (Game view with per-word score breakdown, Single photo view); it is not the final frontend. Its defaults (Game view, `corners_200.pt`, `test_photos/gameplay`) are the constants at the top of the file; README.md describes how to use it. Live camera tracking and the final frontend come later; don't add them unless asked.

The dataset flow, with every path under `dataset/`:
1. The user takes the photos with a phone and copies them into `photos/`. There is deliberately no camera capture code.
2. `scrabble/annotate.py` is an OpenCV GUI. The user clicks the 4 outer corners of the whole printed board (coordinate strips and SCRABBLE bar included) IN ORDER: 1 is the board's top-left (the board's, not the photo's), then clockwise. Points are stored exactly in click order and never re-sorted, so they encode board orientation. Don't reintroduce automatic corner sorting. Saving is refused unless `is_clockwise_convex` holds. On save, it writes `labels/<photo stem>.json` as `{"image": "<name>", "corners": [[x, y] x4]}`, in original-resolution pixels, and then moves the photo to `annotated/`. Labels are linked to photos by file stem. Running it on `dataset/annotated` reloads the existing labels and rewrites them in place without moving anything. It refuses to overwrite a photo of the same name that is already in `annotated/`. README.md holds the user-facing instructions.
3. `scrabble/corners.py` trains a ResNet18 heatmap model on these labels (`models/corners.pt`, gitignored) that predicts the 4 board corners. `scrabble/board.py` maps board corners to the 15×15 cell grid with the fixed `GRID_IN_BOARD` fractions (measured on the average of all rectified labelled photos), warps the photo to a top-down grid and cuts it into cells. `python -m scrabble.check grid` re-measures `GRID_IN_BOARD`; `python -m scrabble.check labels` draws contact sheets for checking labels.
4. `scrabble/tiles.py` draws 6 tile layouts (the whole Polish tile set in alphabetical order from a different start cell each) and cuts labelled 96×96 cell crops (64 px cell + 16 px context) from photos of them into `dataset/tiles/cells/<class>/` (gitignored, regenerable). Tile photos live in `dataset/tiles/layout_NN/{photos,annotated,labels}`, annotated with `./start dataset/tiles/layout_NN/photos --dataset dataset/tiles/layout_NN`. `scrabble/letters.py` trains a ResNet18 cell classifier (34 classes: 32 letters, `_blank`, `_empty`; `models/letters.pt`) with validation on whole held-out photos, and `predict` runs the full photo → corners → cells → letters pipeline. Current accuracy and known weaknesses are in the "Current results" section of README.md; update it after retraining.
5. `scrabble/game.py` finds the move between two read boards (new tiles in one line, no gaps, touching old tiles, the first move on the centre) and scores it: Polish tile values, premium squares only under new tiles, cross words, +50 for all 7 tiles. Words are not checked against a dictionary and a blank scores 0; both are deliberate for now, and so is the 7-tile limit per move. `Game` keeps the tiles it has accepted: a later photo that reads an old tile differently doesn't change it (reported in `misread`), and `play(board, force=True)` accepts a refused move's tiles for 0 points. `python -m scrabble.game PHOTOS_DIR` scores a game from photos in the terminal.

`cv2.imread` applies the EXIF orientation of phone photos, so label coordinates refer to the rotated image. Any future loader must read photos the same way and must not use `IMREAD_IGNORE_ORIENTATION`.
