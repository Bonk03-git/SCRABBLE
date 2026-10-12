# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SCRABBLE is a system for analyzing a Scrabble game in real time. It reads the board, the tiles on it, and each player's score, much like chess.com's live analysis during chess tournaments. The user talks to Claude in Polish, but **everything in the repository is written in English**: code, comments, on-screen text, messages, README and folder names.

**Hard rule: YOLO must not be used anywhere in this project**, in any version or form (ultralytics, darknet, etc.).

## Commands

The stack is Python 3.11 (`/usr/bin/python3.11`) + OpenCV, with a local virtualenv in `.venv`. numpy is pinned to 2.4.x because 2.5+ requires Python 3.12.

```bash
./start [annotate] [IMAGES_DIR]                      # mark board corners (default IMAGES_DIR: dataset/photos)
/usr/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt   # setup
env -u PYTHONPATH .venv/bin/pytest -q                # all tests
env -u PYTHONPATH .venv/bin/pytest tests/test_annotate.py::test_save_moves_photo_and_writes_label   # single test
```

`./start` unsets `PYTHONPATH` and creates `.venv` if it is missing. It also sets `QT_QPA_PLATFORM=xcb`, because the user runs Wayland and opencv-python's bundled Qt has only the xcb plugin. It symlinks `/usr/share/fonts/truetype/dejavu` into `cv2/qt/fonts`, because that bundled Qt ships no fonts and ignores `QT_QPA_FONTDIR`. Always unset `PYTHONPATH` when running anything by hand: the user's shell exports ROS Jazzy / MoveIt paths (Python 3.12 site-packages), which leak foreign packages into the venv.

## Architecture

The current scope is only building the board dataset. The user deliberately removed the earlier classical detector and model code; don't re-add detection or training code unless asked. A non-YOLO model is planned on top of this dataset later.

The dataset flow, with every path under `dataset/`:
1. The user takes the photos with a phone and copies them into `photos/`. There is deliberately no camera capture code.
2. `scrabble/annotate.py` is an OpenCV GUI. The user clicks the 4 corners of the 15×15 cell grid IN ORDER: 1 is the board's top-left (the board's, not the photo's), then clockwise. Points are stored exactly in click order and never re-sorted, so they encode board orientation. Don't reintroduce automatic corner sorting. Saving is refused unless `is_clockwise_convex` holds. On save, it writes `labels/<photo stem>.json` as `{"image": "<name>", "corners": [[x, y] x4]}`, in original-resolution pixels, and then moves the photo to `annotated/`. Labels are linked to photos by file stem. Running it on `dataset/annotated` reloads the existing labels and rewrites them in place without moving anything. It refuses to overwrite a photo of the same name that is already in `annotated/`. README.md holds the user-facing instructions.

`cv2.imread` applies the EXIF orientation of phone photos, so label coordinates refer to the rotated image. Any future loader must read photos the same way and must not use `IMREAD_IGNORE_ORIENTATION`.
