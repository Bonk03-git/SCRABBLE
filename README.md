# SCRABBLE

A system for analyzing a Scrabble game in real time. A camera above the board photographs it after every move; the system reads the board and the tiles on it, finds the move, scores it and keeps each player's score, similarly to the live analysis chess.com offers during chess tournaments.

## What it does

- **Finds the board** in a photo: a neural network (ResNet18 with heatmaps) finds the corners of the 15×15 grid, and the grid is then fitted to the board's printed lines, so every cell is cut out precisely, even when the camera looks at the board at an angle.
- **Reads the tiles**: a second network classifies every cell as one of the 32 Polish letters, a blank, or empty.
- **Scores the moves** like a referee: it finds the tiles placed since the last photo, checks that they make a legal placement and counts every word made, with premium squares, cross words and the 50-point bonus for using all 7 tiles.
- **Runs the game**: a chess-style clock in the browser, where every press takes a photo with a USB camera or a phone and scores the move for the player who pressed it. It follows the Polish Scrabble Federation (PFS) tournament rules for pauses, losses, time-outs and the end of the game.
- **Shows the game**: a live preview with the digital board, the words of every move and a per-word score breakdown.

No YOLO is used anywhere in the project.

## Status

Working: the board and tile datasets, both models, reading boards from photos, scoring, the game clock with the camera page and the live game preview. In tests on photos no model was trained on, every move of the two sample games is scored correctly (see "Current results" in [ARCHITECTURE.md](ARCHITECTURE.md#current-results)). Words are not checked against a dictionary yet, and only one game at a time is supported.

## Quick start

Requires Python 3.11 (Linux; Windows works for training and the test app).

```bash
./start clock    # a game: clock, camera page and live preview in the browser
./start app      # test app: check reading and scoring on sample photos
./start          # mark board corners on photos (dataset building)
```

`./start` creates the `.venv` environment and installs `requirements.txt` on the first run. Both models (`models/corners.pt`, `models/letters.pt`) are not in the repository; train them as described in [ARCHITECTURE.md](ARCHITECTURE.md).

Tests:

```bash
env -u PYTHONPATH .venv/bin/pytest -q
```

## Repository layout

```
scrabble/      the code (see ARCHITECTURE.md for every module)
tests/         tests
test_photos/   photos for testing: empty boards, words, sample games (gameplay/1, gameplay/2)
dataset/       photos, labels and tile cells for training (mostly gitignored)
models/        trained models (gitignored)
start          launcher script
```

## Documentation

[ARCHITECTURE.md](ARCHITECTURE.md) describes how everything works and how to use each part: building the datasets, training and checking the models, the top-down board view, scoring, the test app, the game clock, the camera, and the current results.
