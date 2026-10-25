"""Game logic: find the move between two read boards and score it.

A board is a CELLS x CELLS grid of tile symbols ('.' = empty, '?' = blank), as returned by
scrabble.letters.read_board. Words are not checked against a dictionary yet, and a blank
always scores 0 (which letter it stands for is not asked).

Usage:
  python -m scrabble.game PHOTOS_DIR [--players NAME ...]

The photos in PHOTOS_DIR, sorted by name (numbers by value: 2 before 10), are the board after each move, one move per photo,
starting with the first move. A photo with no new tiles counts as a pass or an exchange. A photo
that is not a legal move is skipped, or with --accept-refused its tiles go on the board for 0
points. The grid is fitted to the board's printed lines unless --no-fit.
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scrabble.board import CELLS
from scrabble.tiles import BLANK, EMPTY, cell_name

# Polish tile values; a blank is worth 0.
VALUES = {
    "A": 1, "Ą": 5, "B": 3, "C": 2, "Ć": 6, "D": 2, "E": 1, "Ę": 5, "F": 5, "G": 3,
    "H": 3, "I": 1, "J": 3, "K": 2, "L": 2, "Ł": 3, "M": 2, "N": 1, "Ń": 7, "O": 1,
    "Ó": 5, "P": 2, "R": 1, "S": 1, "Ś": 5, "T": 2, "U": 3, "W": 1, "Y": 2, "Z": 1,
    "Ź": 9, "Ż": 5, BLANK: 0,
}
# Premium squares: W = triple word, w = double word (the centre too), L = triple letter, l = double letter.
PREMIUMS = [
    "W..l...W...l..W",
    ".w...L...L...w.",
    "..w...l.l...w..",
    "l..w...l...w..l",
    "....w.....w....",
    ".L...L...L...L.",
    "..l...l.l...l..",
    "W..l...w...l..W",
    "..l...l.l...l..",
    ".L...L...L...L.",
    "....w.....w....",
    "l..w...l...w..l",
    "..w...l.l...w..",
    ".w...L...L...w.",
    "W..l...W...l..W",
]
LETTER_BONUS = {"L": 3, "l": 2}
WORD_BONUS = {"W": 3, "w": 2}
CENTRE = (CELLS // 2, CELLS // 2)
RACK = 7
BINGO = 50                      # bonus for playing all 7 tiles in one move


class MoveError(ValueError):
    """The change between two boards is not a legal move."""


@dataclass
class Letter:
    symbol: str
    cell: tuple                 # (row, col)
    value: int                  # tile value
    premium: str                # premium square used in this move ('W', 'w', 'L', 'l'), '' if none
    new: bool                   # placed in this move

    @property
    def points(self):
        return self.value * LETTER_BONUS.get(self.premium, 1)


@dataclass
class Word:
    text: str
    start: str                  # board coordinate of the first letter, e.g. "H8"
    across: bool
    letters: list               # Letter for every tile of the word

    @property
    def multiplier(self):
        return int(np.prod([WORD_BONUS.get(l.premium, 1) for l in self.letters]))

    @property
    def score(self):
        return sum(l.points for l in self.letters) * self.multiplier


@dataclass
class Move:
    tiles: list                 # (row, col, symbol) of every newly placed tile
    words: list = field(default_factory=list)
    bingo: bool = False

    @property
    def score(self):
        return sum(w.score for w in self.words) + (BINGO if self.bingo else 0)


def parse(rows):
    """Board from 15 strings, one per row, e.g. for tests."""
    board = np.array([list(r) for r in rows])
    assert board.shape == (CELLS, CELLS), board.shape
    return board


def empty_board():
    return np.full((CELLS, CELLS), EMPTY)


def word_through(board, r, c, dr, dc):
    """Cells of the run of tiles through (r, c) in direction (dr, dc)."""
    while 0 <= r - dr and 0 <= c - dc and board[r - dr, c - dc] != EMPTY:
        r, c = r - dr, c - dc
    cells = []
    while r < CELLS and c < CELLS and board[r, c] != EMPTY:
        cells.append((r, c))
        r, c = r + dr, c + dc
    return cells


def word_letters(board, cells, new):
    """The letters of a word; premium squares count only under the tiles placed in this move."""
    return [Letter(str(board[r, c]), (r, c), VALUES[board[r, c]],
                   PREMIUMS[r][c].strip(".") if (r, c) in new else "", (r, c) in new) for r, c in cells]


def find_move(before, after):
    """The move that turns board `before` into `after`; None if no tile was added."""
    before, after = np.asarray(before), np.asarray(after)
    changed = [(r, c) for r, c in zip(*np.nonzero(before != after))]
    moved = [cell_name(r * CELLS + c) for r, c in changed if before[r, c] != EMPTY]
    if moved:
        raise MoveError(f"Tiles already on the board changed or disappeared: {', '.join(moved)}")
    new = {(int(r), int(c)) for r, c in changed}
    if not new:
        return None
    if len(new) > RACK:
        raise MoveError(f"{len(new)} new tiles, but a player has only {RACK}")
    rows, cols = {r for r, _ in new}, {c for _, c in new}
    if len(rows) > 1 and len(cols) > 1:
        raise MoveError("The new tiles are not in one row or column")

    across = len(rows) == 1 and (len(new) > 1 or len(word_through(after, *next(iter(new)), 0, 1)) > 1)
    dr, dc = (0, 1) if across else (1, 0)
    main = word_through(after, *min(new), dr, dc)
    if not new <= set(main):
        raise MoveError("There is a gap between the new tiles")

    first = not (before != EMPTY).any()
    if first and CENTRE not in new:
        raise MoveError(f"The first move must cover the centre square {cell_name(CENTRE[0] * CELLS + CENTRE[1])}")
    if not first and not any(
            0 <= r + a < CELLS and 0 <= c + b < CELLS and before[r + a, c + b] != EMPTY
            for r, c in new for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0))):
        raise MoveError("The new tiles don't touch any tile already on the board")

    words = [main] + [word_through(after, r, c, dc, dr) for r, c in sorted(new)]
    move = Move(tiles=[(r, c, str(after[r, c])) for r, c in sorted(new)], bingo=len(new) == RACK)
    for i, cells in enumerate(words):
        if len(cells) > 1:
            (r, c) = cells[0]
            move.words.append(Word("".join(after[r, c] for r, c in cells), cell_name(r * CELLS + c),
                                   across if i == 0 else not across, word_letters(after, cells, new)))
    if not move.words:
        raise MoveError("A single tile is not a word")
    return move


class Game:
    """Players take turns; each new board is checked against the last one and scored.

    Tiles never move once played, so the game keeps the tiles it has accepted and looks only
    for new tiles on empty cells. A read board that disagrees about an old tile (a misread,
    or a grid that slid off a little) doesn't change it; those cells end up in `misread`."""

    def __init__(self, players):
        self.players = list(players)
        self.scores = {p: 0 for p in self.players}
        self.board = empty_board()
        self.turn = 0
        self.history = []           # (player, Move or None for a pass / exchange)
        self.misread = []           # cells of old tiles the last read board got wrong

    @property
    def player(self):
        return self.players[self.turn % len(self.players)]

    def play(self, board, force=False):
        """Score the move that leads to `board` for the current player and pass the turn on.
        Raises MoveError (and changes nothing) if it is not a legal move, unless `force`: then
        the new tiles are accepted anyway, for 0 points (players broke a rule, or a misread)."""
        board = np.asarray(board)
        old = self.board != EMPTY
        self.misread = [cell_name(r * CELLS + c) for r, c in zip(*np.nonzero(old & (board != self.board)))]
        seen = np.where(old, self.board, board)
        try:
            move = find_move(self.board, seen)
        except MoveError:
            if not force:
                raise
            move = Move(tiles=[(int(r), int(c), str(seen[r, c])) for r, c in zip(*np.nonzero(seen != self.board))])
        player = self.player
        if move is not None:
            self.scores[player] += move.score
            self.board = seen
        self.history.append((player, move))
        self.turn += 1
        return player, move


def describe(player, move):
    if move is None:
        return f"{player}: pass or exchange"
    words = ", ".join(f"{w.text} ({w.start} {'across' if w.across else 'down'}) {w.score}" for w in move.words)
    return f"{player}: {words}{f' + {BINGO} for all {RACK} tiles' if move.bingo else ''} = {move.score}"


def natural_key(path):
    """Sort key that puts '2.jpg' before '10.jpg'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def main():
    import cv2
    from scrabble.corners import load_model as load_corner_model, predict_corners
    from scrabble.letters import IMAGE_EXTS, load_model, read_board

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", type=Path)
    parser.add_argument("--players", nargs="+", default=["Player 1", "Player 2"])
    parser.add_argument("--corner-model", type=Path, default=Path("models/corners.pt"))
    parser.add_argument("--accept-refused", action="store_true",
                        help="put the tiles of a refused move on the board for 0 points instead of skipping the photo")
    parser.add_argument("--no-fit", action="store_true", help="don't fit the grid to the board's printed lines")
    args = parser.parse_args()

    corner_model, model = load_corner_model(args.corner_model), load_model()
    game = Game(args.players)
    for path in sorted((p for p in args.photos.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key):
        image = cv2.imread(str(path))      # applies EXIF orientation, like the labels
        corners, _ = predict_corners(corner_model, image, fit=not args.no_fit)
        board, _ = read_board(model, image, corners)
        try:
            print(f"{path.name}  {describe(*game.play(board))}")
        except MoveError as e:
            if not args.accept_refused:
                print(f"{path.name}  skipped: {e}")
                print("\n".join(" ".join(row) for row in board))
                continue
            player, _ = game.play(board, force=True)
            print(f"{path.name}  {player}: accepted anyway for 0 points ({e})")
        if game.misread:
            print(f"  kept the old tiles at {', '.join(game.misread)} (read differently in this photo)")
    print("\n" + "\n".join(f"{p}: {s}" for p, s in game.scores.items()))


if __name__ == "__main__":
    main()
