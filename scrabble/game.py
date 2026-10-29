"""Game logic: find the move between two read boards and score it.

A board is a CELLS x CELLS grid of tile symbols ('.' = empty, '?' = blank), as returned by
scrabble.letters.read_board. Words are not checked against a dictionary yet, and a blank
always scores 0 (which letter it stands for is not asked).

Usage:
  python -m scrabble.game PHOTOS_DIR [--players NAME ...]

The game ends as in the PFS tournament rules (Game.end): when the bag is empty and a player has
played their last tile (5.1.1a; the bag and every rack are counted from the tiles played, as
everyone draws back up to 7), when every player still playing passed twice in a row (5.1.1b: four
pauses for two players; an exchange is not a pause, but only the clock can tell it from a pass),
or when every player's time has run out (5.1.1c). A player whose time runs out makes no more
moves and the others play on (6.2.1: alone, two pauses end the game). Then the points of the tiles
left on each rack are taken off its owner's score (5.1.4), and a player who played out gets those
of all the others (final_scores).

The photos in PHOTOS_DIR, sorted by name (numbers by value: 2 before 10), are the board after each move, one move per photo,
starting with the first move. A photo with no new tiles counts as a pass or an exchange. A photo
that is not a legal move is skipped, or with --accept-refused its tiles go on the board for 0
points. The grid is fitted to the board's printed lines unless --no-fit.
"""

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scrabble.board import CELLS
from scrabble.tiles import BLANK, EMPTY, TILE_COUNTS, cell_name

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
RECORD = "game.json"            # the clock's record of a game, next to its photos
TILES = sum(count for _, count in TILE_COUNTS)
PASS_ROUNDS = 2                 # the game ends when every player passed this many times in a row
# Why a game ended: a player played out, every player still playing passed, every player's time ran out.
OUT, PASSES, TIME = "out", "passes", "time"


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


def linked_tiles(board, start):
    """Cells of the tiles linked to the cells in `start` through tiles side by side."""
    seen = {(r, c) for r, c in start if board[r, c] != EMPTY}
    todo = list(seen)
    while todo:
        r, c = todo.pop()
        for a, b in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= a < CELLS and 0 <= b < CELLS and board[a, b] != EMPTY and (a, b) not in seen:
                seen.add((a, b))
                todo.append((a, b))
    return seen


def drop_isolated(board, anchors=(CENTRE,)):
    """`board` without the tiles that are not linked to any anchor cell, and the names of the
    dropped cells. On a real board every tile is linked to the centre square, so a tile apart
    from the others is a misread (a premium square, a tile lying beside the board). If no
    anchor holds a tile (the centre is not read on the first move), the largest group stays."""
    board = np.asarray(board)
    keep = linked_tiles(board, anchors)
    if not keep:
        groups, left = [], {(int(r), int(c)) for r, c in zip(*np.nonzero(board != EMPTY))}
        while left:
            group = linked_tiles(board, [min(left)])
            groups.append(group)
            left -= group
        keep = max(groups, key=len, default=set())
    dropped = [(int(r), int(c)) for r, c in zip(*np.nonzero(board != EMPTY)) if (r, c) not in keep]
    board = board.copy()
    for r, c in dropped:
        board[r, c] = EMPTY
    return board, [cell_name(r * CELLS + c) for r, c in dropped]


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
    or a grid that slid off a little) doesn't change it; those cells end up in `misread`.
    New tiles not linked to the tiles on the board (or to the centre) are ignored, see
    drop_isolated; those cells end up in `ignored`."""

    def __init__(self, players):
        self.players = list(players)
        self.scores = {p: 0 for p in self.players}
        self.board = empty_board()
        self.turn = 0
        self.history = []           # (player, Move or None for a pass / exchange)
        self.misread = []           # cells of old tiles the last read board got wrong
        self.ignored = []           # cells of tiles apart from the others in the last read board
        self.bag = TILES - RACK * len(self.players)
        self.racks = {p: RACK for p in self.players}     # tiles on each rack
        self.passes = 0             # pauses in a row: turns without new tiles that were not exchanges
        self.timed_out = set()      # players whose time ran out: they make no more moves

    @property
    def player(self):
        return self.players[self.turn % len(self.players)]

    def play(self, board, force=False, exchange=False):
        """Score the move that leads to `board` for the current player and pass the turn on.
        Raises MoveError (and changes nothing) if it is not a legal move, unless `force`: then
        the new tiles are accepted anyway, for 0 points (players broke a rule, or a misread).
        A turn without new tiles is a pause, unless `exchange` says the player exchanged tiles."""
        board = np.asarray(board)
        old = self.board != EMPTY
        self.misread = [cell_name(r * CELLS + c) for r, c in zip(*np.nonzero(old & (board != self.board)))]
        seen, self.ignored = drop_isolated(np.where(old, self.board, board),
                                           [CENTRE] + [(int(r), int(c)) for r, c in zip(*np.nonzero(old))])
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
            # The player draws back up to 7 while the bag lasts. A misread can make a move look
            # bigger than the rack; the counts then stay at zero instead of going negative.
            played = min(len(move.tiles), self.racks[player])
            drawn = min(played, self.bag)
            self.racks[player] += drawn - played
            self.bag -= drawn
            self.passes = 0
        else:
            self.passes = 0 if exchange else self.passes + 1
        self.history.append((player, move))
        self.turn += 1
        return player, move


    def time_out(self, player):
        """`player`'s time ran out: they make no more moves, and the pauses are counted anew among
        the players still playing."""
        self.timed_out.add(player)
        self.passes = 0

    def end(self):
        """(OUT, the player) if the last move played out the bag and the rack, (PASSES, None) if every
        player still playing passed twice in a row, (TIME, None) if every player's time ran out,
        otherwise None."""
        playing = [p for p in self.players if p not in self.timed_out]
        if not playing:
            return TIME, None
        if self.history:
            player, move = self.history[-1]
            if move is not None and self.bag == 0 and self.racks[player] == 0:
                return OUT, player
        if self.passes >= PASS_ROUNDS * len(playing):
            return PASSES, None
        return None

    def left_value(self):
        """Points of all the tiles not on the board yet (in the bag and on the racks)."""
        left = Counter({letter: count for letter, count in TILE_COUNTS})
        left.subtract(Counter(self.board[self.board != EMPTY].tolist()))
        return sum(VALUES[letter] * max(count, 0) for letter, count in left.items())


def final_scores(scores, left, out=None):
    """Scores at the end of the game: everyone loses the points `left` on their rack, and the player
    who played `out` (if any) gains those of all the others."""
    final = {p: s - left.get(p, 0) for p, s in scores.items()}
    if out is not None:
        final[out] += sum(v for p, v in left.items() if p != out)
    return final


def end_text(reason, player=None):
    return {OUT: f"{player} played their last tile with the bag empty",
            PASSES: f"every player still playing passed {PASS_ROUNDS} times in a row",
            TIME: "every player's time ran out"}[reason]


def describe(player, move, exchange=None):
    """`exchange`: whether a turn without new tiles was an exchange (True) or a pause (False), if known."""
    if move is None:
        return f"{player}: {'pass or exchange' if exchange is None else 'exchange' if exchange else 'pause'}"
    words = ", ".join(f"{w.text} ({w.start} {'across' if w.across else 'down'}) {w.score}" for w in move.words)
    return f"{player}: {words}{f' + {BINGO} for all {RACK} tiles' if move.bingo else ''} = {move.score}"


def load_record(folder):
    """The clock's record of the game in `folder`, or None:
    {"players": [names], "moves": [{"photo": name, "player": name, "forced": bool, "exchange": bool}]},
    one move per photo in playing order; "forced" means the clock accepted a refused move for 0
    points, "exchange" that the player marked the turn as an exchange on the clock. A finished game
    also has "end": {"reason": OUT / PASSES / TIME, "player": who played out (or null),
    "left": {name: points on their rack}}."""
    path = Path(folder) / RECORD
    return json.loads(path.read_text()) if path.is_file() else None


def save_record(folder, players, moves, end=None):
    record = {"players": list(players), "moves": moves} | ({"end": end} if end else {})
    (Path(folder) / RECORD).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def natural_key(path):
    """Sort key that puts '2.jpg' before '10.jpg'."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def main():
    import cv2
    from scrabble.corners import load_model as load_corner_model, predict_corners
    from scrabble.board import IMAGE_EXTS
    from scrabble.letters import load_model, read_board

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
        if game.ignored:
            print(f"  ignored the tiles at {', '.join(game.ignored)} (apart from the other tiles)")
        if game.end():
            print(f"Game over: {end_text(*game.end())}; the racks' points are not taken off below")
            break
    print("\n" + "\n".join(f"{p}: {s}" for p, s in game.scores.items()))


if __name__ == "__main__":
    main()
