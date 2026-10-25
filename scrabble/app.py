"""Local web app for trying the whole pipeline on photos and seeing the results.

Usage (from the repository root):
  ./start app
  python -m streamlit run scrabble/app.py

Photos: one photo at a time, with the detected corners and grid, the rectified board with the
read tiles (unsure ones in red) and the board as a clean digital grid.
Game (the default, on test_photos/gameplay): photos of the board after each move, in name order
(2 before 10). Step through the moves and see who played which words, how every word's points
were counted (tile values, premium squares) and the running totals. A refused move can be
accepted anyway (its tiles go on the board for 0 points).

Photos come from a folder under test_photos/, from files uploaded in the browser, or (in the
Photos view) from the computer's camera.
"""

import hashlib
import html
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from scrabble.board import CELLS, rectify
from scrabble.corners import draw, fit, load_model as load_corner_model, predict_corners
from scrabble.game import (BINGO, LETTER_BONUS, PREMIUMS, RACK, VALUES, WORD_BONUS, Game, MoveError,
                           natural_key)
from scrabble.letters import IMAGE_EXTS, MODEL_PATH, draw_read, load_model, read_board
from scrabble.tiles import EMPTY, ROW_NAMES, CELL_PX

PHOTOS_ROOT = Path("test_photos")
DEFAULT_FOLDER = PHOTOS_ROOT / "gameplay"
DEFAULT_CORNER_MODEL = "corners_200.pt"
UNSURE = 0.7                    # same threshold as the red tiles in draw_read
# Premium squares as printed on the board: (short name, name, colour, text colour).
PREMIUM_STYLE = {
    "W": ("TW", "triple word", "#d9463b", "#fff"),
    "w": ("DW", "double word", "#f2aea3", "#5a1d16"),
    "L": ("TL", "triple letter", "#2f6fc0", "#fff"),
    "l": ("DL", "double letter", "#a9cdeb", "#123a5c"),
}
PLAIN = "#1f7a6d"
BONUS = LETTER_BONUS | WORD_BONUS
CSS = """
<style>
.sb {border-collapse: collapse; font-family: sans-serif; margin: 4px 0}
.sb th {width: 34px; height: 22px; font-size: 11px; color: #888; font-weight: normal}
.sb td {width: 34px; height: 34px; padding: 0; border: 1px solid #0d4d44; position: relative;
        text-align: center; font-size: 9px; font-weight: bold}
.sb td.tile {background: #f3e2b3; color: #222; font-size: 18px}
.sb td.tile sub {position: absolute; right: 2px; bottom: 1px; font-size: 8px; font-weight: normal}
.sb td.word {box-shadow: inset 0 0 0 2px #e0a800}
.sb td.new {background: #ffd84d; box-shadow: inset 0 0 0 2px #b07d00}
.sb td.unsure {color: #d0021b}
.sb td.misread {box-shadow: inset 0 0 0 3px #d0021b}
.sb .badge {position: absolute; left: 0; top: 0; font-size: 7px; line-height: 9px; padding: 0 2px;
            border-bottom-right-radius: 3px}
.legend {display: flex; flex-wrap: wrap; gap: 6px 14px; font: 12px sans-serif; margin: 6px 0}
.legend span.chip {display: inline-block; width: 22px; height: 16px; border-radius: 3px; margin-right: 4px;
                   vertical-align: middle; text-align: center; font-size: 9px; font-weight: bold; line-height: 16px}
.card {border: 1px solid #ddd; border-radius: 8px; padding: 8px 12px; margin: 0 0 10px; font-family: sans-serif}
.card .title {font-size: 16px; font-weight: bold; margin-bottom: 6px}
.card .title small {font-weight: normal; color: #888; margin-left: 6px}
.card .title b {float: right; font-size: 18px}
.card .tiles {display: flex; flex-wrap: wrap; gap: 4px; align-items: flex-start}
.card .t {width: 46px; text-align: center}
.card .face {position: relative; height: 38px; line-height: 38px; border-radius: 4px; background: #f3e2b3;
             font-size: 20px; font-weight: bold; color: #222; border: 2px solid transparent}
.card .face.new {background: #ffd84d; border-color: #b07d00}
.card .face sub {position: absolute; right: 3px; bottom: -12px; font-size: 9px; font-weight: normal}
.card .face.old {opacity: .55}
.card .pts {font-size: 12px; margin-top: 2px}
.card .tag {display: inline-block; font-size: 9px; font-weight: bold; padding: 0 3px; border-radius: 3px; margin-top: 1px}
.card .sum {margin-top: 6px; font-size: 13px; color: #444}
.total {font: 15px sans-serif; margin: 4px 0 12px}
</style>
"""


# --- models and reading -----------------------------------------------------

@st.cache_resource
def models(corner_path):
    return load_corner_model(corner_path), load_model()


@st.cache_data(max_entries=300, show_spinner=False)
def analyse(name, data, corner_path):
    """Corners, read board and preview images of one photo (given as file bytes)."""
    # cv2.imread applies the EXIF orientation of phone photos, like the labels; go through a file.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"photo{Path(name).suffix or '.jpg'}"
        path.write_bytes(data)
        image = cv2.imread(str(path))
    if image is None:
        return None
    corner_model, model = models(corner_path)
    corners, conf = predict_corners(corner_model, image)
    labels, prob = read_board(model, image, corners)
    small = fit(image, 1000)
    s = small.shape[0] / image.shape[0]
    return {
        "photo": cv2.cvtColor(draw(small, corners * s, conf), cv2.COLOR_BGR2RGB),
        "board": cv2.cvtColor(draw_read(rectify(image, corners, CELL_PX), labels, prob), cv2.COLOR_BGR2RGB),
        "labels": labels, "prob": prob, "conf": conf,
    }


# --- drawing ----------------------------------------------------------------

def tag(premium, cls="badge"):
    short, name, colour, text = PREMIUM_STYLE[premium]
    return f'<span class="{cls}" style="background:{colour};color:{text}" title="{name}">{short}</span>'


def board_html(board, prob=None, new=(), words=(), misread=()):
    """The board as a digital grid: premium squares named and in colour, tiles with their values.
    `new` tiles (this move) are yellow and show the premium square under them; other tiles of
    the move's `words` get a yellow frame."""
    rows = ["<tr><th></th>" + "".join(f"<th>{c + 1}</th>" for c in range(CELLS)) + "</tr>"]
    for r in range(CELLS):
        cells = [f"<th>{ROW_NAMES[r]}</th>"]
        for c in range(CELLS):
            symbol, premium = board[r][c], PREMIUMS[r][c]
            if symbol == EMPTY:
                if premium == ".":
                    cells.append(f'<td style="background:{PLAIN}"></td>')
                else:
                    short, name, colour, text = PREMIUM_STYLE[premium]
                    label = "&#9733;" if (r, c) == (CELLS // 2, CELLS // 2) else short
                    cells.append(f'<td style="background:{colour};color:{text}" title="{name}">{label}</td>')
                continue
            classes = ["tile"]
            if (r, c) in new:
                classes.append("new")
            elif (r, c) in words:
                classes.append("word")
            if prob is not None and prob[r][c] < UNSURE:
                classes.append("unsure")
            if (r, c) in misread:
                classes.append("misread")
            badge = tag(premium) if (r, c) in new and premium != "." else ""
            title = f"{ROW_NAMES[r]}{c + 1}" + (f": {prob[r][c]:.0%} sure" if prob is not None else "")
            cells.append(f'<td class="{" ".join(classes)}" title="{title}">{badge}{html.escape(symbol)}'
                         f"<sub>{VALUES.get(symbol, '')}</sub></td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    legend = "".join(f'<div><span class="chip" style="background:{colour};color:{text}">{short}</span>{name}'
                     f'{" (letter ×" if p in LETTER_BONUS else " (word ×"}{BONUS[p]})</div>'
                     for p, (short, name, colour, text) in PREMIUM_STYLE.items())
    return (CSS + '<table class="sb">' + "".join(rows) + "</table>" + f'<div class="legend">{legend}</div>')


def word_card(word):
    """How one word's points add up: every tile's value, the premium squares it used, the total."""
    tiles, parts = [], []
    for l in word.letters:
        face = "new" if l.new else "old"
        pts = f"{l.value}×{BONUS[l.premium]}" if l.premium in LETTER_BONUS else str(l.value)
        under = tag(l.premium, "tag") if l.premium else ("" if l.new else '<span class="tag" style="color:#999">on board</span>')
        tiles.append(f'<div class="t"><div class="face {face}">{html.escape(l.symbol)}<sub>{l.value}</sub></div>'
                     f'<div class="pts">{pts}</div>{under}</div>')
        parts.append(pts)
    letters_total = sum(l.points for l in word.letters)
    text = f"{' + '.join(parts)} = {letters_total}"
    word_premiums = [PREMIUM_STYLE[l.premium][1] for l in word.letters if l.premium in WORD_BONUS]
    if word_premiums:
        text += f", × {word.multiplier} ({' and '.join(word_premiums)}) = {word.score}"
    direction = "across" if word.across else "down"
    return (f'<div class="card"><div class="title">{html.escape(word.text)}<small>{word.start} {direction}</small>'
            f'<b>{word.score}</b></div><div class="tiles">{"".join(tiles)}</div><div class="sum">{text}</div></div>')


def score_html(player, move):
    cards = "".join(word_card(w) for w in move.words)
    bonus = f" + {BINGO} for playing all {RACK} tiles" if move.bingo else ""
    parts = " + ".join(str(w.score) for w in move.words) + bonus
    sum_ = f"{parts} = " if len(move.words) > 1 or bonus else ""
    return CSS + cards + f'<div class="total">{html.escape(player)}: {sum_}<b>{move.score}</b> points</div>'


def show_board(board, **kwargs):
    st.html(board_html(board, **kwargs))


def describe_words(move):
    words = ", ".join(f"{w.text} {w.score}" for w in move.words)
    return words + (f", +{BINGO} for all {RACK} tiles" if move.bingo else "")


# --- photo sources ----------------------------------------------------------

def photo_folders():
    if not PHOTOS_ROOT.is_dir():
        return []
    folders = [PHOTOS_ROOT, *sorted(p for p in PHOTOS_ROOT.rglob("*") if p.is_dir())]
    return [f for f in folders if any(p.suffix.lower() in IMAGE_EXTS for p in f.iterdir())]


def choose_photos(allow_camera):
    """[(name, bytes)] picked in the sidebar."""
    sources = ["Folder", "Upload"] + (["Camera"] if allow_camera else [])
    source = st.sidebar.radio("Photos from", sources, horizontal=True)
    if source == "Folder":
        folders = photo_folders()
        if not folders:
            st.sidebar.warning(f"No photos under {PHOTOS_ROOT}/")
            return []
        default = folders.index(DEFAULT_FOLDER) if DEFAULT_FOLDER in folders else 0
        folder = st.sidebar.selectbox("Folder", folders, index=default, format_func=str)
        paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
        return [(p.name, p.read_bytes()) for p in paths]
    if source == "Upload":
        files = st.sidebar.file_uploader("Photos", type=[e[1:] for e in IMAGE_EXTS], accept_multiple_files=True)
        return sorted(((f.name, f.getvalue()) for f in files or []), key=lambda f: natural_key(Path(f[0])))
    shot = st.camera_input("Take a photo of the board")
    return [("camera.jpg", shot.getvalue())] if shot else []


# --- views ------------------------------------------------------------------

def stepper(label, labels, key, default=0):
    """A slider over `labels` with ◀ and ▶ buttons for one step back or forward; returns the index."""
    n = len(labels)
    if n == 1:
        return 0
    if st.session_state.get(key, n) >= n:          # first run, or fewer photos than before
        st.session_state[key] = min(default, n - 1)

    def shift(step):
        st.session_state[key] = min(max(st.session_state[key] + step, 0), n - 1)

    k = st.session_state[key]
    back, slider, forward = st.columns([1, 12, 1], vertical_alignment="bottom")
    back.button("◀", key=f"{key}_back", on_click=shift, args=(-1,), disabled=k == 0, width="stretch",
                help="One step back")
    forward.button("▶", key=f"{key}_forward", on_click=shift, args=(1,), disabled=k == n - 1, width="stretch",
                   help="One step forward")
    with slider:
        return st.select_slider(label, range(n), key=key, format_func=lambda i: labels[i])


def photos_view(photos, corner_path):
    k = stepper("Photo", [name for name, _ in photos], "photo")
    name, data = photos[k]
    with st.spinner("Reading the board..."):
        result = analyse(name, data, corner_path)
    if result is None:
        st.error(f"Can't open {name}")
        return
    labels, prob = result["labels"], result["prob"]
    tiles = int((labels != EMPTY).sum())
    unsure = [(f"{ROW_NAMES[r]}{c + 1}", labels[r, c], f"{prob[r, c]:.0%}")
              for r in range(CELLS) for c in range(CELLS) if labels[r, c] != EMPTY and prob[r, c] < UNSURE]

    left, right = st.columns(2)
    left.image(result["photo"], caption=f"{name}: detected corners (confidence) and grid", width="stretch")
    right.image(result["board"], caption="Rectified board with the read tiles (red = unsure)", width="stretch")
    left, right = st.columns([3, 2])
    with left:
        show_board(labels, prob=prob)
        st.caption("Hover over a tile to see how sure the network is. Red letters: unsure.")
    with right:
        st.metric("Tiles read", tiles)
        st.write("Corner confidence: " + ", ".join(f"{i + 1}: {c:.2f}" for i, c in enumerate(result["conf"])))
        if unsure:
            st.write(f"Unsure tiles (< {UNSURE:.0%}):")
            st.dataframe(pd.DataFrame(unsure, columns=["cell", "read as", "sure"]), hide_index=True)
        st.text_area("Board as text", "\n".join(" ".join(row) for row in labels), height=330)


def play_game(photos, corner_path, players):
    """One step per photo: who played, the move (or why it was refused), board and scores after it."""
    forced = st.session_state.setdefault("forced", set())     # photos accepted despite a refused move
    game, steps = Game(players), []
    progress = st.progress(0.0, "Reading the photos...")
    for i, (name, data) in enumerate(photos):
        result = analyse(name, data, corner_path)
        progress.progress((i + 1) / len(photos), f"Reading {name}...")
        key = hashlib.md5(data).hexdigest()
        step = {"name": name, "key": key, "result": result, "player": game.player, "move": None, "error": None,
                "forced": None, "misread": []}
        if result is None:
            step["error"] = "can't open the photo"
        else:
            try:
                game.play(result["labels"])
            except MoveError as e:
                step["error"] = str(e)
                if key in forced:
                    step["error"], step["forced"] = None, str(e)
                    game.play(result["labels"], force=True)
            if not step["error"]:
                step["move"] = game.history[-1][1]
                step["misread"] = list(game.misread)
        step["board"], step["scores"] = game.board.copy(), dict(game.scores)
        steps.append(step)
    progress.empty()
    return steps


def game_view(photos, corner_path):
    names = st.sidebar.text_input("Players (comma separated)", "Player 1, Player 2")
    players = [p.strip() for p in names.split(",") if p.strip()] or ["Player 1"]
    steps = play_game(photos, corner_path, players)
    forced = st.session_state["forced"]

    k = stepper("Move", [f"{i + 1}. {s['name']}" for i, s in enumerate(steps)], "move")
    step = steps[k]
    move = step["move"]

    columns = st.columns(len(players))
    for col, p in zip(columns, players):
        before = steps[k - 1]["scores"][p] if k else 0
        col.metric(p, step["scores"][p], step["scores"][p] - before or None)

    board_col, score_col = st.columns([11, 9])
    with score_col:
        st.subheader(f"Move {k + 1}: {step['player']}")
        if step["error"]:
            st.error(f"Not counted: {step['error']}. The turn stays with {step['player']}.")
            if st.button("Accept this board anyway (0 points)"):
                forced.add(step["key"])
                st.rerun()
        elif step["forced"]:
            st.warning(f"Accepted anyway for 0 points ({step['forced']}).")
            if st.button("Undo accepting this board"):
                forced.discard(step["key"])
                st.rerun()
        elif move is None:
            st.info("No new tiles: pass or exchange.")
        else:
            st.html(score_html(step["player"], move))
        if step["misread"]:
            st.warning("Read differently than before, kept the old tiles at: " + ", ".join(step["misread"]))
    with board_col:
        new = {(r, c) for r, c, _ in move.tiles} if move else set()
        words = {l.cell for w in move.words for l in w.letters} if move else set()
        misread = {(ROW_NAMES.index(m[0]), int(m[1:]) - 1) for m in step["misread"]}
        show_board(step["board"], new=new, words=words, misread=misread)
        st.caption("Yellow: tiles of this move, with the premium square they cover. Yellow frame: tiles already "
                   "on the board that are part of this move's words. Red frame: old tile read differently in "
                   "this photo (the game keeps the old one).")

    if step["result"] is not None:
        left, right = st.columns(2)
        left.image(step["result"]["photo"], caption=f"{step['name']}: detected corners and grid", width="stretch")
        right.image(step["result"]["board"], caption="Rectified board with the read tiles (red = unsure)",
                    width="stretch")

    rows = []
    for i, s in enumerate(steps):
        m = s["move"]
        words = (f"not counted: {s['error']}" if s["error"] else f"accepted anyway: {s['forced']}" if s["forced"]
                 else describe_words(m) if m else "pass")
        rows.append({"#": i + 1, "photo": s["name"], "player": s["player"], "words": words,
                     "points": m.score if m else 0,
                     **{p: s["scores"][p] for p in players}})
    st.subheader("All moves")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def main():
    st.set_page_config(page_title="SCRABBLE", layout="wide")
    st.sidebar.title("SCRABBLE")
    view = st.sidebar.radio("View", ["Game", "Photos"], horizontal=True)
    corner_models = sorted(Path("models").glob("corners*.pt"))
    if not corner_models or not MODEL_PATH.exists():
        st.error("Models missing: train them first (python -m scrabble.corners train, "
                 "python -m scrabble.letters train).")
        return
    names = [p.name for p in corner_models]
    default = names.index(DEFAULT_CORNER_MODEL) if DEFAULT_CORNER_MODEL in names else 0
    corner_path = st.sidebar.selectbox("Corner model", corner_models, index=default, format_func=lambda p: p.name)
    photos = choose_photos(allow_camera=view == "Photos")
    if not photos:
        st.info("Pick photos in the sidebar.")
        return
    (photos_view if view == "Photos" else game_view)(photos, str(corner_path))


main()
