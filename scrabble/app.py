"""Local web app for trying the whole pipeline on photos and seeing the results.

Usage (from the repository root):
  ./start app
  python -m streamlit run scrabble/app.py

Game (the default, on test_photos/gameplay/1): photos of the board after each move, in name order
(2 before 10). Step through the moves and see who played which words, how every word's points
were counted (tile values, premium squares) and the running totals. A refused move can be
accepted anyway (its tiles go on the board for 0 points).
Single photo: one photo at a time, with the detected corners and grid, the rectified board with
the read tiles (unsure ones in red) and the board as a clean digital grid.

Photos come from a folder under test_photos/, from files uploaded in the browser, or (in the
Single photo view) from the computer's camera.
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
DEFAULT_FOLDER = PHOTOS_ROOT / "gameplay" / "1"
DEFAULT_CORNER_MODEL = "corners.pt"
UNSURE = 0.7                    # same threshold as the red tiles in draw_read
GAME, PHOTO = "Game", "Single photo"
# Premium squares as printed on the board: (short name, name, colour, text colour).
PREMIUM_STYLE = {
    "W": ("TW", "triple word", "#d9463b", "#fff"),
    "w": ("DW", "double word", "#f2aea3", "#5a1d16"),
    "L": ("TL", "triple letter", "#2f6fc0", "#fff"),
    "l": ("DL", "double letter", "#a9cdeb", "#123a5c"),
}
PLAYER_COLOURS = ["#c8372d", "#2f6fc0", "#c98a00", "#7a4bb5"]
BONUS = LETTER_BONUS | WORD_BONUS
# All sizes on the board are in cqw (percent of the board's width), so it scales with its column.
CSS = """
<style>
.brand {display: flex; gap: 3px; margin: 0 0 4px}
.brand span {position: relative; width: 27px; height: 30px; line-height: 30px; text-align: center;
             border-radius: 4px; font: 800 17px sans-serif; color: #2a2118;
             background: linear-gradient(160deg, #fbefcf, #ecd49c); box-shadow: 0 2px 0 #b8955a}
.brand sub {position: absolute; right: 2px; bottom: 1px; font-size: 7px; line-height: 7px}
.tagline {font-size: 13px; color: #7a705f; margin: 6px 0 2px}

.board-wrap {container-type: inline-size; max-width: 660px; margin: 0 auto}
.board {box-sizing: border-box; display: grid; grid-template-columns: 4cqw repeat(15, minmax(0, 1fr)); gap: .45cqw; padding: 1.6cqw 2cqw 2cqw 1.2cqw;
        background: #0e4b43; border-radius: 2cqw; box-shadow: 0 1.5cqw 4cqw rgba(30, 20, 0, .25),
        inset 0 0 0 .5cqw #0a3a34; font-family: sans-serif; user-select: none}
.board .hd {display: flex; align-items: center; justify-content: center; font-size: 1.9cqw; color: #8fc3b8}
.board .sq {position: relative; aspect-ratio: 1; border-radius: .7cqw; background: #1e7a6b;
            display: flex; align-items: center; justify-content: center; font-size: 1.7cqw; font-weight: 700}
.board .sq.star {font-size: 3.4cqw}
.board .tile {background: linear-gradient(160deg, #fbefcf, #ecd49c); color: #2a2118; font-size: 3.7cqw;
              font-weight: 800; box-shadow: 0 .35cqw 0 #b8955a, 0 .5cqw .9cqw rgba(0, 0, 0, .35)}
.board .tile sub {position: absolute; right: .45cqw; bottom: .15cqw; font-size: 1.45cqw; font-weight: 600}
.board .tile.word {box-shadow: 0 0 0 .32cqw #ffd44d, 0 .35cqw 0 #b8955a}
.board .tile.new {background: linear-gradient(160deg, #fff4b8, #ffd34a); z-index: 1;
                  box-shadow: 0 0 0 .45cqw #f08c00, 0 .6cqw 1.2cqw rgba(0, 0, 0, .45)}
.board .tile.ghost {background: rgba(251, 239, 207, .55); color: #5e1410; box-shadow: none;
                    outline: .4cqw dashed #ff6b5e; outline-offset: -.2cqw}
.board .tile.misread {box-shadow: 0 0 0 .5cqw #e5332a, 0 .35cqw 0 #b8955a}
.board .tile.unsure {color: #c0161b}
.board .tile.unsure::after {content: ""; position: absolute; left: .55cqw; bottom: .55cqw; width: 1cqw; height: 1cqw;
                            border-radius: 50%; background: #e5332a}
.board .badge {position: absolute; left: .25cqw; top: .2cqw; font-size: 1.25cqw; line-height: 1.6cqw;
               padding: 0 .35cqw; border-radius: .4cqw}

.legend {display: flex; flex-wrap: wrap; justify-content: center; gap: 6px 16px; max-width: 660px;
         margin: 12px auto 0; font: 12.5px sans-serif; color: #5c5547}
.legend .chip {display: inline-block; width: 24px; height: 18px; line-height: 18px; border-radius: 4px;
               margin-right: 5px; vertical-align: middle; text-align: center; font-size: 9px; font-weight: 800}
.legend .chip.tile {background: linear-gradient(160deg, #fbefcf, #ecd49c); box-shadow: 0 1px 0 #b8955a}
.legend .chip.new {background: linear-gradient(160deg, #fff4b8, #ffd34a); box-shadow: 0 0 0 2px #f08c00}
.legend .chip.word {box-shadow: 0 0 0 2px #ffd44d}
.legend .chip.misread {box-shadow: 0 0 0 2px #e5332a}
.legend .chip.ghost {background: rgba(251, 239, 207, .6); outline: 2px dashed #ff6b5e; outline-offset: -1px}
.legend .chip.unsure {color: #c0161b}

.players {display: flex; flex-wrap: wrap; gap: 12px; margin: 0 0 4px; font-family: sans-serif}
.player {flex: 1 1 190px; display: flex; align-items: center; gap: 14px; padding: 12px 16px; border-radius: 14px;
         background: #fffdf7; border: 1px solid #e6dcc6; box-shadow: 0 1px 3px rgba(60, 40, 0, .06)}
.player.active {border-color: var(--c); box-shadow: inset 0 0 0 1px var(--c), 0 4px 14px rgba(60, 40, 0, .1)}
.avatar {flex: none; width: 42px; height: 42px; border-radius: 50%; background: var(--c); color: #fff;
         display: flex; align-items: center; justify-content: center; font: 800 19px sans-serif}
.avatar.sm {width: 30px; height: 30px; font-size: 14px}
.player .name {font-size: 14px; color: #6b6457; white-space: nowrap; overflow: hidden; text-overflow: ellipsis}
.player .score {font-size: 30px; font-weight: 800; line-height: 1.05; color: #2b2a26}
.player .delta {display: inline-block; vertical-align: middle; margin-left: 8px; padding: 1px 8px; border-radius: 99px;
                font-size: 13px; font-weight: 700; color: #17693f; background: #dff3e6}
.player .state {margin-left: auto; text-align: right; font-size: 11px; font-weight: 700; letter-spacing: .06em;
                text-transform: uppercase; color: var(--c)}

.move-head {display: flex; align-items: center; gap: 12px; margin: 2px 0 12px; font-family: sans-serif}
.move-head .kicker {font-size: 12px; color: #8a8170; text-transform: uppercase; letter-spacing: .06em}
.move-head .kicker .file {text-transform: none; letter-spacing: 0}
.move-head .who {font-size: 22px; font-weight: 800; color: #2b2a26; line-height: 1.15}
.pill {margin-left: auto; padding: 4px 12px; border-radius: 99px; font-size: 13px; font-weight: 700; white-space: nowrap}
.pill.ok {background: #dff3e6; color: #17693f}
.pill.bad {background: #fde3e1; color: #a3231b}
.pill.warn {background: #fff0c9; color: #8a5a00}
.pill.pass {background: #ece7dc; color: #5c5547}

.card {background: #fffdf7; border: 1px solid #e6dcc6; border-radius: 12px; padding: 12px 14px 10px;
       margin: 0 0 10px; font-family: sans-serif}
.card .title {display: flex; align-items: baseline; gap: 10px; margin-bottom: 10px}
.card .word {font-size: 19px; font-weight: 800; letter-spacing: .08em; color: #2b2a26}
.card .where {font-size: 12px; color: #8a8170}
.card .score {margin-left: auto; font-size: 22px; font-weight: 800; color: #17695d}
.card .tiles {display: flex; flex-wrap: wrap; gap: 6px; align-items: flex-start}
.card .t {width: 44px; text-align: center}
.card .face {position: relative; height: 40px; line-height: 40px; border-radius: 5px; font-size: 21px; font-weight: 800;
             color: #2a2118; background: linear-gradient(160deg, #fbefcf, #ecd49c); box-shadow: 0 2px 0 #b8955a}
.card .face.new {background: linear-gradient(160deg, #fff4b8, #ffd34a); box-shadow: 0 0 0 2px #f08c00, 0 2px 0 #b8955a}
.card .face.old {filter: saturate(.4); opacity: .7}
.card .face sub {position: absolute; right: 3px; bottom: -11px; font-size: 9px; font-weight: 600}
.card .pts {font-size: 12px; margin-top: 3px; color: #2b2a26}
.card .tag {display: inline-block; font-size: 9px; font-weight: 800; padding: 0 4px; border-radius: 3px; margin-top: 2px}
.card .tag.plain {color: #a39a88; font-weight: 600; white-space: nowrap}
.card .sum {margin-top: 8px; padding-top: 8px; border-top: 1px dashed #e6dcc6; font: 12.5px monospace; color: #5c5547}
.total {display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 10px 16px;
        border-radius: 12px; background: #17695d; color: #fff; font: 14px sans-serif}
.total b {font-size: 24px}

.chips {display: flex; flex-wrap: wrap; gap: 6px; font: 13px sans-serif}
.chips span {padding: 3px 10px; border-radius: 99px; background: #fde3e1; color: #a3231b}
.chips b {color: #2b2a26}
</style>
"""


# --- models and reading -----------------------------------------------------

@st.cache_resource
def models(corner_path):
    return load_corner_model(corner_path), load_model()


@st.cache_data(max_entries=300, show_spinner=False)
def analyse(name, data, reading):
    """Corners, read board and preview images of one photo (given as file bytes).
    `reading` is (corner model path, whether to fit the grid to the board's printed lines)."""
    corner_path, fit_grid = reading
    # cv2.imread applies the EXIF orientation of phone photos, like the labels; go through a file.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"photo{Path(name).suffix or '.jpg'}"
        path.write_bytes(data)
        image = cv2.imread(str(path))
    if image is None:
        return None
    corner_model, model = models(corner_path)
    corners, conf = predict_corners(corner_model, image, fit=fit_grid)
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


def board_html(board, prob=None, new=(), words=(), misread=(), ghosts=None):
    """The board as a digital grid: premium squares named and in colour, tiles with their values.
    `new` tiles (this move) are yellow and show the premium square under them; other tiles of
    the move's `words` get a yellow frame. `ghosts` ({cell: symbol}) are tiles read in the photo
    that the game did not put on the board (a refused move)."""
    ghosts = ghosts or {}
    cells = ['<div class="hd"></div>'] + [f'<div class="hd">{c + 1}</div>' for c in range(CELLS)]
    for r in range(CELLS):
        cells.append(f'<div class="hd">{ROW_NAMES[r]}</div>')
        for c in range(CELLS):
            symbol, premium, where = board[r][c], PREMIUMS[r][c], f"{ROW_NAMES[r]}{c + 1}"
            if symbol == EMPTY and (r, c) in ghosts:
                symbol = ghosts[r, c]
                cells.append(f'<div class="sq tile ghost" title="{where}: read in the photo, not on the board">'
                             f"{html.escape(symbol)}<sub>{VALUES.get(symbol, '')}</sub></div>")
                continue
            if symbol == EMPTY:
                if premium == ".":
                    cells.append(f'<div class="sq" title="{where}"></div>')
                else:
                    short, name, colour, text = PREMIUM_STYLE[premium]
                    star = (r, c) == (CELLS // 2, CELLS // 2)
                    cells.append(f'<div class="sq{" star" if star else ""}" style="background:{colour};color:{text}" '
                                 f'title="{where}: {name}">{"&#9733;" if star else short}</div>')
                continue
            classes = ["sq", "tile"]
            if (r, c) in new:
                classes.append("new")
            elif (r, c) in words:
                classes.append("word")
            if prob is not None and prob[r][c] < UNSURE:
                classes.append("unsure")
            if (r, c) in misread:
                classes.append("misread")
            badge = tag(premium) if (r, c) in new and premium != "." else ""
            title = where + (f": {prob[r][c]:.0%} sure" if prob is not None else "")
            cells.append(f'<div class="{" ".join(classes)}" title="{title}">{badge}{html.escape(symbol)}'
                         f"<sub>{VALUES.get(symbol, '')}</sub></div>")
    return f'<div class="board-wrap"><div class="board">{"".join(cells)}</div></div>'


def legend_html(*states):
    """Premium squares, then the tile `states` that the board above can show."""
    items = [f'<span><span class="chip" style="background:{colour};color:{text}">{short}</span>{name} '
             f'({"letter" if p in LETTER_BONUS else "word"} ×{BONUS[p]})</span>'
             for p, (short, name, colour, text) in PREMIUM_STYLE.items()]
    names = {
        "new": ("new", "A", "placed in this move"),
        "word": ("tile word", "A", "part of this move's words"),
        "misread": ("tile misread", "A", "read differently, old tile kept"),
        "ghost": ("ghost", "A", "read in the photo, not counted"),
        "unsure": ("tile unsure", "A", "network unsure (hover for %)"),
    }
    items += [f'<span><span class="chip {cls}">{letter}</span>{text}</span>'
              for cls, letter, text in (names[s] for s in states)]
    return f'<div class="legend">{"".join(items)}</div>'


def word_card(word):
    """How one word's points add up: every tile's value, the premium squares it used, the total."""
    tiles, parts = [], []
    for l in word.letters:
        face = "new" if l.new else "old"
        pts = f"{l.value}×{BONUS[l.premium]}" if l.premium in LETTER_BONUS else str(l.value)
        under = tag(l.premium, "tag") if l.premium else ("" if l.new else '<span class="tag plain">on board</span>')
        tiles.append(f'<div class="t"><div class="face {face}">{html.escape(l.symbol)}<sub>{l.value}</sub></div>'
                     f'<div class="pts">{pts}</div>{under}</div>')
        parts.append(pts)
    letters_total = sum(l.points for l in word.letters)
    text = f"{' + '.join(parts)} = {letters_total}"
    word_premiums = [PREMIUM_STYLE[l.premium][1] for l in word.letters if l.premium in WORD_BONUS]
    if word_premiums:
        text += f", × {word.multiplier} ({' and '.join(word_premiums)}) = {word.score}"
    direction = "across" if word.across else "down"
    return (f'<div class="card"><div class="title"><span class="word">{html.escape(word.text)}</span>'
            f'<span class="where">{word.start} {direction}</span><span class="score">{word.score}</span></div>'
            f'<div class="tiles">{"".join(tiles)}</div><div class="sum">{text}</div></div>')


def score_html(player, move):
    cards = "".join(word_card(w) for w in move.words)
    bonus = f" + {BINGO} for playing all {RACK} tiles" if move.bingo else ""
    parts = " + ".join(str(w.score) for w in move.words) + bonus
    sum_ = f"{parts} = " if len(move.words) > 1 or bonus else ""
    return (cards + f'<div class="total"><span>{html.escape(player)} scores {sum_}</span>'
            f"<b>{move.score}</b></div>")


def avatar(player, colour, cls="avatar"):
    return f'<div class="{cls}" style="--c:{colour}">{html.escape(player[:1].upper())}</div>'


def players_html(players, colours, scores, before, played):
    """One card per player: total after the move, the points just scored, who played the move."""
    leader = max(scores.values())
    cards = []
    for p in players:
        gained = scores[p] - before[p]
        delta = f'<span class="delta">+{gained}</span>' if gained else ""
        state = "played" if p == played else ""
        if leader and scores[p] == leader and list(scores.values()).count(leader) == 1:
            state = (state + "<br>" if state else "") + "&#9733; leading"
        cards.append(f'<div class="player{" active" if p == played else ""}" style="--c:{colours[p]}">'
                     f'{avatar(p, colours[p])}<div style="min-width:0"><div class="name">{html.escape(p)}</div>'
                     f'<div class="score">{scores[p]}{delta}</div></div><div class="state">{state}</div></div>')
    return f'<div class="players">{"".join(cards)}</div>'


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
    sidebar = st.sidebar
    sidebar.subheader("Photos", divider="gray")
    sources = {"Folder": ":material/folder:", "Upload": ":material/upload:"}
    if allow_camera:
        sources["Camera"] = ":material/photo_camera:"
    source = sidebar.segmented_control("Photos from", list(sources), default="Folder", required=True,
                                       format_func=lambda s: f"{sources[s]} {s}", key="source", width="stretch",
                                       label_visibility="collapsed")
    if source not in sources:                      # Camera was picked, then the view changed
        source = "Folder"
    if source == "Folder":
        folders = photo_folders()
        if not folders:
            sidebar.warning(f"No photos under {PHOTOS_ROOT}/")
            return []
        default = folders.index(DEFAULT_FOLDER) if DEFAULT_FOLDER in folders else 0
        folder = sidebar.selectbox("Folder", folders, index=default, format_func=str)
        paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key)
        sidebar.caption(f"{len(paths)} photos, in name order (2 before 10)")
        return [(p.name, p.read_bytes()) for p in paths]
    if source == "Upload":
        files = sidebar.file_uploader("Photos", type=[e[1:] for e in IMAGE_EXTS], accept_multiple_files=True,
                                      help="Photos of the board after each move; they are sorted by name.")
        return sorted(((f.name, f.getvalue()) for f in files or []), key=lambda f: natural_key(Path(f[0])))
    shot = st.camera_input("Take a photo of the board")
    return [("camera.jpg", shot.getvalue())] if shot else []


def choose_players():
    sidebar = st.sidebar
    sidebar.subheader("Players", divider="gray")
    n = sidebar.segmented_control("Number of players", [2, 3, 4], default=2, required=True, key="n_players")
    players = []
    for i in range(n):
        name = sidebar.text_input(f"Player {i + 1}", f"Player {i + 1}", key=f"player_{i}",
                                  help="In playing order: player 1 makes the first move.").strip()
        name = name or f"Player {i + 1}"
        while name in players:                     # scores are kept by name
            name += "'"
        players.append(name)
    return players


# --- views ------------------------------------------------------------------

def stepper(labels, key, default=0):
    """A slider over `labels` with buttons (and arrow keys) for the first, previous, next and last
    step; returns the index."""
    n = len(labels)
    if n == 1:
        return 0
    if st.session_state.get(key, n) >= n:          # first run, or fewer photos than before
        st.session_state[key] = min(default, n - 1)

    def go(i):
        st.session_state[key] = min(max(i, 0), n - 1)

    def button(icon, i, disabled, help, shortcut):
        st.button("", icon=icon, key=f"{key}_{shortcut}", on_click=go, args=(i,), disabled=disabled, help=help,
                  shortcut=shortcut)

    k = st.session_state[key]
    with st.container(horizontal=True, vertical_alignment="center"):
        button(":material/first_page:", 0, k == 0, "First", "Home")
        button(":material/chevron_left:", k - 1, k == 0, "One step back", "Left")
        k = st.select_slider(key, range(n), key=key, format_func=lambda i: labels[i], label_visibility="collapsed",
                             width="stretch")
        button(":material/chevron_right:", k + 1, k == n - 1, "One step forward", "Right")
        button(":material/last_page:", n - 1, k == n - 1, "Last", "End")
    return k


def photos_view(photos, reading):
    k = stepper([name for name, _ in photos], "photo")
    name, data = photos[k]
    with st.spinner("Reading the board..."):
        result = analyse(name, data, reading)
    if result is None:
        st.error(f"Can't open {name}", icon=":material/broken_image:")
        return
    labels, prob, conf = result["labels"], result["prob"], result["conf"]
    tiles = int((labels != EMPTY).sum())
    unsure = [(f"{ROW_NAMES[r]}{c + 1}", labels[r, c], prob[r, c])
              for r in range(CELLS) for c in range(CELLS) if labels[r, c] != EMPTY and prob[r, c] < UNSURE]

    stats = st.columns(3)
    stats[0].metric("Tiles read", tiles, border=True, help=f"In {name}")
    stats[1].metric("Unsure tiles", len(unsure), border=True, help=f"Tiles read with less than {UNSURE:.0%} certainty")
    stats[2].metric("Weakest corner", f"{min(conf):.2f}", border=True,
                    help="Lowest confidence of the 4 detected board corners (1 = sure). All four: "
                         + ", ".join(f"{i + 1}: {c:.2f}" for i, c in enumerate(conf)))

    board_col, side_col = st.columns([11, 9], gap="large")
    with board_col:
        st.html(board_html(labels, prob=prob) + legend_html("unsure"))
    with side_col:
        photo_tab, top_tab = st.tabs([":material/photo: Photo", ":material/grid_on: Top-down view"])
        photo_tab.image(result["photo"], caption="Detected corners (with confidence) and the grid", width="stretch")
        top_tab.image(result["board"], caption="Rectified board with the read tiles (red = unsure)", width="stretch")
        if unsure:
            st.markdown(f"**Unsure tiles** (below {UNSURE:.0%})")
            st.html('<div class="chips">' + "".join(f"<span><b>{html.escape(s)}</b> at {cell} · {p:.0%}</span>"
                                                    for cell, s, p in unsure) + "</div>")
        with st.expander("Board as text", icon=":material/notes:"):
            st.code("\n".join(" ".join(row) for row in labels), language=None)


def play_game(photos, reading, players):
    """One step per photo: who played, the move (or why it was refused), board and scores after it."""
    forced = st.session_state.setdefault("forced", set())     # photos accepted despite a refused move
    game, steps = Game(players), []
    progress = st.progress(0.0, "Reading the photos...")
    for i, (name, data) in enumerate(photos):
        result = analyse(name, data, reading)
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


def move_panel(step, k, steps, colours):
    """Who played the move and how it was scored, or why it was refused."""
    move, player = step["move"], step["player"]
    if step["error"]:
        pill = '<span class="pill bad">Not counted</span>'
    elif step["forced"]:
        pill = '<span class="pill warn">Accepted for 0</span>'
    elif move is None:
        pill = '<span class="pill pass">Pass / exchange</span>'
    else:
        pill = f'<span class="pill ok">+{move.score} points</span>'
    st.html(f'<div class="move-head">{avatar(player, colours[player], "avatar sm")}<div>'
            f'<div class="kicker">Move {k + 1} of {len(steps)} <span class="file">· {html.escape(step["name"])}</span></div>'
            f'<div class="who">{html.escape(player)}</div></div>{pill}</div>')

    forced = st.session_state["forced"]
    if step["error"]:
        st.error(f"**This board can't be a legal move:** {step['error']}.\n\nThe turn stays with {player}. "
                 "Tiles read in the photo but not counted have a dashed red frame on the board.",
                 icon=":material/block:")
        if st.button("Accept this board anyway (0 points)", type="primary", icon=":material/check:",
                     help="Put its tiles on the board for no points, so the rest of the game still counts."):
            forced.add(step["key"])
            st.rerun()
    elif step["forced"]:
        st.warning(f"Accepted anyway for 0 points. It was refused because: {step['forced']}.",
                   icon=":material/warning:")
        if st.button("Undo accepting this board", icon=":material/undo:"):
            forced.discard(step["key"])
            st.rerun()
    elif move is None:
        st.info("No new tiles on the board: a pass or an exchange. The turn moves on.", icon=":material/skip_next:")
    else:
        st.html(score_html(player, move))
    if step["misread"]:
        st.warning("These old tiles were read differently in this photo; the game kept the old ones: "
                   + ", ".join(step["misread"]), icon=":material/visibility:")


def moves_table(steps, players, k):
    """All moves with the running totals; the current one highlighted. Picking a row jumps to it."""
    rows = []
    for i, s in enumerate(steps):
        m = s["move"]
        status = ("⛔ not counted" if s["error"] else "⚠️ accepted for 0" if s["forced"]
                  else "✅ scored" if m else "⏭️ pass")
        words = (s["error"] if s["error"] else s["forced"] if s["forced"] else describe_words(m) if m else "")
        rows.append({"#": i + 1, "Player": s["player"], "Status": status, "Words": words,
                     "Points": m.score if m else 0, **{p: s["scores"][p] for p in players}, "Photo": s["name"]})
    table = pd.DataFrame(rows).style.apply(
        lambda row: ["background-color: #fff1c2" if row.name == k else ""] * len(row), axis=1)
    st.dataframe(table, hide_index=True, width="stretch", height=35 * (min(len(rows), 20) + 1) + 3,
                 key="moves_table", on_select="rerun",
                 selection_mode="single-row", column_config={
                     "#": st.column_config.NumberColumn(width=40),
                     "Words": st.column_config.TextColumn(width="large"),
                     "Points": st.column_config.NumberColumn(format="%d"),
                 })


def jump_to_picked_row():
    """A row picked in the moves table moves the stepper there (once per pick)."""
    picked = st.session_state.get("moves_table")
    rows = list(picked.selection.rows) if picked else []
    if rows and rows != st.session_state.get("moves_table_applied"):
        st.session_state["move"] = rows[0]
    st.session_state["moves_table_applied"] = rows


def game_view(photos, reading, players):
    colours = {p: PLAYER_COLOURS[i % len(PLAYER_COLOURS)] for i, p in enumerate(players)}
    steps = play_game(photos, reading, players)

    jump_to_picked_row()
    scores = st.empty()                            # filled once the stepper says which move is shown
    k = stepper([f"Move {i + 1} · {s['name']}" for i, s in enumerate(steps)], "move")
    step = steps[k]
    before = steps[k - 1]["scores"] if k else {p: 0 for p in players}
    played = step["player"] if step["move"] else None
    scores.html(players_html(players, colours, step["scores"], before, played))
    move = step["move"]

    board_col, score_col = st.columns([11, 9], gap="large")
    with score_col:
        move_panel(step, k, steps, colours)
    with board_col:
        board_tab, photo_tab, top_tab = st.tabs([":material/grid_view: Board", ":material/photo: Photo",
                                                 ":material/grid_on: Top-down view"])
        new = {(r, c) for r, c, _ in move.tiles} if move else set()
        words = {l.cell for w in move.words for l in w.letters} if move else set()
        misread = {(ROW_NAMES.index(m[0]), int(m[1:]) - 1) for m in step["misread"]}
        ghosts = {}
        if step["error"] and step["result"] is not None:
            labels = step["result"]["labels"]
            ghosts = {(r, c): labels[r, c] for r, c in zip(*np.nonzero((labels != EMPTY) & (step["board"] == EMPTY)))}
        states = [s for s, cells in (("new", new), ("word", words - new), ("misread", misread), ("ghost", ghosts))
                  if cells]
        board_tab.html(board_html(step["board"], new=new, words=words, misread=misread, ghosts=ghosts)
                       + legend_html(*states))
        if step["result"] is None:
            photo_tab.error(f"Can't open {step['name']}")
        else:
            photo_tab.image(step["result"]["photo"], caption="Detected corners and the grid", width="stretch")
            top_tab.image(step["result"]["board"], caption="Rectified board with the read tiles (red = unsure)",
                          width="stretch")

    st.subheader("All moves", divider="gray")
    st.caption("Click a row's checkbox to jump to that move.")
    moves_table(steps, players, k)


def main():
    st.set_page_config(page_title="SCRABBLE", page_icon=":material/grid_view:", layout="wide")
    st.html(CSS)
    brand = "".join(f"<span>{ch}<sub>{VALUES[ch]}</sub></span>" for ch in "SCRABBLE")
    st.sidebar.html(f'<div class="brand">{brand}</div><div class="tagline">Live game analysis from photos</div>')
    views = {GAME: ":material/scoreboard:", PHOTO: ":material/photo_camera:"}
    view = st.sidebar.segmented_control("View", list(views), default=GAME, required=True, key="view",
                                        format_func=lambda v: f"{views[v]} {v}", width="stretch")

    corner_models = sorted(Path("models").glob("corners*.pt"))
    if not corner_models or not MODEL_PATH.exists():
        st.error("**Models missing.** Train them first:\n\n"
                 "```\npython -m scrabble.corners train\npython -m scrabble.letters train\n```",
                 icon=":material/error:")
        return
    photos = choose_photos(allow_camera=view == PHOTO)
    players = choose_players() if view == GAME else None
    with st.sidebar.expander("Settings", icon=":material/tune:"):
        names = [p.name for p in corner_models]
        default = names.index(DEFAULT_CORNER_MODEL) if DEFAULT_CORNER_MODEL in names else 0
        corner_path = st.selectbox("Corner model", corner_models, index=default, format_func=lambda p: p.name,
                                   help="Network that finds the 4 board corners (models/corners*.pt).")
        fit_grid = st.toggle("Fit the grid to the board lines", value=True,
                             help="Move the grid found from the corners onto the board's printed lines, so that "
                                  "corners a few pixels off don't shift the cells near the edges.")
    reading = (str(corner_path), fit_grid)
    st.sidebar.caption("Keyboard: ← → one step, Home / End first and last.")

    if not photos:
        st.info("**No photos yet.** Pick a folder or upload photos in the sidebar"
                + (" (or take one with the camera)." if view == PHOTO else ": one photo of the board after "
                   "each move, sorted by name."), icon=":material/arrow_back:")
        return
    if view == PHOTO:
        photos_view(photos, reading)
    else:
        game_view(photos, reading, players)


main()
