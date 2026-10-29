"""Game clock in the browser: each player's time counts down while it is their turn.

Usage (from the repository root):
  ./start clock [--players NAME ...] [--minutes N] [--port N] [--games DIR | --photos DIR | --clock-only]
  python -m scrabble.clock

Enter or Space (or a click on the clock) ends the turn and the time moves to the next player;
X ends it as an exchange (a turn without new tiles is a pause otherwise: only pauses count towards
the end of the game), P pauses, R resets (while paused). The rules are PFS's: a player whose time
runs out makes no more moves and the others play on (6.2.1). The game ends when the bag is empty
and a player has played their last tile, when every player still playing paused twice in a row, or
when everyone's time has run out (see scrabble.game); the clock then stops, asks for the points
left on each rack (filled in where the tiles played tell them) and shows the final scores.
Every end of a turn is sent to the server, which prints who played and how long it took.

A game has three pages, all linked from the clock:
  /          the clock
  /camera    the camera page (scrabble.camera_page): the live view of the camera of the device it is
             open on, with the grid found in it; every end of a turn takes a photo with it. A phone
             on the same Wi-Fi opens it at https://<this computer's IP>:8643/camera (a browser gives
             a page the camera only over https; the certificate is self-signed, so the phone asks
             once whether to trust it)
  preview    the test app (scrabble/app.py, started with the clock) following this game
Each photo is read and the move scored for the player who pressed the clock; the clock shows the
result and the scores. Each game (until a reset) gets a new folder DIR/<date and time>/ (DIR is
--games, default dataset/camera) with the photos 001.jpg, 002.jpg, ... and game.json, the record
of who played each photo, which the preview replays.
With --photos DIR, the server instead takes the next photo from DIR that it has not used yet (in
name order, 2 before 10), so a sample game replays as it was played, and photos copied into DIR
during a game are taken in turn. --clock-only runs the clock alone.
"""

import argparse
import html
import json
import socket
import ssl
import subprocess
import sys
import threading
import webbrowser
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from scrabble.game import OUT, PASSES, TIME, Game, MoveError, describe, end_text, final_scores, natural_key, save_record

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scrabble clock</title>
<style>
  :root { --bg: #111; --panel: #222; --active: #2e7d32; --text: #eee; --dim: #888; --over: #e53935; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text);
               font-family: system-ui, sans-serif; user-select: none; }
  #players { display: flex; height: calc(100% - 6em); gap: 8px; padding: 8px; box-sizing: border-box; }
  .player { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center;
            background: var(--panel); border-radius: 16px; cursor: pointer; min-width: 0; }
  .player.active { background: var(--active); }
  .player.out { opacity: .4; cursor: default; }
  .name { font-size: 4vmin; }
  .time { font-size: 16vmin; font-variant-numeric: tabular-nums; }
  .time.over { color: var(--over); }
  .score { font-size: 6vmin; color: var(--dim); }
  .active .score { color: #cfe8d0; }
  .active .time.over { color: #fff; text-shadow: 0 0 12px var(--over); }
  #last { height: 3em; display: flex; align-items: center; justify-content: center; padding: 0 1em;
          text-align: center; overflow: hidden; }
  #status { height: 3em; display: flex; align-items: center; justify-content: center; gap: 1.5em; color: var(--dim); }
  #status a { color: var(--dim); }
  #end { position: fixed; inset: 0; display: flex; align-items: center; justify-content: center;
         background: rgba(0, 0, 0, .75); padding: 16px; }
  #end[hidden] { display: none; }
  #end .box { background: var(--panel); border-radius: 16px; padding: 1.5em 2em; max-width: 32em; width: 100%;
              box-sizing: border-box; user-select: text; }
  #end h2 { margin: 0 0 .3em; font-size: 2em; }
  #end p { color: var(--dim); }
  #end label { display: flex; justify-content: space-between; align-items: center; gap: 1em; margin: .5em 0; }
  #end input { width: 5em; font: inherit; padding: .3em; border-radius: 6px; border: 1px solid #555;
               background: #111; color: var(--text); text-align: right; }
  #end button { font: inherit; padding: .5em 1.2em; border-radius: 8px; border: 0; background: var(--active);
                color: #fff; cursor: pointer; margin-top: .5em; }
  #end .final { font-size: 1.4em; margin: .3em 0; display: flex; justify-content: space-between; }
</style>
</head>
<body>
<div id="players"></div>
<div id="last"></div>
<div id="status"><span id="keys"></span>__LINKS__</div>
<div id="end" hidden><div class="box">
  <h2>Game over</h2>
  <p id="why"></p>
  <form id="racks"></form>
  <div id="final"></div>
</div></div>
<script>
const PLAYERS = __PLAYERS__;
const LIMIT = __LIMIT__ * 1000;           // ms per player
const SCORING = __SCORING__;              // the server reads and scores the board after each turn
let left, turn, running, since;           // since: when the current turn's time last resumed
let scores, last, bag, reading = 0;       // reading: turns sent but not answered yet; bag: tiles in it
let over = null;                          // why the game ended (the server's end_info), once it has
let out;                                  // out[i]: player i's time ran out, so they make no more moves
// Turns go to the server at once, so the camera takes each photo when the clock is pressed; the
// server scores them in order, and an answer older than the one shown, or from before a reset, is dropped.
let sent = 0, shown = 0, epoch = 0;

function send(path, data, then) {
  const n = ++sent, e = epoch;
  reading++; render();
  fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)})
    .then(r => r.status === 200 ? r.json() : null)
    .then(result => {
      if (!result || e !== epoch || n <= shown) return;
      shown = n; scores = result.scores; last = result.text; bag = result.bag;
      if (result.end && !over) gameOver(result.end);
      if (then) then(result);
    })
    .catch(() => { if (e === epoch) last = "The server did not answer"; })
    .finally(() => { reading--; render(); });
}

function reset() {
  left = PLAYERS.map(() => LIMIT);
  turn = 0; running = false; since = null;
  scores = SCORING ? PLAYERS.map(() => 0) : null; last = ""; bag = null; over = null;
  out = PLAYERS.map(() => false);
  document.getElementById("end").hidden = true;
  epoch++;
  if (SCORING) send("reset", {});
  render();
}

function now() { return performance.now(); }

function timeLeft(i) {
  return Math.max(0, i === turn && running ? left[i] - (now() - since) : left[i]);
}

function format(ms) {
  const s = Math.ceil(ms / 1000);
  return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
}

function stop() {
  left[turn] = timeLeft(turn);
  running = false;
}

function next(i) {                       // the next player who still has time
  for (let k = 1; k <= PLAYERS.length; k++) if (!out[(i + k) % PLAYERS.length]) return (i + k) % PLAYERS.length;
  return i;
}

function endTurn(exchange) {
  if (over) return;
  if (!running) {                         // the first press starts the game, or resumes it
    running = true; since = now();
    render(); return;
  }
  const used = now() - since;
  stop();
  send("turn", {turn: turn, player: PLAYERS[turn], used: used / 1000, left: left[turn] / 1000,
                exchange: exchange === true});
  turn = next(turn);
  running = true; since = now();
  render();
}

function pause() {
  if (over) return;
  if (running) stop(); else if (since !== null) { running = true; since = now(); }
  render();
}

// PFS 6.2: a player whose time runs out makes no more moves (a move in progress is taken back)
// and the others play on; the game ends when everyone's time has run out.
function timeUp() {
  stop();
  out[turn] = true;
  if (SCORING) send("timeout", {turn: turn, player: PLAYERS[turn]});   // the last one's answer ends the game
  if (out.every(o => o)) {
    if (!SCORING) gameOver({reason: "time", player: null, text: "every player's time ran out"});
    render(); return;
  }
  if (!SCORING) last = PLAYERS[turn] + "'s time ran out: no more moves for them";
  turn = next(turn);
  running = true; since = now();
}

// The end of the game: the clock stops, and a form asks for the points of the tiles left on each
// rack (filled in where the tiles played tell them); the server then reckons the final scores.
function gameOver(info) {
  if (running) stop();
  over = info;
  document.getElementById("why").textContent = info.text[0].toUpperCase() + info.text.slice(1) + ".";
  const form = document.getElementById("racks"), final = document.getElementById("final");
  form.innerHTML = ""; final.innerHTML = "";
  if (SCORING) {
    PLAYERS.forEach((name, i) => {
      const label = document.createElement("label");
      const tiles = info.bag === 0 ? " (" + info.racks[i] + (info.racks[i] === 1 ? " tile)" : " tiles)") : "";
      label.textContent = name + tiles;
      const input = document.createElement("input");
      input.type = "number"; input.min = 0; input.placeholder = "0";
      const known = info.known[i];
      if (known !== null) input.value = known;
      if (info.reason === "out" && name === info.player) { input.value = 0; input.disabled = true; }
      label.appendChild(input);
      form.appendChild(label);
    });
    const hint = document.createElement("p");
    hint.textContent = "Points of the tiles left on each rack" + (info.bag === 0
      ? " (all the tiles not on the board, worth " + info.left + " points together)."
      : "; " + info.bag + " tiles are still in the bag.")
      + (info.reason === "out" ? " " + info.player + " gets the others' points." : "");
    form.prepend(hint);
    const button = document.createElement("button");
    button.textContent = "Final score";
    form.appendChild(button);
    form.onsubmit = e => {
      e.preventDefault();
      const racks = [...form.querySelectorAll("input")].map(input => Math.max(0, parseInt(input.value) || 0));
      send("end", {reason: info.reason, player: info.player, left: racks}, () => showFinal(form, final));
    };
    setTimeout(() => { const first = form.querySelector("input:not([disabled])"); if (first) first.focus(); }, 0);
  } else {
    showFinal(form, final);
  }
  document.getElementById("end").hidden = false;
  render();
}

function showFinal(form, final) {
  form.innerHTML = "";
  if (scores) {
    const best = Math.max(...scores);
    PLAYERS.forEach((name, i) => {
      const row = document.createElement("div");
      row.className = "final";
      row.innerHTML = "<span></span><b></b>";
      row.firstChild.textContent = (scores[i] === best ? "★ " : "") + name;
      row.lastChild.textContent = scores[i] + " pts";
      final.appendChild(row);
    });
  }
  const p = document.createElement("p");
  p.textContent = "R: new game";
  final.appendChild(p);
}

function render() {
  const box = document.getElementById("players");
  if (!box.children.length) {
    PLAYERS.forEach(name => {
      const div = document.createElement("div");
      div.className = "player";
      div.innerHTML = '<div class="name"></div><div class="time"></div><div class="score"></div>';
      div.querySelector(".name").textContent = name;
      div.onclick = endTurn;
      box.appendChild(div);
    });
  }
  [...box.children].forEach((div, i) => {
    const ms = timeLeft(i);
    div.classList.toggle("active", i === turn && running);
    div.classList.toggle("out", out[i]);
    const t = div.querySelector(".time");
    t.textContent = format(ms);
    t.classList.toggle("over", ms <= 0);
    div.querySelector(".score").textContent = scores ? scores[i] + " pts" : "";
  });
  document.getElementById("last").textContent = reading ? "Reading the board…" : last;
  document.getElementById("keys").textContent =
    (over ? "Game over  ·  R: new game"
     : since === null ? "Enter / Space: start  ·  " + PLAYERS[0] + " begins"
     : running ? "Enter / Space: end turn  ·  X: end turn with an exchange  ·  P: pause"
     : "Paused  ·  P: resume  ·  R: reset")
    + (bag !== null && bag !== undefined ? "  ·  Bag: " + bag + " tiles" : "");
}

document.addEventListener("keydown", e => {
  if (e.repeat || e.target.tagName === "INPUT") return;      // typing the racks' points
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); endTurn(); }
  else if ((e.key === "x" || e.key === "X") && running) endTurn(true);
  else if (e.key === "p" || e.key === "P") pause();
  else if ((e.key === "r" || e.key === "R") && !running) reset();
});
setInterval(() => { if (running) { if (timeLeft(turn) <= 0) timeUp(); render(); } }, 100);
reset();
</script>
</body>
</html>
"""


def page(players, minutes, scoring=False, links=()):
    """`links`: (text, URL) pairs shown under the clock, opened in a new tab."""
    # The names go into a <script>: escape "<" so a name can't close the tag.
    names = json.dumps(list(players)).replace("<", "\\u003c")
    links = "".join(f'<a href="{html.escape(url)}" target="_blank">{html.escape(text)}</a>' for text, url in links)
    return (PAGE.replace("__PLAYERS__", names).replace("__LIMIT__", str(round(minutes * 60)))
            .replace("__SCORING__", json.dumps(scoring)).replace("__LINKS__", links))


def board_reader(corner_model, fit=True):
    """A function photo path -> read board, with the letter model loaded once."""
    import cv2
    from scrabble.corners import predict_corners
    from scrabble.letters import load_model, read_board

    model = load_model()

    def read(path):
        image = cv2.imread(str(path))      # applies EXIF orientation, like the labels
        if image is None:
            raise ValueError(f"can't read {path.name}")
        corners, _ = predict_corners(corner_model, image, fit=fit)
        return read_board(model, image, corners)[0]

    return read


class Scorer:
    """Reads the board after every turn and scores the move for the player who pressed the clock.

    Each turn takes the next photo in `photos` (in name order) that has not been used yet or,
    with a `camera` (anything with `frame()`, e.g. a CameraPage), saves the camera's frame in this
    game's folder (a new folder in `photos`, made when the game starts) and records the move there
    (game.save_record). A refused move is skipped and its photo stays used, as in
    `python -m scrabble.game`; with `accept_refused` its tiles go on the board for 0 points instead.
    Turns and resets coming from several requests at once are run in order with `in_order`."""

    def __init__(self, players, photos, read, accept_refused=False, camera=None):
        self.players, self.photos, self.read = list(players), Path(photos), read
        self.accept_refused, self.camera = accept_refused, camera
        self.order = threading.Condition()
        self.issued = self.served = 0               # tickets given out, and turns or resets run
        self.folder = None
        self.reset()

    def reset(self):
        self.close()
        self.game = Game(self.players)
        self.used = set()
        self.folder, self.moves = None, []      # with a camera: this game's folder and record
        self.final = None                       # scores after the end of the game's reckoning
        if self.camera is not None:             # the preview shows the new game at once
            save_record(self.game_folder(), self.players, self.moves)

    def close(self):
        """Remove this game's folder if no photo was taken in it."""
        from scrabble.board import IMAGE_EXTS
        if self.folder is None or any(p.suffix.lower() in IMAGE_EXTS for p in self.folder.iterdir()):
            return
        for p in self.folder.iterdir():
            p.unlink()
        self.folder.rmdir()

    @contextmanager
    def in_order(self, photo=False):
        """Run a turn or a reset after the ones that came before it. With `photo`, the camera is
        asked for this turn's photo right away (if it can be asked: CameraPage.request), not when
        the turns before it are scored."""
        with self.order:
            ticket = self.issued
            self.issued += 1
            if photo and hasattr(self.camera, "request"):
                self.camera.request()
            self.order.wait_for(lambda: self.served == ticket)
        try:
            yield
        finally:
            with self.order:
                self.served += 1
                self.order.notify_all()

    def game_folder(self):
        """This game's folder for the camera's photos, made when it is first needed."""
        if self.folder is None:
            from datetime import datetime
            name = f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
            folder, n = self.photos / name, 1
            while folder.exists():
                n += 1
                folder = self.photos / f"{name}_{n}"
            folder.mkdir(parents=True)
            self.folder = folder
        return self.folder

    def next_photo(self):
        from scrabble.board import IMAGE_EXTS
        new = [p for p in self.photos.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.name not in self.used]
        return min(new, key=natural_key, default=None)

    def take_photo(self):
        """This turn's photo: a new frame from the camera, or the next unused photo in the folder."""
        if self.camera is None:
            return self.next_photo()
        import cv2
        from scrabble.camera import next_path
        frame = self.camera.frame()
        path = next_path(self.game_folder())
        cv2.imwrite(str(path), frame)
        return path

    def record(self, path, index, forced=False, exchange=False):
        if self.camera is not None:
            self.moves.append({"photo": path.name, "player": self.players[index], "forced": forced,
                               "exchange": exchange})
            save_record(self.folder, self.players, self.moves)

    def turn(self, index, exchange=False):
        """Score the turn of player `index` (`exchange`: they marked it as an exchange on the clock);
        returns the text to show."""
        try:
            path = self.take_photo()
        except ValueError as e:
            return f"{self.players[index]}: {e}"
        if path is None:
            return f"{self.players[index]}: no new photo in {self.photos}"
        self.used.add(path.name)
        try:
            board = self.read(path)
        except ValueError as e:
            self.record(path, index, exchange=exchange)
            return f"{path.name}  {self.players[index]}: {e}"
        self.game.turn = index          # the clock, not the order of the photos, says who played
        try:
            text = describe(*self.game.play(board, exchange=exchange), exchange=exchange)
            self.record(path, index, exchange=exchange)
        except MoveError as e:
            self.record(path, index, forced=self.accept_refused, exchange=exchange)
            if not self.accept_refused:
                return f"{path.name}  {self.players[index]}: move refused ({e})"
            self.game.play(board, force=True, exchange=exchange)
            text = f"{self.players[index]}: accepted anyway for 0 points ({e})"
        if self.game.misread:
            text += f"; kept the old tiles at {', '.join(self.game.misread)}"
        if self.game.ignored:
            text += f"; ignored the tiles at {', '.join(self.game.ignored)}"
        return f"{path.name}  {text}"

    def time_out(self, index):
        """Player `index`'s time ran out (PFS 6.2.1); returns the text to show."""
        player = self.players[index]
        self.game.time_out(player)
        playing = [p for p in self.players if p not in self.game.timed_out]
        if not playing:
            return f"{player}'s time ran out too"
        return f"{player}'s time ran out: no more moves for them; {', '.join(playing)} play on"

    def end_info(self, reason, player=None):
        """What the clock's end-of-game form needs: why the game ended, how many tiles each rack
        holds and what the tiles not on the board are worth, and the points left on each rack
        where the tiles played tell them (else None)."""
        game = self.game
        racks = [game.racks[p] for p in self.players]
        known = [None] * len(self.players)
        if game.bag == 0:                       # then the racks hold all the tiles not on the board
            holders = [i for i, n in enumerate(racks) if n > 0]
            for i, n in enumerate(racks):
                if n == 0:
                    known[i] = 0
            if len(holders) == 1:
                known[holders[0]] = game.left_value()
        return {"reason": reason, "player": player, "text": end_text(reason, player), "racks": racks,
                "bag": game.bag, "left": game.left_value(), "known": known}

    def finish(self, reason, player, left):
        """End the game: take the points `left` on each rack (a list in player order) off the scores
        and give them to the player who played out; returns the text to show."""
        left = dict(zip(self.players, left))
        self.final = final_scores(self.game.scores, left, player if reason == OUT else None)
        if self.camera is not None:
            save_record(self.game_folder(), self.players, self.moves,
                        {"reason": reason, "player": player, "left": left})
        return f"Game over: {end_text(reason, player)}"

    def result(self, text):
        scores = self.final or self.game.scores
        result = {"text": text, "scores": [scores[p] for p in self.players], "bag": self.game.bag}
        end = self.game.end()
        if end and self.final is None:
            result["end"] = self.end_info(*end)
        return result


def on_turn_end(player, used, left, exchange=False):
    """Called when a player ends their turn."""
    minus = "-" if left < 0 else ""
    left = abs(left)
    print(f"{player}: {'exchange, ' if exchange else ''}turn took {used:.1f} s, "
          f"{minus}{int(left // 60)}:{int(left % 60):02d} left", flush=True)


MAX_PHOTO = 50 * 2**20                  # bytes: bigger uploads are refused


def make_server(players, minutes, port=0, host="127.0.0.1", scorer=None, camera=None, find_grid=None,
                preview=None, phone=None):
    """`scorer`: a Scorer that reads and scores the board after each turn, or None for the clock alone.
    `camera`: a CameraPage served at /camera (it should be the scorer's camera too).
    `find_grid`: a function JPEG bytes -> grid for the camera page (camera_page.grid_finder).
    `preview`: the URL of the game's preview, `phone`: the camera page's URL for a phone; both linked
    from the clock."""
    from scrabble.camera_page import PAGE as CAMERA_PAGE
    links = ([("Camera", "camera")] if camera else []) + ([(f"Phone: {phone}", phone)] if phone else []) \
        + ([("Game preview", preview)] if preview else [])
    pages = {"/": page(players, minutes, scorer is not None, links).encode()}
    if camera:
        pages["/camera"] = CAMERA_PAGE.encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body=None, content_type="application/json"):
            if body is None:
                self.send_response(204)
                self.end_headers()
                return
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_PHOTO:
                raise ValueError("too big")
            return self.rfile.read(size)

        def do_GET(self):
            url = urlsplit(self.path)
            if url.path in pages:
                self.reply(pages[url.path], "text/html; charset=utf-8")
            elif camera and url.path == "/camera/next":
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    self.send_error(400)
                    return
                self.reply({"shot": camera.next_shot(after)})
            else:
                self.send_error(404)

        def do_POST(self):
            url = urlsplit(self.path)
            if camera and url.path in ("/camera/photo", "/camera/grid"):
                self.camera_post(url)
                return
            if url.path not in ("/turn", "/timeout", "/reset", "/end"):
                self.send_error(404)
                return
            try:
                data = json.loads(self.body())
                if url.path in ("/turn", "/timeout"):
                    player = str(data["player"])
                    index = int(data.get("turn", players.index(player) if player in players else -1))
                    if not 0 <= index < len(players):
                        raise ValueError("no such player")
                if url.path == "/turn":
                    used, left, exchange = float(data["used"]), float(data["left"]), bool(data.get("exchange"))
                elif url.path == "/end":
                    reason, player = str(data["reason"]), data.get("player")
                    if reason not in (OUT, PASSES, TIME) or (player is not None and player not in players):
                        raise ValueError("bad end")
                    racks = [max(int(v), 0) for v in data["left"]]
                    if len(racks) != len(players):
                        raise ValueError("one number per player")
            except (ValueError, KeyError, TypeError, AttributeError):
                self.send_error(400)
                return
            if url.path == "/turn":
                on_turn_end(player, used, left, exchange)
            elif url.path == "/timeout":
                print(f"{player}: time ran out", flush=True)
            if scorer is None:
                self.reply()
                return
            with scorer.in_order(photo=url.path == "/turn"):
                if url.path == "/reset":
                    scorer.reset()
                    text = ""
                elif url.path == "/timeout":
                    text = scorer.time_out(index)
                elif url.path == "/end":
                    text = scorer.finish(reason, player, racks)
                    print(f"  {text}: {', '.join(f'{p} {s}' for p, s in scorer.final.items())}", flush=True)
                else:
                    text = scorer.turn(index, exchange)
                    print(f"  {text}", flush=True)
                result = scorer.result(text)
            self.reply(result)

        def camera_post(self, url):
            try:
                data = self.body()
                if url.path == "/camera/photo":
                    shot = int(parse_qs(url.query)["shot"][0])
                    self.reply() if camera.deliver(shot, data) else self.send_error(409, "photo no longer wanted")
                elif find_grid is None:
                    self.send_error(404)
                else:
                    self.reply(find_grid(data))
            except (ValueError, KeyError):
                self.send_error(400)

        def log_message(self, *args):
            pass                        # keep the terminal for the turn log

    return ThreadingHTTPServer((host, port), Handler)


CERTS = Path(".cert")                   # self-signed certificates for the phone's https page


def lan_ip():
    """This computer's address in the local network (127.0.0.1 if it has none)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))      # no packet is sent; it only picks the interface
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def certificate(ip, folder=CERTS):
    """(certificate, key) files of a self-signed certificate for `ip`, made with openssl if missing."""
    cert, key = folder / f"{ip}.pem", folder / f"{ip}.key"
    if not (cert.is_file() and key.is_file()):
        folder.mkdir(exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
                        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=Scrabble clock",
                        "-addext", f"subjectAltName=IP:{ip},DNS:localhost"],
                       check=True, capture_output=True)
    return cert, key


def use_https(server, cert, key):
    """Make `server` speak https. The TLS handshake runs in each request's thread, so a client
    that stalls in it doesn't hold up the others."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
    return server


def start_preview(port):
    """Start the test app (scrabble/app.py) following the clock's game; returns the process and its URL."""
    root = Path(__file__).resolve().parent.parent
    process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "scrabble/app.py",
                                "--server.headless", "true", "--server.port", str(port)],
                               cwd=root, stdout=subprocess.DEVNULL)
    return process, f"http://localhost:{port}/?game=live"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", nargs="+", default=["Player 1", "Player 2"])
    parser.add_argument("--minutes", type=float, default=25, help="time per player")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--games", type=Path, default=Path("dataset/camera"),
                        help="folder for the camera page's games, one folder per game")
    parser.add_argument("--photos", type=Path,
                        help="read and score the board from the photos in this folder instead of the camera page")
    parser.add_argument("--clock-only", action="store_true", help="the clock alone: no photos, no scoring")
    parser.add_argument("--preview-port", type=int, default=8501, help="port of the game preview (the test app)")
    parser.add_argument("--no-preview", action="store_true", help="don't start the game preview")
    parser.add_argument("--phone-port", type=int, default=8643,
                        help="https port for the camera page on a phone in the same network")
    parser.add_argument("--no-phone", action="store_true", help="serve the camera page only on this computer")
    parser.add_argument("--corner-model", type=Path, default=Path("models/corners.pt"))
    parser.add_argument("--accept-refused", action="store_true",
                        help="put the tiles of a refused move on the board for 0 points instead of skipping the photo")
    parser.add_argument("--no-fit", action="store_true", help="don't fit the grid to the board's printed lines")
    args = parser.parse_args()
    if len(args.players) < 2:
        parser.error("at least 2 players")
    if args.photos and args.clock_only:
        parser.error("--photos and --clock-only don't go together")
    if args.photos and not args.photos.is_dir():
        parser.error(f"no such folder: {args.photos}")
    scorer = camera = find_grid = preview = None
    if not args.clock_only:
        from scrabble.corners import load_model as load_corner_model
        print("Loading the models...", flush=True)
        corner_model = load_corner_model(args.corner_model)
        read = board_reader(corner_model, fit=not args.no_fit)
        if args.photos:
            scorer = Scorer(args.players, args.photos, read, args.accept_refused)
        else:
            from scrabble.camera_page import CameraPage, grid_finder
            camera, find_grid = CameraPage(), grid_finder(corner_model, fit=not args.no_fit)
            args.games.mkdir(parents=True, exist_ok=True)
            scorer = Scorer(args.players, args.games, read, args.accept_refused, camera)
            print(f"Camera page: each game goes into a new folder in {args.games}", flush=True)
    if camera and not args.no_preview:
        preview, preview_url = start_preview(args.preview_port)
    phone = phone_url = None
    if camera and not args.no_phone:
        ip = lan_ip()
        phone_url = f"https://{ip}:{args.phone_port}/camera"
    options = dict(scorer=scorer, camera=camera, find_grid=find_grid, preview=preview_url if preview else None,
                   phone=phone_url)
    server = make_server(args.players, args.minutes, args.port, **options)
    if phone_url:
        # The same pages for the phone, over https and open to the local network.
        phone = use_https(make_server(args.players, args.minutes, args.phone_port, "0.0.0.0", **options),
                          *certificate(ip))
        threading.Thread(target=phone.serve_forever, daemon=True).start()
    url = f"http://localhost:{server.server_port}/"
    print(f"Clock for {', '.join(args.players)} ({args.minutes:g} min each) at {url}  (Ctrl+C to quit)", flush=True)
    if camera:
        print(f"Camera page at {url}camera", flush=True)
    if phone:
        print(f"Camera page for a phone on the same Wi-Fi at {phone_url} (trust the certificate once)", flush=True)
    if preview:
        print(f"Game preview at {preview_url}", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if phone:
            phone.shutdown()
            phone.server_close()
        if scorer:
            scorer.close()
        if preview:
            preview.terminate()


if __name__ == "__main__":
    main()
