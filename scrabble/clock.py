"""Game clock in the browser: each player's time counts down while it is their turn.

Usage (from the repository root):
  ./start clock [--players NAME ...] [--minutes N] [--port N] [--photos DIR] [--camera SOURCE]
  python -m scrabble.clock

Enter or Space (or a click on the clock) ends the turn and the time moves to the next player;
P pauses, R resets (while paused). Time over the limit is shown in red as minus time.
Every end of a turn is sent to the server, which prints who played and how long it took.
With --photos, the server also takes the next photo from DIR that it has not used yet (in name
order, 2 before 10), reads the board in it and scores the move for the player who pressed the
clock; the result and the scores are shown on the page. A sample game replays as it was played,
and photos copied into DIR during a game are taken in turn.
With --camera (a USB camera's number or a phone's stream URL, see scrabble.camera), every end of a
turn takes a photo with the camera instead and scores it. Each game (until a reset) gets a new
folder DIR/<date and time>/ (DIR defaults to dataset/camera) with the photos 001.jpg, 002.jpg, ...
and game.json, the record of who played each photo, which the test app replays.
"""

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from scrabble.game import Game, MoveError, describe, natural_key, save_record

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
  .name { font-size: 4vmin; }
  .time { font-size: 16vmin; font-variant-numeric: tabular-nums; }
  .time.over { color: var(--over); }
  .score { font-size: 6vmin; color: var(--dim); }
  .active .score { color: #cfe8d0; }
  .active .time.over { color: #fff; text-shadow: 0 0 12px var(--over); }
  #last { height: 3em; display: flex; align-items: center; justify-content: center; padding: 0 1em;
          text-align: center; overflow: hidden; }
  #status { height: 3em; display: flex; align-items: center; justify-content: center; color: var(--dim); }
</style>
</head>
<body>
<div id="players"></div>
<div id="last"></div>
<div id="status"></div>
<script>
const PLAYERS = __PLAYERS__;
const LIMIT = __LIMIT__ * 1000;           // ms per player
const SCORING = __SCORING__;              // the server reads and scores the board after each turn
let left, turn, running, since;           // since: when the current turn's time last resumed
let scores, last, reading = 0;            // reading: turns sent but not answered yet
let queue = Promise.resolve();            // turns go to the server one at a time, in order

function send(path, data) {
  reading++; render();
  queue = queue
    .then(() => fetch(path, {method: "POST", headers: {"Content-Type": "application/json"},
                             body: JSON.stringify(data)}))
    .then(r => r.status === 200 ? r.json() : null)
    .then(result => { if (result) { scores = result.scores; last = result.text; } })
    .catch(() => { last = "The server did not answer"; })
    .finally(() => { reading--; render(); });
}

function reset() {
  left = PLAYERS.map(() => LIMIT);
  turn = 0; running = false; since = null;
  scores = SCORING ? PLAYERS.map(() => 0) : null; last = "";
  if (SCORING) send("reset", {});
  render();
}

function now() { return performance.now(); }

function timeLeft(i) {
  return i === turn && running ? left[i] - (now() - since) : left[i];
}

function format(ms) {
  const s = Math.ceil(Math.abs(ms) / 1000);
  const text = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  return ms < 0 ? "-" + text : text;
}

function stop() {
  left[turn] = timeLeft(turn);
  running = false;
}

function endTurn() {
  if (!running) {                         // the first press starts the game, or resumes it
    running = true; since = now();
    render(); return;
  }
  const used = now() - since;
  stop();
  send("turn", {turn: turn, player: PLAYERS[turn], used: used / 1000, left: left[turn] / 1000});
  turn = (turn + 1) % PLAYERS.length;
  running = true; since = now();
  render();
}

function pause() {
  if (running) stop(); else if (since !== null) { running = true; since = now(); }
  render();
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
    const t = div.querySelector(".time");
    t.textContent = format(ms);
    t.classList.toggle("over", ms < 0);
    div.querySelector(".score").textContent = scores ? scores[i] + " pts" : "";
  });
  document.getElementById("last").textContent = reading ? "Reading the board…" : last;
  document.getElementById("status").textContent =
    since === null ? "Enter / Space: start  ·  " + PLAYERS[0] + " begins"
    : running ? "Enter / Space: end turn  ·  P: pause"
    : "Paused  ·  P: resume  ·  R: reset";
}

document.addEventListener("keydown", e => {
  if (e.repeat) return;
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); endTurn(); }
  else if (e.key === "p" || e.key === "P") pause();
  else if ((e.key === "r" || e.key === "R") && !running) reset();
});
setInterval(() => { if (running) render(); }, 100);
reset();
</script>
</body>
</html>
"""


def page(players, minutes, scoring=False):
    # The names go into a <script>: escape "<" so a name can't close the tag.
    names = json.dumps(list(players)).replace("<", "\\u003c")
    return (PAGE.replace("__PLAYERS__", names).replace("__LIMIT__", str(round(minutes * 60)))
            .replace("__SCORING__", json.dumps(scoring)))


def board_reader(corner_path=Path("models/corners.pt"), fit=True):
    """A function photo path -> read board, with the models loaded once."""
    import cv2
    from scrabble.corners import load_model as load_corner_model, predict_corners
    from scrabble.letters import load_model, read_board

    corner_model, model = load_corner_model(corner_path), load_model()

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
    with a `camera`, saves the camera's current frame in this game's folder (a new folder in
    `photos`, made at the game's first photo) and records the move there (game.save_record).
    A refused move is skipped and its photo stays used, as in `python -m scrabble.game`; with
    `accept_refused` its tiles go on the board for 0 points instead."""

    def __init__(self, players, photos, read, accept_refused=False, camera=None):
        self.players, self.photos, self.read = list(players), Path(photos), read
        self.accept_refused, self.camera = accept_refused, camera
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.game = Game(self.players)
        self.used = set()
        self.folder, self.moves = None, []      # with a camera: this game's folder and record

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
        from scrabble.letters import IMAGE_EXTS
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

    def record(self, path, index, forced=False):
        if self.camera is not None:
            self.moves.append({"photo": path.name, "player": self.players[index], "forced": forced})
            save_record(self.folder, self.players, self.moves)

    def turn(self, index):
        """Score the turn of player `index`; returns the text to show."""
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
            self.record(path, index)
            return f"{path.name}  {self.players[index]}: {e}"
        self.game.turn = index          # the clock, not the order of the photos, says who played
        try:
            text = describe(*self.game.play(board))
            self.record(path, index)
        except MoveError as e:
            self.record(path, index, forced=self.accept_refused)
            if not self.accept_refused:
                return f"{path.name}  {self.players[index]}: move refused ({e})"
            self.game.play(board, force=True)
            text = f"{self.players[index]}: accepted anyway for 0 points ({e})"
        if self.game.misread:
            text += f"; kept the old tiles at {', '.join(self.game.misread)}"
        if self.game.ignored:
            text += f"; ignored the tiles at {', '.join(self.game.ignored)}"
        return f"{path.name}  {text}"

    def result(self, text):
        return {"text": text, "scores": [self.game.scores[p] for p in self.players]}


def on_turn_end(player, used, left):
    """Called when a player ends their turn."""
    minus = "-" if left < 0 else ""
    left = abs(left)
    print(f"{player}: turn took {used:.1f} s, {minus}{int(left // 60)}:{int(left % 60):02d} left", flush=True)


def make_server(players, minutes, port=0, host="127.0.0.1", scorer=None):
    """`scorer`: a Scorer that reads and scores the board after each turn, or None for the clock alone."""
    body = page(players, minutes, scorer is not None).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path not in ("/turn", "/reset"):
                self.send_error(404)
                return
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/turn":
                    player, used, left = str(data["player"]), float(data["used"]), float(data["left"])
                    index = int(data.get("turn", players.index(player) if player in players else -1))
                    if not 0 <= index < len(players):
                        raise ValueError("no such player")
            except (ValueError, KeyError, TypeError, AttributeError):
                self.send_error(400)
                return
            if self.path == "/turn":
                on_turn_end(player, used, left)
            if scorer is None:
                self.send_response(204)
                self.end_headers()
                return
            with scorer.lock:
                if self.path == "/reset":
                    scorer.reset()
                    text = ""
                else:
                    text = scorer.turn(index)
                    print(f"  {text}", flush=True)
                reply = json.dumps(scorer.result(text)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass                        # keep the terminal for the turn log

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--players", nargs="+", default=["Player 1", "Player 2"])
    parser.add_argument("--minutes", type=float, default=25, help="time per player")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--photos", type=Path, help="read and score the board from the photos in this folder")
    parser.add_argument("--camera", metavar="SOURCE",
                        help="take the photos with this camera (number or stream URL); each game goes into "
                             "a new folder in --photos (default dataset/camera)")
    parser.add_argument("--corner-model", type=Path, default=Path("models/corners.pt"))
    parser.add_argument("--accept-refused", action="store_true",
                        help="put the tiles of a refused move on the board for 0 points instead of skipping the photo")
    parser.add_argument("--no-fit", action="store_true", help="don't fit the grid to the board's printed lines")
    args = parser.parse_args()
    if len(args.players) < 2:
        parser.error("at least 2 players")
    scorer = camera = None
    if args.camera is not None:
        from scrabble.camera import Camera
        args.photos = args.photos or Path("dataset/camera")
        args.photos.mkdir(parents=True, exist_ok=True)
        try:
            camera = Camera(args.camera)
        except ValueError as e:
            parser.error(str(e))
        print(f"Camera {args.camera}: each game goes into a new folder in {args.photos}", flush=True)
    if args.photos:
        if not args.photos.is_dir():
            parser.error(f"no such folder: {args.photos}")
        print("Loading the models...", flush=True)
        scorer = Scorer(args.players, args.photos, board_reader(args.corner_model, fit=not args.no_fit),
                        args.accept_refused, camera)
    server = make_server(args.players, args.minutes, args.port, scorer=scorer)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"Clock for {', '.join(args.players)} ({args.minutes:g} min each) at {url}  (Ctrl+C to quit)", flush=True)
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if camera:
            camera.close()


if __name__ == "__main__":
    main()
