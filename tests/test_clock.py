import json
import threading
import urllib.request

import numpy as np
import pytest

from scrabble import clock, game


@pytest.fixture
def server():
    server = clock.make_server(["Ann", "<Bob>"], 25)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


def test_page_has_players_and_limit(server):
    page = urllib.request.urlopen(server).read().decode()
    assert '["Ann", "\\u003cBob>"]' in page
    assert "const LIMIT = 1500 * 1000;" in page


def test_turn_end_is_reported(server, capsys):
    body = json.dumps({"player": "Ann", "used": 12.34, "left": -65}).encode()
    request = urllib.request.Request(server + "turn", body, {"Content-Type": "application/json"})
    assert urllib.request.urlopen(request).status == 204
    assert "Ann: turn took 12.3 s, -1:05 left" in capsys.readouterr().out


def test_bad_turn_is_refused(server):
    request = urllib.request.Request(server + "turn", b"{}", {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request)
    assert e.value.code == 400


def board_with(*tiles):
    board = game.empty_board()
    for r, c, letter in tiles:
        board[r, c] = letter
    return board


@pytest.fixture
def scored(tmp_path):
    """A clock with a scorer whose 'photos' are boards looked up by file name."""
    boards = {
        "1.jpg": board_with((7, 7, "A"), (7, 8, "S")),
        "2.jpg": board_with((7, 7, "A"), (7, 8, "S"), (8, 7, "B")),
        "10.jpg": board_with((7, 7, "A"), (7, 8, "S"), (8, 7, "B")),
    }
    for name in boards:
        (tmp_path / name).write_bytes(b"")
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: boards[path.name])
    server = clock.make_server(["Ann", "Bob"], 25, scorer=scorer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()
    server.server_close()


def post(url, data):
    request = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(request).read())


def test_turns_are_read_and_scored_in_photo_order(scored):
    assert "const SCORING = true;" in urllib.request.urlopen(scored).read().decode()
    first = post(scored + "turn", {"turn": 0, "player": "Ann", "used": 5, "left": 100})
    assert first["text"].startswith("1.jpg  Ann: AS")
    assert first["scores"] == [4, 0]                # (A 1 + S 1) x 2 on the centre
    second = post(scored + "turn", {"turn": 1, "player": "Bob", "used": 5, "left": 100})
    assert second["text"].startswith("2.jpg  Bob: AB")
    assert second["scores"] == [4, 4]
    third = post(scored + "turn", {"turn": 0, "player": "Ann", "used": 5, "left": 100})
    assert third["text"] == "10.jpg  Ann: pass or exchange"
    assert "no new photo" in post(scored + "turn", {"turn": 1, "player": "Bob", "used": 5, "left": 100})["text"]
    assert post(scored + "reset", {})["scores"] == [0, 0]
    assert post(scored + "turn", {"turn": 0, "player": "Ann", "used": 5, "left": 100})["text"].startswith("1.jpg")


def test_the_clock_decides_who_played(scored):
    # Bob presses first (say Ann's first turn was a pass the camera never saw): the move is his.
    assert post(scored + "turn", {"turn": 1, "player": "Bob", "used": 5, "left": 100})["scores"] == [0, 4]


def test_turn_of_unknown_player_is_refused(scored):
    request = urllib.request.Request(scored + "turn", json.dumps({"turn": 5, "player": "Eve", "used": 1, "left": 1})
                                     .encode(), {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(request)
    assert e.value.code == 400


def test_refused_move_can_be_accepted(tmp_path):
    (tmp_path / "1.jpg").write_bytes(b"")
    board = board_with(*[(7, c, "A") for c in range(8)])          # 8 tiles at once
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: board)
    assert "move refused" in scorer.turn(0)
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: board, accept_refused=True)
    assert "accepted anyway for 0 points" in scorer.turn(0)
    assert (scorer.game.board == board).all()


class FakeCamera:
    """Frames whose brightness numbers them (60, 120, ...), so the reader can tell which one it got."""

    def __init__(self):
        self.taken = 0

    def frame(self):
        self.taken += 1
        return np.full((16, 16, 3), 60 * self.taken, np.uint8)


def test_camera_photos_and_record_are_saved_per_game(tmp_path):
    import cv2
    old = [(7, 7, "A"), (7, 8, "S"), (8, 7, "B")]
    boards = {1: board_with(*old[:2]), 2: board_with(*old), 3: board_with(*old, (7, 9, "T"), (9, 7, "U"))}
    read = lambda path: boards.get(round(cv2.imread(str(path))[8, 8, 0] / 60), board_with(*old[:2]))
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, read, camera=FakeCamera())
    assert list(tmp_path.iterdir()) == []            # no folder before the first photo
    assert scorer.turn(1).startswith("001.jpg  Bob: AS")
    assert scorer.turn(0).startswith("002.jpg  Ann: AB")
    assert "move refused" in scorer.turn(1)          # new tiles not in one line
    assert scorer.game.scores == {"Ann": 4, "Bob": 4}
    first = scorer.folder
    assert sorted(p.name for p in first.iterdir()) == ["001.jpg", "002.jpg", "003.jpg", "game.json"]
    assert game.load_record(first) == {"players": ["Ann", "Bob"], "moves": [
        {"photo": "001.jpg", "player": "Bob", "forced": False},
        {"photo": "002.jpg", "player": "Ann", "forced": False},
        {"photo": "003.jpg", "player": "Bob", "forced": False}]}
    scorer.reset()                                    # a new game gets a new folder
    scorer.turn(0)
    assert scorer.folder != first and scorer.folder.parent == tmp_path
    assert [m["photo"] for m in game.load_record(scorer.folder)["moves"]] == ["001.jpg"]


def test_camera_without_picture_is_reported(tmp_path):
    class Broken:
        def frame(self):
            raise ValueError("no picture from the camera 4")
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: None, camera=Broken())
    assert scorer.turn(0) == "Ann: no picture from the camera 4"
