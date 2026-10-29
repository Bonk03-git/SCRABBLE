import json
import threading
import urllib.request

import numpy as np
import pytest

from scrabble import camera_page, clock, game


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
    assert third["text"] == "10.jpg  Ann: pause"
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
    assert game.load_record(scorer.folder) == {"players": ["Ann", "Bob"], "moves": []}   # made at the start
    assert scorer.turn(1).startswith("001.jpg  Bob: AS")
    assert scorer.turn(0).startswith("002.jpg  Ann: AB")
    assert "move refused" in scorer.turn(1)          # new tiles not in one line
    assert scorer.game.scores == {"Ann": 4, "Bob": 4}
    first = scorer.folder
    assert sorted(p.name for p in first.iterdir()) == ["001.jpg", "002.jpg", "003.jpg", "game.json"]
    assert game.load_record(first) == {"players": ["Ann", "Bob"], "moves": [
        {"photo": "001.jpg", "player": "Bob", "forced": False, "exchange": False},
        {"photo": "002.jpg", "player": "Ann", "forced": False, "exchange": False},
        {"photo": "003.jpg", "player": "Bob", "forced": False, "exchange": False}]}
    scorer.reset()                                    # a new game gets a new folder
    scorer.turn(0)
    assert scorer.folder != first and scorer.folder.parent == tmp_path
    assert [m["photo"] for m in game.load_record(scorer.folder)["moves"]] == ["001.jpg"]
    scorer.reset()
    scorer.reset()                                    # a game without photos leaves no folder
    assert len(list(tmp_path.iterdir())) == 3 and scorer.folder.exists()


def test_camera_without_picture_is_reported(tmp_path):
    class Broken:
        def frame(self):
            raise ValueError("no picture from the camera 4")
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: None, camera=Broken())
    assert scorer.turn(0) == "Ann: no picture from the camera 4"


def jpeg(value):
    import cv2
    return cv2.imencode(".jpg", np.full((16, 16, 3), value, np.uint8))[1].tobytes()


def test_camera_page_takes_the_photos_in_order():
    page = camera_page.CameraPage(timeout=2)
    with pytest.raises(ValueError, match="not open"):         # nobody asked the server for photos yet
        page.request()
        page.frame()
    assert page.next_shot(0, wait=0) is None
    page.request()
    page.request()
    assert page.next_shot(0, wait=0) == 2                     # the first one was dropped above
    assert page.next_shot(2, wait=0) == 3
    assert not page.deliver(1, jpeg(10))                      # nobody waits for it any more
    assert page.deliver(3, jpeg(200)) and page.deliver(2, jpeg(100))
    assert page.next_shot(0, wait=0) is None
    assert abs(int(page.frame()[8, 8, 0]) - 100) < 5
    assert abs(int(page.frame()[8, 8, 0]) - 200) < 5
    page.request()
    with pytest.raises(ValueError, match="no photo in 2 s"):
        page.frame()


def test_camera_page_server_scores_each_turn_with_its_photo(tmp_path):
    import cv2
    old = [(7, 7, "A"), (7, 8, "S")]
    boards = {1: board_with(*old), 2: board_with(*old, (8, 7, "B"))}
    read = lambda path: boards[round(cv2.imread(str(path))[8, 8, 0] / 100)]
    page = camera_page.CameraPage(timeout=5)
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, read, camera=page)
    server = clock.make_server(["Ann", "Bob"], 25, scorer=scorer, camera=page,
                               find_grid=lambda data: {"lines": [], "conf": [len(data)]}, preview="http://x/")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    clock_page = urllib.request.urlopen(url).read().decode()
    assert 'href="camera"' in clock_page and 'href="http://x/"' in clock_page
    assert "getUserMedia" in urllib.request.urlopen(url + "camera").read().decode()

    def camera(shots):
        """The camera page: answers each asked-for photo with a frame numbered by its order."""
        after = 0
        for value in shots:
            after = json.loads(urllib.request.urlopen(url + f"camera/next?after={after}").read())["shot"]
            urllib.request.urlopen(urllib.request.Request(url + f"camera/photo?shot={after}", jpeg(value)))

    browser = threading.Thread(target=camera, args=([100, 200],))
    browser.start()
    while not page.is_open():                  # the camera page is waiting for the clock
        pass
    results = [None, None]

    def press(i, player):
        results[i] = post(url + "turn", {"turn": i, "player": player, "used": 5, "left": 100})

    presses = [threading.Thread(target=press, args=(0, "Ann"))]
    presses[0].start()
    while page.next_id < 2:                    # the second press comes after the first
        pass
    presses.append(threading.Thread(target=press, args=(1, "Bob")))
    presses[1].start()
    for t in presses + [browser]:
        t.join(10)
    assert results[0]["text"].startswith("001.jpg  Ann: AS") and results[1]["text"].startswith("002.jpg  Bob: AB")
    assert results[1]["scores"] == [4, 4]
    grid = urllib.request.urlopen(urllib.request.Request(url + "camera/grid", b"abc")).read()
    assert json.loads(grid) == {"lines": [], "conf": [3]}
    server.shutdown()
    server.server_close()


def test_grid_lines_follow_the_grid():
    from scrabble import corners
    from scrabble.board import board_from_grid
    grid = np.float32([[0, 0], [150, 0], [150, 150], [0, 150]])
    lines = corners.grid_lines(board_from_grid(grid))
    assert lines.shape == (32, 2, 2)
    assert np.allclose(lines[0], [[0, 0], [0, 150]], atol=1e-3)       # first vertical line
    assert np.allclose(lines[3], [[0, 10], [150, 10]], atol=1e-3)     # second horizontal line


def test_phone_gets_the_camera_page_over_https(tmp_path):
    import ssl
    page = camera_page.CameraPage()
    server = clock.make_server(["Ann", "Bob"], 25, camera=page, phone="https://192.168.1.2:8643/camera")
    clock.use_https(server, *clock.certificate("127.0.0.1", tmp_path))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    context = ssl.create_default_context(cafile=tmp_path / "127.0.0.1.pem")    # trusted, as on the phone
    url = f"https://127.0.0.1:{server.server_port}/"
    assert "getUserMedia" in urllib.request.urlopen(url + "camera", context=context).read().decode()
    assert "https://192.168.1.2:8643/camera" in urllib.request.urlopen(url, context=context).read().decode()
    server.shutdown()
    server.server_close()


def test_end_of_game_is_reckoned_and_recorded(tmp_path):
    board = board_with((7, 7, "A"), (7, 8, "S"))
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: board, camera=FakeCamera())
    scorer.game.bag, scorer.game.racks["Ann"] = 0, 2
    server = clock.make_server(["Ann", "Bob"], 25, scorer=scorer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    result = post(url + "turn", {"turn": 0, "player": "Ann", "used": 5, "left": 100})
    end = result["end"]                               # Ann played her last 2 tiles
    assert end["reason"] == "out" and end["player"] == "Ann" and end["racks"] == [0, 7]
    assert end["known"] == [0, end["left"]]           # the rest is all on Bob's rack
    final = post(url + "end", {"reason": "out", "player": "Ann", "left": [0, 12]})
    assert final["scores"] == [4 + 12, -12] and "end" not in final
    assert game.load_record(scorer.folder)["end"] == {"reason": "out", "player": "Ann", "left": {"Ann": 0, "Bob": 12}}
    # PFS 6.2.1: when Ann's time runs out, Bob plays on alone; two pauses of his end the game.
    post(url + "reset", {})
    scorer.read = lambda path: game.empty_board()
    assert "Bob play on" in post(url + "timeout", {"turn": 0, "player": "Ann"})["text"]
    assert "end" not in post(url + "turn", {"turn": 1, "player": "Bob", "used": 5, "left": 100})
    end = post(url + "turn", {"turn": 1, "player": "Bob", "used": 5, "left": 100})["end"]
    assert end["reason"] == "passes" and end["known"] == [None, None]     # the bag is full: racks unknown
    assert post(url + "end", {"reason": "passes", "player": None, "left": [5, 7]})["scores"] == [-5, -7]
    # Everyone's time running out ends the game too.
    post(url + "reset", {})
    post(url + "timeout", {"turn": 1, "player": "Bob"})
    assert post(url + "timeout", {"turn": 0, "player": "Ann"})["end"]["reason"] == "time"
    server.shutdown()
    server.server_close()


def test_exchange_marked_on_the_clock_is_not_a_pause(tmp_path):
    scorer = clock.Scorer(["Ann", "Bob"], tmp_path, lambda path: game.empty_board(), camera=FakeCamera())
    for i in range(3):
        scorer.turn(i % 2)
    assert scorer.turn(1, exchange=True).endswith("Bob: exchange")
    assert scorer.game.passes == 0 and scorer.game.end() is None
    assert game.load_record(scorer.folder)["moves"][-1]["exchange"] is True
