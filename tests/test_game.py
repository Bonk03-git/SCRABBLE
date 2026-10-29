import pytest

from scrabble.game import PREMIUMS, VALUES, Game, MoveError, drop_isolated, empty_board, final_scores, find_move
from scrabble.tiles import TILE_COUNTS


def place(board, row, col, text, across=True):
    board = board.copy()
    for i, letter in enumerate(text):
        board[row + (0 if across else i), col + (i if across else 0)] = letter
    return board


def test_every_tile_has_a_value_and_premiums_are_symmetric():
    assert set(VALUES) == {letter for letter, _ in TILE_COUNTS}
    assert PREMIUMS == PREMIUMS[::-1]
    assert all(row == row[::-1] for row in PREMIUMS)
    assert ["".join(col) for col in zip(*PREMIUMS)] == PREMIUMS


def test_first_move_doubles_on_the_centre():
    move = find_move(empty_board(), place(empty_board(), 7, 6, "KOT"))
    assert [(w.text, w.start, w.across) for w in move.words] == [("KOT", "H7", True)]
    assert move.score == (2 + 1 + 2) * 2
    word = move.words[0]
    assert [(l.symbol, l.premium) for l in word.letters] == [("K", ""), ("O", "w"), ("T", "")]
    assert word.multiplier == 2


def test_blank_scores_zero():
    assert find_move(empty_board(), place(empty_board(), 7, 6, "K?T")).score == (2 + 0 + 2) * 2


def test_all_seven_tiles_get_the_bonus():
    move = find_move(empty_board(), place(empty_board(), 7, 1, "AAAAAAA"))
    assert move.bingo
    assert move.score == (7 + 1) * 2 + 50          # one A on a double letter, centre doubles the word


def test_single_tile_scores_the_word_it_makes():
    before = place(empty_board(), 7, 6, "KOT")
    move = find_move(before, place(before, 8, 8, "A"))
    assert [(w.text, w.across) for w in move.words] == [("TA", False)]
    assert move.score == 2 + 1 * 2                  # the old T's centre bonus is not counted again
    assert [(l.premium, l.new, l.points) for l in move.words[0].letters] == [("", False, 2), ("l", True, 2)]


def test_cross_words_are_scored_too():
    before = place(empty_board(), 7, 6, "KOT")
    move = find_move(before, place(before, 8, 7, "AS"))
    assert {w.text: w.score for w in move.words} == {"AS": 1 + 1 * 2, "OA": 1 + 1, "TS": 2 + 1 * 2}


def test_move_may_fill_around_old_tiles():
    before = place(empty_board(), 7, 6, "KOT")
    move = find_move(before, place(before, 7, 5, "AKOTY"))
    assert [w.text for w in move.words] == ["AKOTY"]
    assert [t[1] for t in move.tiles] == [5, 9]


def test_no_new_tiles_is_a_pass():
    board = place(empty_board(), 7, 6, "KOT")
    assert find_move(board, board) is None


@pytest.mark.parametrize("before_word, row, col, text, across, message", [
    (None, 7, 3, "KOT", True, "centre"),
    (None, 7, 7, "K", True, "single tile"),
    ("KOT", 0, 0, "AB", True, "touch"),
    ("KOT", 8, 6, "AB", False, None),      # legal: AB down from row I, A under K
])
def test_first_move_and_touching_rules(before_word, row, col, text, across, message):
    before = place(empty_board(), 7, 6, before_word) if before_word else empty_board()
    board = place(before, row, col, text, across)
    if message is None:
        assert find_move(before, board).words[0].text == "KAB"
    else:
        with pytest.raises(MoveError, match=message):
            find_move(before, board)


def test_tiles_off_one_line_are_refused():
    before = place(empty_board(), 7, 6, "KOT")
    board = place(before, 8, 6, "A")
    board[9, 7] = "B"
    with pytest.raises(MoveError, match="one row or column"):
        find_move(before, board)


def test_gap_and_changed_tiles_are_refused():
    before = place(empty_board(), 7, 6, "KOT")
    gap = place(before, 7, 10, "A")
    gap[7, 12] = "B"
    with pytest.raises(MoveError, match="gap"):
        find_move(before, gap)
    changed = place(before, 7, 6, "K")
    changed[7, 7] = "A"
    with pytest.raises(MoveError, match="H8"):
        find_move(before, changed)


def test_game_adds_scores_and_rotates_players():
    game = Game(["Ala", "Ola"])
    b1 = place(empty_board(), 7, 6, "KOT")
    b2 = place(b1, 8, 8, "A")
    assert game.play(b1)[0] == "Ala"
    assert game.play(b2)[0] == "Ola"
    assert game.play(b2) == ("Ala", None)
    assert game.scores == {"Ala": 10, "Ola": 4}
    off_line = place(place(b2, 6, 6, "X"), 8, 7, "Y")
    with pytest.raises(MoveError):
        game.play(off_line)
    assert game.player == "Ola" and game.scores == {"Ala": 10, "Ola": 4}


def test_game_keeps_old_tiles_that_are_misread():
    game = Game(["Ala", "Ola"])
    game.play(place(empty_board(), 7, 6, "KOT"))
    read = place(empty_board(), 7, 6, "KDT")            # O misread as D ...
    read[7, 8] = "."                                    # ... T not seen at all
    player, move = game.play(place(read, 8, 7, "AS"))
    assert game.misread == ["H8", "H9"]
    assert {w.text for w in move.words} == {"AS", "OA", "TS"}
    assert "".join(game.board[7, 6:9]) == "KOT"


def test_forced_move_adds_the_tiles_for_no_points():
    game = Game(["Ala", "Ola"])
    eight = place(empty_board(), 7, 3, "PODŁOŻYĆ")
    with pytest.raises(MoveError):
        game.play(eight)
    player, move = game.play(eight, force=True)
    assert player == "Ala" and move.score == 0 and len(move.tiles) == 8
    assert game.play(place(eight, 8, 3, "A"))[1].score == (2 + 1) * 1    # PA down, nothing doubled


def test_tiles_apart_from_the_others_are_dropped():
    board = place(place(empty_board(), 7, 6, "KOT"), 0, 0, "AB")
    board[14, 14] = "?"
    kept, dropped = drop_isolated(board)
    assert dropped == ["A1", "A2", "O15"]
    assert (kept == place(empty_board(), 7, 6, "KOT")).all()


def test_largest_group_stays_without_a_tile_on_the_centre():
    board = place(place(empty_board(), 3, 2, "KOT"), 12, 12, "A")
    kept, dropped = drop_isolated(board)
    assert dropped == ["M13"] and "".join(kept[3, 2:5]) == "KOT"


def test_game_ignores_a_misread_tile_apart_from_the_others():
    game = Game(["Ala", "Ola"])
    game.play(place(empty_board(), 7, 6, "KOT"))
    player, move = game.play(place(place(game.board, 8, 7, "AS"), 3, 3, "?"))     # a star read as a blank
    assert game.ignored == ["D4"]
    assert {w.text for w in move.words} == {"AS", "OA", "TS"}
    assert game.board[3, 3] == "."


def test_bag_and_racks_follow_the_tiles_played():
    game = Game(["Ala", "Ola"])
    assert game.bag == 86 and game.racks == {"Ala": 7, "Ola": 7}
    board = place(empty_board(), 7, 6, "KOT")
    game.play(board)
    assert game.bag == 83 and game.racks == {"Ala": 7, "Ola": 7}      # Ala drew 3
    game.bag = 1                                                        # near the end
    game.play(place(board, 8, 6, "AL", across=False))
    assert game.bag == 0 and game.racks == {"Ala": 7, "Ola": 6}        # only 1 of 2 drawn
    assert game.end() is None


def test_game_ends_when_a_player_plays_out_with_the_bag_empty():
    game = Game(["Ala", "Ola"])
    game.bag, game.racks["Ala"] = 0, 3
    game.play(place(empty_board(), 7, 6, "KOT"))
    assert game.end() == ("out", "Ala")
    # What is not on the board is on Ola's rack: all 100 tiles but K, O, T.
    assert game.left_value() == sum(VALUES[l] * n for l, n in TILE_COUNTS) - 2 - 1 - 2


def test_game_ends_when_everyone_passes_twice():
    game = Game(["Ala", "Ola", "Ela"])
    board = place(empty_board(), 7, 6, "KOT")
    game.play(board)
    for _ in range(5):
        game.play(board)
        assert game.end() is None
    game.play(board)
    assert game.end() == ("passes", None)


def test_final_scores_take_off_the_racks_and_reward_playing_out():
    scores = {"Ala": 100, "Ola": 90, "Ela": 80}
    assert final_scores(scores, {"Ala": 0, "Ola": 8, "Ela": 2}, out="Ala") == {"Ala": 110, "Ola": 82, "Ela": 78}
    assert final_scores(scores, {"Ala": 3, "Ola": 8, "Ela": 2}) == {"Ala": 97, "Ola": 82, "Ela": 78}


def test_pfs_time_out_lets_the_others_play_on():
    game = Game(["Ala", "Ola"])
    board = place(empty_board(), 7, 6, "KOT")
    game.play(board)
    game.play(board)                                    # Ola pauses
    game.time_out("Ala")                                # the pauses are counted anew
    game.turn = 1
    game.play(board)
    assert game.end() is None
    game.turn = 1
    game.play(board)                                    # two pauses of Ola alone end it (6.2.1)
    assert game.end() == ("passes", None)
    game = Game(["Ala", "Ola"])
    game.time_out("Ala")
    game.time_out("Ola")
    assert game.end() == ("time", None)


def test_exchange_breaks_the_pauses():
    game = Game(["Ala", "Ola"])
    board = place(empty_board(), 7, 6, "KOT")
    game.play(board)
    for exchange in (False, False, True, False, False, False):
        game.play(board, exchange=exchange)
        assert game.end() is None
    game.play(board)
    assert game.end() == ("passes", None)
