import cv2
import numpy as np

from scrabble.board import (CELL, CELLS, GRID_IN_BOARD, board_from_grid, cell_crops, fit_grid, grid_corners, rectify,
                            split_cells)


def checkerboard():
    """Flat 15x15 board image where cell (row, col) has the gray value 10 * row + col."""
    board = np.zeros((CELLS * CELL, CELLS * CELL, 3), np.uint8)
    for r in range(CELLS):
        for c in range(CELLS):
            board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = 10 * r + c
    return board


def test_grid_corners_of_an_axis_aligned_board():
    left, top, right, bottom = GRID_IN_BOARD
    grid = grid_corners([[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    np.testing.assert_allclose(grid, [[left * 1000, top * 1000], [right * 1000, top * 1000],
                                      [right * 1000, bottom * 1000], [left * 1000, bottom * 1000]], atol=1e-3)


def test_rectify_undoes_perspective_and_keeps_corner_order():
    flat = checkerboard()
    side = CELLS * CELL
    # Photo of the board rotated 90 degrees clockwise and seen at an angle.
    grid = np.float32([[900, 120], [880, 760], [150, 700], [200, 90]])
    square = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    H = cv2.getPerspectiveTransform(square, grid)
    photo = cv2.warpPerspective(flat, H, (1000, 900), flags=cv2.INTER_NEAREST)
    # Board corners that put the grid exactly there.
    left, top, right, bottom = GRID_IN_BOARD
    gx, gy = (right - left), (bottom - top)
    board = np.float32([[-left / gx, -top / gy], [(1 - left) / gx, -top / gy],
                        [(1 - left) / gx, (1 - top) / gy], [-left / gx, (1 - top) / gy]]) * side
    board_corners = cv2.perspectiveTransform(board.reshape(-1, 1, 2), H).reshape(-1, 2)

    cells = split_cells(rectify(photo, board_corners))

    assert cells.shape == (CELLS, CELLS, CELL, CELL, 3)
    centre = CELL // 2
    for r in range(CELLS):
        for c in range(CELLS):
            assert abs(int(cells[r, c, centre, centre, 0]) - (10 * r + c)) <= 1


def test_split_cells_grayscale():
    board = np.arange(CELLS * CELL * CELLS * CELL, dtype=np.int32).reshape(CELLS * CELL, CELLS * CELL)
    cells = split_cells(board)
    assert cells.shape == (CELLS, CELLS, CELL, CELL)
    assert cells[2, 3, 0, 0] == board[2 * CELL, 3 * CELL]


def test_cell_crops_include_padding_around_each_cell():
    flat = checkerboard()
    side = CELLS * CELL
    left, top, right, bottom = GRID_IN_BOARD
    gx, gy = (right - left), (bottom - top)
    # Board corners of an axis-aligned photo whose grid is exactly the flat checkerboard.
    board_corners = np.float32([[-left / gx, -top / gy], [(1 - left) / gx, -top / gy],
                                [(1 - left) / gx, (1 - top) / gy], [-left / gx, (1 - top) / gy]]) * side
    pad = 8
    crops = cell_crops(flat, board_corners, CELL, pad)
    assert crops.shape == (CELLS, CELLS, CELL + 2 * pad, CELL + 2 * pad, 3)
    middle = pad + CELL // 2
    assert abs(int(crops[3, 4, middle, middle, 0]) - 34) <= 1
    assert abs(int(crops[3, 4, middle, 2, 0]) - 33) <= 1       # the padding shows the neighbour on the left


def test_board_from_grid_inverts_grid_corners():
    board = np.float32([[120, 80], [930, 140], [880, 900], [90, 860]])
    np.testing.assert_allclose(board_from_grid(grid_corners(board)), board, atol=1e-2)


def printed_board(rng):
    """Flat board like the real one: teal cells, thin light grid lines, some tiles; grid at a 2-cell margin."""
    cell, margin = 40, 80
    side = CELLS * cell + 2 * margin
    flat = np.full((side, side, 3), (110, 120, 20), np.uint8)
    for i in range(CELLS + 1):
        p = margin + i * cell
        cv2.line(flat, (p, margin), (p, side - margin), (210, 220, 220), 2)
        cv2.line(flat, (margin, p), (side - margin, p), (210, 220, 220), 2)
    for r, c in rng.integers(0, CELLS, (40, 2)):       # tiles: bright squares with a dark letter
        x, y = margin + c * cell, margin + r * cell
        cv2.rectangle(flat, (x + 3, y + 3), (x + cell - 3, y + cell - 3), (170, 220, 240), -1)
        cv2.putText(flat, "A", (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
    grid = np.float32([[0, 0], [CELLS, 0], [CELLS, CELLS], [0, CELLS]]) * cell + margin
    return flat, grid


def test_fit_grid_moves_shifted_corners_onto_the_printed_lines():
    rng = np.random.default_rng(0)
    flat, grid_flat = printed_board(rng)
    photo_grid = np.float32([[260, 180], [1010, 240], [960, 980], [210, 930]])
    H = cv2.getPerspectiveTransform(grid_flat, photo_grid)
    photo = cv2.warpPerspective(flat, H, (1200, 1100))
    true = board_from_grid(photo_grid)
    cell = np.linalg.norm(photo_grid[1] - photo_grid[0]) / CELLS
    off = true + rng.uniform(-0.35, 0.35, (4, 2)).astype(np.float32) * cell

    def worst(corners):                                 # worst grid-corner error, in cells
        return np.linalg.norm(grid_corners(corners) - photo_grid, axis=1).max() / cell

    assert worst(off) > 0.2
    assert worst(fit_grid(photo, off)) < 0.1


def test_fit_grid_keeps_the_corners_without_lines():
    blank = np.full((900, 900, 3), 90, np.uint8)
    corners = np.float32([[100, 100], [800, 120], [790, 800], [110, 780]])
    np.testing.assert_array_equal(fit_grid(blank, corners), corners)
