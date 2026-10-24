import cv2
import numpy as np

from scrabble.board import CELL, CELLS, GRID_IN_BOARD, grid_corners, rectify, split_cells


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
