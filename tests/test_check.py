import cv2
import numpy as np

from scrabble.board import CELLS, board_from_grid, grid_corners
from scrabble.check import label_grid
from tests.test_board import printed_board


def test_label_grid_moves_an_inaccurate_label_onto_the_printed_lines():
    rng = np.random.default_rng(0)
    flat, grid_flat = printed_board(rng)
    photo_grid = np.float32([[260, 180], [1010, 240], [960, 980], [210, 930]])
    photo = cv2.warpPerspective(flat, cv2.getPerspectiveTransform(grid_flat, photo_grid), (1200, 1100))
    cell = np.linalg.norm(photo_grid[1] - photo_grid[0]) / CELLS
    label = board_from_grid(photo_grid) + rng.uniform(-0.35, 0.35, (4, 2)).astype(np.float32) * cell

    grid, moved = label_grid(photo, label)

    assert np.linalg.norm(grid - photo_grid, axis=1).max() / cell < 0.1
    assert moved > 0


def test_label_grid_keeps_the_label_without_lines():
    blank = np.full((900, 900, 3), 90, np.uint8)
    corners = np.float64([[100, 100], [800, 120], [790, 800], [110, 780]])
    grid, moved = label_grid(blank, corners)
    np.testing.assert_allclose(grid, grid_corners(corners), atol=1e-3)
    assert moved == 0
