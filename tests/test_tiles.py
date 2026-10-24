from collections import Counter

import numpy as np

from scrabble.board import CELLS
from scrabble.tiles import (EMPTY, LAYOUT_STARTS, TILE_COUNTS, cell_name, class_name, layout, save_cells,
                            tile_sequence)
import scrabble.tiles as tiles


def test_tile_set_has_100_tiles():
    assert len(tile_sequence()) == 100
    assert len(TILE_COUNTS) == 33          # 32 letters + blank


def test_every_layout_holds_the_whole_tile_set():
    for i in range(len(LAYOUT_STARTS)):
        placed = Counter(t for row in layout(i) for t in row if t != EMPTY)
        assert placed == Counter(dict(TILE_COUNTS))


def test_layout_starts_at_its_cell_and_wraps():
    grid = layout(len(LAYOUT_STARTS) - 1)            # starts at cell 190, so it wraps past O15
    r, c = divmod(LAYOUT_STARTS[-1], CELLS)
    assert grid[r][c] == "A"
    assert grid[r][c - 1] == EMPTY
    assert grid[0][0] != EMPTY                        # continued from A1
    assert grid[CELLS - 1][CELLS - 1] != EMPTY


def test_every_cell_is_empty_in_some_layout_and_covered_in_another():
    grids = [layout(i) for i in range(len(LAYOUT_STARTS))]
    for r in range(CELLS):
        for c in range(CELLS):
            values = {g[r][c] == EMPTY for g in grids}
            assert values == {True, False}


def test_names():
    assert cell_name(0) == "A1"
    assert cell_name(38) == "C9"
    assert cell_name(224) == "O15"
    assert class_name(".") == "_empty" and class_name("?") == "_blank" and class_name("Ż") == "Ż"


def test_save_cells_sorts_crops_into_class_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(tiles, "TILES_DIR", tmp_path)
    cells = np.zeros((CELLS, CELLS, 12, 12, 3), np.uint8)
    save_cells(cells, layout(0), "L01_x")
    assert (tmp_path / "cells" / "A" / "L01_x_A1.png").exists()
    assert len(list((tmp_path / "cells").rglob("*.png"))) == CELLS * CELLS
    assert len(list((tmp_path / "cells" / "A").iterdir())) == 9
