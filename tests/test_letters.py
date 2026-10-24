from pathlib import Path

import numpy as np

from scrabble.letters import CLASSES, INPUT, centre, photo_of, random_cell


def test_classes_are_32_letters_blank_and_empty():
    assert len(CLASSES) == 34
    assert CLASSES[-2:] == ["_blank", "_empty"]
    assert len(set(CLASSES)) == 34


def test_photo_of_cell_name():
    assert photo_of(Path("cells/A/L01_20261024_203525_C9.png")) == "L01_20261024_203525"
    assert photo_of(Path("cells/Ż/L06_WhatsApp Image (2)_O15.png")) == "L06_WhatsApp Image (2)"


def test_centre_and_random_cell_give_one_cell():
    crop = np.zeros((96, 96, 3), np.uint8)
    crop[16:80, 16:80] = 200                     # the cell, padding around it is black
    assert centre(crop).shape == (INPUT, INPUT, 3)
    assert centre(crop).min() == 200
    view = random_cell(crop, np.random.default_rng(0))
    assert view.shape == (INPUT, INPUT, 3)
    assert view[INPUT // 2, INPUT // 2, 0] == 200
