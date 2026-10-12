import json
from pathlib import Path

import numpy as np
import pytest

from scrabble.annotate import fit_scale, is_clockwise_convex, list_images, load_corners, make_label, save_annotation

CORNERS = [[100.0, 120.0], [880.0, 100.0], [900.0, 600.0], [120.0, 610.0]]
CLICKS = CORNERS


def test_clockwise_convex_accepts_any_starting_corner():
    # The board's top-left may be anywhere in a rotated photo; only the direction matters.
    for shift in range(4):
        assert is_clockwise_convex(np.roll(CORNERS, shift, axis=0))


def test_clockwise_convex_rejects_wrong_order():
    tl, tr, br, bl = CORNERS
    assert not is_clockwise_convex([tl, bl, br, tr])  # counter-clockwise
    assert not is_clockwise_convex([tl, br, tr, bl])  # crossing edges


def test_make_label_keeps_click_order():
    rotated = [CORNERS[1], CORNERS[2], CORNERS[3], CORNERS[0]]  # board turned 90 degrees in the photo
    assert make_label("a.jpg", rotated) == {"image": "a.jpg", "corners": rotated}


def test_save_moves_photo_and_writes_label(tmp_path: Path):
    photos = tmp_path / "dataset" / "photos"
    photos.mkdir(parents=True)
    photo = photos / "a.jpg"
    photo.write_bytes(b"jpeg")

    target = save_annotation(photo, CLICKS, tmp_path / "dataset")

    assert target == tmp_path / "dataset" / "annotated" / "a.jpg"
    assert target.read_bytes() == b"jpeg" and not photo.exists()
    label = tmp_path / "dataset" / "labels" / "a.json"
    assert json.loads(label.read_text()) == {"image": "a.jpg", "corners": CORNERS}
    assert load_corners(label) == [tuple(c) for c in CORNERS]


def test_relabel_in_annotated_overwrites_label_without_moving(tmp_path: Path):
    dataset = tmp_path / "dataset"
    done = dataset / "annotated"
    done.mkdir(parents=True)
    photo = done / "a.jpg"
    photo.write_bytes(b"jpeg")
    save_annotation(photo, CLICKS, dataset)

    moved = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert save_annotation(photo, moved, dataset) == photo
    assert photo.exists()
    assert json.loads((dataset / "labels" / "a.json").read_text())["corners"] == moved


def test_save_refuses_to_overwrite_annotated_photo(tmp_path: Path):
    dataset = tmp_path / "dataset"
    (dataset / "annotated").mkdir(parents=True)
    (dataset / "annotated" / "a.jpg").write_bytes(b"old")
    (dataset / "photos").mkdir()
    photo = dataset / "photos" / "a.jpg"
    photo.write_bytes(b"new")

    with pytest.raises(FileExistsError):
        save_annotation(photo, CLICKS, dataset)
    assert photo.exists() and (dataset / "annotated" / "a.jpg").read_bytes() == b"old"
    assert not (dataset / "labels" / "a.json").exists()


def test_list_images_skips_non_images(tmp_path: Path):
    for name in ["2.png", "1.jpg", "photo.JPG", "notes.txt"]:
        (tmp_path / name).touch()
    assert [p.name for p in list_images(tmp_path)] == ["1.jpg", "2.png", "photo.JPG"]


def test_fit_scale():
    assert fit_scale(640, 480) == 1.0
    assert fit_scale(2560, 1440) == 0.5
