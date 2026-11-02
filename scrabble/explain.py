"""Heatmaps of what the corner model looks at.

For each photo this writes one sheet with a row per corner (1 = the board's top-left, then
clockwise). The photo is shown in gray as the network sees it (256x256 letterboxed); the
stronger the colour, the more that place matters for the corner:

  output     the network's own heatmap of the corner
  occlusion  how much the corner's peak drops when a square there is covered with the mean
             colour, as a share of the peak (a measurement, on an absolute scale: the peak
             value of the panel is printed on it)
  layer4/3/2 Grad-CAM at that backbone layer (coarse to fine; 8, 16 and 32 cells a side).
             Each map is scaled to its own maximum, so compare places, not panels

Grad-CAM here multiplies gradients and activations place by place (HiResCAM) instead of
averaging the gradients per channel: one corner is one spot of the output, and with averaged
gradients a channel would light up everywhere it fires, also at the other corners.

The circle is the network's corner (no fit to the board lines). The cross is the labelled
grid corner, when the photo has a label in ../labels/ next to its folder.

Usage:
  python -m scrabble.explain PHOTO_OR_DIR [...] [--model FILE] [--out DIR]

Sheets go to dataset/explain/<photo stem>.jpg by default.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scrabble.corners import INPUT, MODEL_PATH, STRIDE, letterbox_matrix, load_model, to_tensor, transform_points

BACKBONE_LAYERS = {"layer4": 7, "layer3": 6, "layer2": 5}
OCCLUSION_SIZE = 32             # about 2.5 cells of a board that fills the input
OCCLUSION_STRIDE = 16
OCCLUSION_BATCH = 32
PANEL = 320
OVERLAY_BGR = np.array([0, 60, 255], np.float32)    # one hue: more colour means more weight
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png"}


def peak_cells(heat):
    """(y, x) of the highest output cell of each corner; `heat` is 4 x H x W."""
    return [np.unravel_index(int(hm.argmax()), hm.shape) for hm in heat.detach().cpu().numpy()]


def grad_cam(model, x, layer):
    """HiResCAM of every corner's output peak at a backbone layer: 4 x INPUT x INPUT, each
    map scaled to at most 1. `x` is one input as a batch of 1."""
    saved = []
    hook = model.backbone[BACKBONE_LAYERS[layer]].register_forward_hook(lambda _m, _i, out: saved.append(out))
    try:
        with torch.enable_grad():
            heat = model(x)[0]
            maps = []
            for k, (py, px) in enumerate(peak_cells(heat)):
                grad = torch.autograd.grad(heat[k, py, px], saved[0], retain_graph=True)[0]
                maps.append((grad * saved[0]).sum(1).clamp(min=0))
    finally:
        hook.remove()
    cams = torch.nn.functional.interpolate(torch.stack(maps).detach(), size=(INPUT, INPUT), mode="bilinear",
                                           align_corners=False)[:, 0].cpu().numpy()
    return cams / np.maximum(cams.max(axis=(1, 2), keepdims=True), 1e-8)


@torch.no_grad()
def occlusion(model, x, size=OCCLUSION_SIZE, stride=OCCLUSION_STRIDE, batch=OCCLUSION_BATCH):
    """How much each corner's peak drops when a square is covered with the mean colour, as a
    share of the peak and at least 0: 4 x INPUT x INPUT. A pixel gets the average over the
    squares that cover it."""
    heat = model(x)[0]
    cells = peak_cells(heat)
    base = torch.stack([heat[k, py, px] for k, (py, px) in enumerate(cells)])
    starts = range(0, INPUT - size + 1, stride)
    squares = [(top, left) for top in starts for left in starts]
    drops = np.zeros((4, INPUT, INPUT), np.float32)
    counts = np.zeros((INPUT, INPUT), np.float32)
    for i in range(0, len(squares), batch):
        chunk = squares[i:i + batch]
        covered = x.repeat(len(chunk), 1, 1, 1)
        for j, (top, left) in enumerate(chunk):
            covered[j, :, top:top + size, left:left + size] = 0     # 0 is the mean colour after to_tensor
        out = model(covered)
        for k, (py, px) in enumerate(cells):
            drop = ((base[k] - out[:, k, py, px]) / base[k].clamp(min=1e-3)).cpu().numpy()
            for j, (top, left) in enumerate(chunk):
                drops[k, top:top + size, left:left + size] += drop[j]
        for top, left in chunk:
            counts[top:top + size, left:left + size] += 1
    return np.clip(drops / np.maximum(counts, 1), 0, None)


def overlay(gray, values):
    """`values` (0..1, any size) as one colour over a BGR gray image, stronger where higher."""
    values = cv2.resize(np.clip(values, 0, 1).astype(np.float32), gray.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    alpha = 0.85 * values[..., None]
    return (gray * (1 - alpha) + OVERLAY_BGR * alpha).astype(np.uint8)


def panel(gray, values, title, corner, truth):
    out = overlay(gray, values)
    scale = PANEL / INPUT
    if truth is not None:
        cv2.drawMarker(out, tuple(int(v) for v in truth * scale), (255, 255, 0), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
    cv2.circle(out, tuple(int(v) for v in corner * scale), 6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (PANEL, 22), (0, 0, 0), -1)
    cv2.putText(out, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def label_grid(photo):
    """The labelled grid corners of a photo in .../annotated/, or None."""
    label = photo.parent.parent / "labels" / f"{photo.stem}.json"
    if not label.exists():
        return None
    return json.loads(label.read_text(encoding="utf-8")).get("grid")


def explain(model, image, grid=None):
    """One sheet for a BGR photo, with `grid` the labelled grid corners in photo pixels, if any."""
    h, w = image.shape[:2]
    M = letterbox_matrix(w, h)
    view = cv2.warpPerspective(image, M, (INPUT, INPUT))
    x = to_tensor(view)[None]
    with torch.no_grad():
        heat = model(x)[0]
    outputs = heat.clamp(0, 1).numpy()
    corners = np.array([((px + 0.5) * STRIDE - 0.5, (py + 0.5) * STRIDE - 0.5) for py, px in peak_cells(heat)])
    truths = transform_points(M, grid) if grid is not None else [None] * 4
    occluded = occlusion(model, x)
    cams = {layer: grad_cam(model, x, layer) for layer in BACKBONE_LAYERS}

    gray = cv2.cvtColor(cv2.cvtColor(cv2.resize(view, (PANEL, PANEL)), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    gray = (gray * 0.8).astype(np.float32)
    rows = []
    for k in range(4):
        peak = float(heat[k].max())
        cells = [panel(gray, outputs[k], f"corner {k + 1}: output, peak {peak:.2f}", corners[k], truths[k]),
                 panel(gray, occluded[k], f"occlusion, max drop {occluded[k].max():.0%}", corners[k], truths[k])]
        cells += [panel(gray, cams[layer][k], f"{layer}", corners[k], truths[k]) for layer in BACKBONE_LAYERS]
        rows.append(np.hstack(cells))
    return np.vstack(rows)


def photos(paths):
    for path in map(Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="photos or folders of photos")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--out", default="dataset/explain")
    args = parser.parse_args()
    model = load_model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for photo in photos(args.paths):
        image = cv2.imread(str(photo))      # applies EXIF orientation, like the labels
        if image is None:
            print(f"Skipping {photo}: can't read it")
            continue
        path = out / f"{photo.stem}.jpg"
        cv2.imwrite(str(path), explain(model, image, label_grid(photo)), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)


if __name__ == "__main__":
    main()
