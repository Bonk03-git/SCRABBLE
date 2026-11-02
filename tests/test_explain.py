import numpy as np
import torch

from scrabble.corners import INPUT, CornerNet
from scrabble.explain import explain, grad_cam, occlusion, peak_cells


class OneSpotNet(torch.nn.Module):
    """Every corner's output depends only on the input's top-left quarter."""

    def forward(self, x):
        value = x[:, :, :INPUT // 2, :INPUT // 2].mean(dim=(1, 2, 3))
        heat = torch.zeros(len(x), 4, 64, 64)
        heat[:, :, 5, 5] = 1 + value[:, None]
        return heat


def test_occlusion_only_marks_the_place_the_output_depends_on():
    drops = occlusion(OneSpotNet(), torch.ones(1, 3, INPUT, INPUT))

    assert drops.shape == (4, INPUT, INPUT)
    assert drops[0, 20, 20] > 0
    assert drops[0, INPUT - 20, INPUT - 20] == 0


def test_grad_cam_gives_one_scaled_map_per_corner():
    torch.manual_seed(0)
    model = CornerNet(pretrained=False).eval()

    cams = grad_cam(model, torch.rand(1, 3, INPUT, INPUT), "layer3")

    assert cams.shape == (4, INPUT, INPUT)
    assert cams.min() >= 0 and cams.max() <= 1
    assert not any(p.grad is not None for p in model.parameters())


def test_peak_cells_finds_each_corner():
    heat = torch.zeros(4, 64, 64)
    for k in range(4):
        heat[k, k, 10 + k] = 1

    assert [tuple(map(int, p)) for p in peak_cells(heat)] == [(k, 10 + k) for k in range(4)]


def test_explain_makes_a_sheet_with_a_row_per_corner():
    torch.manual_seed(0)
    image = np.random.default_rng(0).integers(0, 255, (300, 400, 3), np.uint8)
    grid = [[50, 50], [350, 50], [350, 250], [50, 250]]

    sheet = explain(CornerNet(pretrained=False).eval(), image, grid)

    assert sheet.shape == (4 * 320, 5 * 320, 3)
