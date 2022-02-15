import json
from pathlib import Path

import math
import torch
from matplotlib import pyplot as plt


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_Langevin(sigma, tau):
    eps = torch.randn_like(sigma)
    return math.sqrt(2.0 * tau) * sigma * eps


@torch.no_grad()
def calc_dsc(seg_fixed, seg_moving, structures_dict):
    batch_size = seg_fixed.size(0)
    DSC = torch.zeros(batch_size, len(structures_dict), device=seg_fixed.device)

    for idx, im_pair in enumerate(range(batch_size)):
        seg_fixed_im_pair = seg_fixed[idx]
        seg_moving_im_pair = seg_moving[idx]

        for structure_idx, structure in enumerate(structures_dict):
            label = structures_dict[structure]

            numerator = 2.0 * ((seg_fixed_im_pair == label) * (seg_moving_im_pair == label)).sum()
            denominator = (seg_fixed_im_pair == label).sum() + (seg_moving_im_pair == label).sum()

            try:
                score = numerator / denominator
            except:
                score = 0.0

            DSC[idx, structure_idx] = score

    return DSC


class SGLD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, curr_state, sigma, tau):
        ctx.sigma = sigma
        return add_noise_Langevin(curr_state, sigma, tau)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.sigma ** 2 * grad_output, None, None


def init_grid_im(size, spacing=2):
    if len(size) == 2:
        im = torch.zeros([1, 1, *size], dtype=torch.float)
        im[:, :, ::spacing, :] = 1
        im[:, :, :, ::spacing] = 1

        return im
    elif len(size) == 3:
        im = torch.zeros([1, 3, *size], dtype=torch.float)  # NOTE (DG): stack in the channel dimension

        im[:, 0, :, ::spacing, :] = 1
        im[:, 0, :, :, ::spacing] = 1

        im[:, 1, ::spacing, :, :] = 1
        im[:, 1, :, :, ::spacing] = 1

        im[:, 2, ::spacing, :, :] = 1
        im[:, 2, :, ::spacing, :] = 1

        return im

    raise NotImplementedError


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def plot_tensor(tensor: torch.Tensor):
    tensor_ = tensor.detach().cpu()
    fig = plt.figure()
    fig.add_subplot()
    fig.axes[0].axes.xaxis.set_visible(False)
    fig.axes[0].axes.yaxis.set_visible(False)
    fig.axes[0].imshow(tensor_[0:1, 0:1, tensor_.size(2) // 2, ...].squeeze(), cmap='gray')
    return fig
