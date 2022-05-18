import json
import math
import os
import torch

from matplotlib import pyplot as plt
from pathlib import Path


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_Langevin(sigma, tau):
    eps = torch.randn_like(sigma)
    return math.sqrt(2.0 * tau) * sigma * eps


@torch.no_grad()
def calc_DSC(seg_fixed, seg_moving, structures_dict):
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


def get_im_or_field_mid_slices_idxs(im_or_field):
    if len(im_or_field.shape) == 3:
        return int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2), int(im_or_field.shape[0] / 2)
    elif len(im_or_field.shape) == 4:
        return int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2), int(im_or_field.shape[1] / 2)
    elif len(im_or_field.shape) == 5:
        return int(im_or_field.shape[4] / 2), int(im_or_field.shape[3] / 2), int(im_or_field.shape[2] / 2)

    raise NotImplementedError


def get_im_or_field_mid_slices(im_or_field):
    mid_idxs = get_im_or_field_mid_slices_idxs(im_or_field)

    if len(im_or_field.shape) == 3:
        return [im_or_field[:, :, mid_idxs[0]],
                im_or_field[:, mid_idxs[1], :],
                im_or_field[mid_idxs[2], :, :]]
    elif len(im_or_field.shape) == 4:
        return [im_or_field[:, :, :, mid_idxs[0]],
                im_or_field[:, :, mid_idxs[1], :],
                im_or_field[:, mid_idxs[2], :, :]]
    if len(im_or_field.shape) == 5:
        return [im_or_field[:, :, :, :, mid_idxs[0]],
                im_or_field[:, :, :, mid_idxs[1], :],
                im_or_field[:, :, mid_idxs[2], :, :]]

    raise NotImplementedError


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


def save_model(args, epoch, step, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim,
               scheduler_sim_pretrain, scheduler_sim):
    path = os.path.join(args.model_dir, f'checkpoint_{epoch}.pt')
    state_dict = {'epoch': epoch, 'step': step, 'model': model.state_dict(),
                  'optimizer_enc': optimizer_enc.state_dict(), 'optimizer_dec': optimizer_dec.state_dict(),
                  'optimizer_sim_pretrain': optimizer_sim_pretrain.state_dict(), 'optimizer_sim': optimizer_sim.state_dict(),
                  'scheduler_sim_pretrain': scheduler_sim_pretrain.state_dict(), 'scheduler_sim': scheduler_sim.state_dict()}

    torch.save(state_dict, path)


def log_images(writer, step, fixed, moving, fixed_masked, moving_masked, moving_warped, grid_warped, phase=''):
    body_axes = ['coronal', 'axial', 'sagittal']

    for im_name, im in {'fixed': fixed['im'], 'moving': moving['im'],
                        'fixed_masked': fixed_masked, 'moving_masked': moving_masked,
                        'moving_warped': moving_warped, 'transformation': grid_warped}.items():
        mid_slices = get_im_or_field_mid_slices(im)

        if im_name == 'transformation':
            mid_slices[0], mid_slices[1], mid_slices[2] = mid_slices[0][:, 2:3, ...], mid_slices[1][:, 1:2, ...], mid_slices[2][:, 0:1, ...]

        for slice_idx, body_axis_name in enumerate(body_axes):
            writer.add_images(f'{phase}/{im_name}/{body_axis_name}', mid_slices[slice_idx], step)


def plot_tensor(tensor: torch.Tensor, grid=False):
    tensor_ = tensor.detach().cpu()
    fig, axes = plt.subplots(1, 3)

    for ax in axes:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if not grid:
        axes[0].imshow(tensor_[0:1, 0:1, tensor_.size(2) // 2, ...].squeeze(), cmap='gray')
        axes[1].imshow(tensor_[0:1, 0:1, ..., tensor_.size(3) // 2, ...].squeeze(), cmap='gray')
        axes[2].imshow(tensor_[0:1, 0:1, ..., tensor_.size(4) // 2].squeeze(), cmap='gray')
    else:
        mid_slices = get_im_or_field_mid_slices(tensor_)
        mid_slices[0], mid_slices[1], mid_slices[2] = mid_slices[0][:, 2:3, ...], mid_slices[1][:, 1:2, ...], mid_slices[2][:, 0:1, ...]

        for ax_idx, ax in enumerate(axes):
            ax.imshow(mid_slices[ax_idx].squeeze(), cmap='gray')

    return fig


def write_hparams(writer, config):
    hparams = ['epochs_pretrain_model', 'loss_init', 'alpha', 'reg_weight', 'lr', 'lr_sim_pretrain', 'lr_sim',
               'sim_gamma_pretrain', 'sim_step_size_pretrain', 'sim_gamma', 'sim_step_size', 'tau',
               'batch_size', 'no_samples_per_epoch', 'no_samples_SGLD', 'dims']
    hparam_dict = dict(zip(hparams, [config[hparam] for hparam in hparams]))

    for k, v in hparam_dict.items():
        if type(v) == list:
            hparam_dict[k] = torch.tensor(v)

    writer.add_hparams(hparam_dict, metric_dict={'dummy_metric': 0.0}, run_name='.')


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)