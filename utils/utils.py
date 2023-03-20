import json
import math
import numpy as np
import os
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt
from pathlib import Path
from torch import nn


class GradientOperator(nn.Module):
    def __init__(self):
        super(GradientOperator, self).__init__()

        # paddings
        self.px = [0, 1, 0, 0, 0, 0]
        self.py = [0, 0, 0, 1, 0, 0]
        self.pz = [0, 0, 0, 0, 0, 1]

        # F.grid_sample(..) takes values in range (-1, 1), so needed for det(J) = 1 when the transformation is identity
        self.pixel_spacing = None

    def _set_spacing(self, field):
        dims = np.asarray(field.shape[2:])
        self.pixel_spacing = 2.0 / (dims - 1.0)

    def forward(self, field):
        self._set_spacing(field)

        # forward differences
        d_dx = F.pad(field[:, :, :, :, 1:] - field[:, :, :, :, :-1], self.px, mode='replicate')
        d_dy = F.pad(field[:, :, :, 1:] - field[:, :, :, :-1], self.py, mode='replicate')
        d_dz = F.pad(field[:, :, 1:] - field[:, :, :-1], self.pz, mode='replicate')

        d_dx /= self.pixel_spacing[2]
        d_dy /= self.pixel_spacing[1]
        d_dz /= self.pixel_spacing[0]

        nabla_x = torch.stack((d_dx[:, 0], d_dy[:, 0], d_dz[:, 0]), 1)
        nabla_y = torch.stack((d_dx[:, 1], d_dy[:, 1], d_dz[:, 1]), 1)
        nabla_z = torch.stack((d_dx[:, 2], d_dy[:, 2], d_dz[:, 2]), 1)

        return torch.stack([nabla_x, nabla_y, nabla_z], dim=-1)


def add_noise_Langevin(field, sigma, tau):
    return field + get_noise_Langevin(sigma, tau)


def get_noise_Langevin(sigma, tau):
    eps = torch.randn_like(sigma)
    return math.sqrt(2.0 * tau) * sigma * eps


@torch.no_grad()
def calc_dsc(seg_fixed, seg_moving, structures_dict):
    batch_size = seg_fixed.size(0)
    dsc = torch.zeros(batch_size, len(structures_dict), device=seg_fixed.device)

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

            dsc[idx, structure_idx] = score

    return dsc


@torch.no_grad()
def calc_asd(seg_fixed, seg_moving, spacing, structures_dict):
    batch_size = seg_fixed.size(0)
    asd = torch.zeros(batch_size, len(structures_dict))

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    for idx, im_pair in enumerate(range(batch_size)):
        for structure_idx, structure_name in enumerate(structures_dict):
            label = structures_dict[structure_name]

            seg_fixed_arr = seg_fixed[idx].squeeze().cpu().numpy()
            seg_moving_arr = seg_moving[idx].squeeze().cpu().numpy()

            seg_fixed_structure = np.where(seg_fixed_arr == label, 1, 0)
            seg_moving_structure = np.where(seg_moving_arr == label, 1, 0)

            seg_fixed_im = sitk.GetImageFromArray(seg_fixed_structure)
            seg_moving_im = sitk.GetImageFromArray(seg_moving_structure)

            seg_fixed_im.SetSpacing(spacing)
            seg_moving_im.SetSpacing(spacing)

            seg_fixed_contour = sitk.LabelContour(seg_fixed_im)
            seg_moving_contour = sitk.LabelContour(seg_moving_im)

            hausdorff_distance_filter.Execute(seg_fixed_contour, seg_moving_contour)
            asd[idx, structure_idx] = hausdorff_distance_filter.GetAverageHausdorffDistance()

    return asd


@torch.no_grad()
def calc_det_J(nabla):
    """
    calculate the Jacobian determinant of a vector field
    :param nabla: field gradients
    :return: Jacobian determinant
    """

    # _, N, D, H, W, _ = nabla.shape
    # Jac = nabla.permute([0, 2, 3, 4, 1, 5]).reshape([-1, N, N])
    # return torch.det(Jac).reshape(-1, D, H, W)  # NOTE (DG): for some reason causes an illegal memory access

    nabla_x = nabla[..., 0]
    nabla_y = nabla[..., 1]
    nabla_z = nabla[..., 2]

    det_J = nabla_x[:, 0] * nabla_y[:, 1] * nabla_z[:, 2] + \
            nabla_y[:, 0] * nabla_z[:, 1] * nabla_x[:, 2] + \
            nabla_z[:, 0] * nabla_x[:, 1] * nabla_y[:, 2] - \
            nabla_x[:, 2] * nabla_y[:, 1] * nabla_z[:, 0] - \
            nabla_y[:, 2] * nabla_z[:, 1] * nabla_x[:, 0] - \
            nabla_z[:, 2] * nabla_x[:, 1] * nabla_y[:, 0]
    det_J *= -1.0  # NOTE (DG): ugly hack

    return det_J


@torch.no_grad()
def calc_no_non_diffeomorphic_voxels(transformation, diff_op=GradientOperator()):
    nabla = diff_op(transformation)
    log_det_J_transformation = torch.log(calc_det_J(nabla))
    return torch.sum(torch.isnan(log_det_J_transformation), dim=(1, 2, 3))


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


def to_device(DEVICE, fixed, moving):
    for key in fixed:
        fixed[key] = fixed[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)
        moving[key] = moving[key].to(DEVICE, memory_format=torch.channels_last_3d, non_blocking=True)

    return fixed, moving


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


def save_model(args, epoch, step, model, optimizer_enc, optimizer_dec, optimizer_sim_pretrain, optimizer_sim):
    path = os.path.join(args.model_dir, f'checkpoint_{epoch}.pt')
    state_dict = {'epoch': epoch, 'step': step, 'model': model.state_dict(),
                  'optimizer_enc': optimizer_enc.state_dict(), 'optimizer_dec': optimizer_dec.state_dict(),
                  'optimizer_sim_pretrain': optimizer_sim_pretrain.state_dict(), 'optimizer_sim': optimizer_sim.state_dict()}

    torch.save(state_dict, path)


def write_hparams(writer, config):
    hparams = ['epochs_pretrain_model', 'loss_init', 'alpha', 'reg_weight', 'lr', 'lr_sim_pretrain', 'lr_sim', 'tau',
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
