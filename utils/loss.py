import math
import numpy as np
import torch
from torch import nn


class LCC(nn.Module):
    def __init__(self, s=1):
        super(LCC, self).__init__()

        self.sz = float((2 * s + 1) ** 3)
        self.kernel = nn.Conv3d(1, 1, kernel_size=2*s+1, stride=1, padding=s, bias=False, padding_mode='replicate')
        nn.init.ones_(self.kernel.weight)

        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, im_fixed, im_moving):
        u_F = self.kernel(im_fixed) / self.sz
        u_M = self.kernel(im_moving) / self.sz

        cross = self.kernel((im_fixed - u_F) * (im_moving - u_M))
        var_F = self.kernel((im_fixed - u_F) ** 2)
        var_M = self.kernel((im_moving - u_M) ** 2)

        z = cross * cross / (var_F * var_M + 1e-5)
        return -1.0 * z.mean(dim=(1, 2, 3, 4))


class MI(nn.Module):
    def __init__(self, no_bins=64, normalised=True, sample_ratio=0.1, vmin=0.0, vmax=1.0):
        super(MI, self).__init__()

        self.normalised = normalised
        self.sample_ratio = sample_ratio

        bins = torch.linspace(vmin, vmax, no_bins).unsqueeze(1)
        self.register_buffer('bins', bins, persistent=False)

        # set the std. dev. of the Gaussian kernel so that FWHM is one bin width
        bin_width = (vmax - vmin) / no_bins
        self.sigma = bin_width * 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

    def __joint_prob(self, im_fixed, im_moving):
        # compute the Parzen window function response
        win_F = torch.exp(-0.5 * (im_fixed - self.bins) ** 2 / (self.sigma ** 2)) / (math.sqrt(2 * math.pi) * self.sigma)
        win_M = torch.exp(-0.5 * (im_moving - self.bins) ** 2 / (self.sigma ** 2)) / (math.sqrt(2 * math.pi) * self.sigma)

        # compute the histogram
        hist = win_F.bmm(win_M.transpose(1, 2))

        # normalise the histogram to get the joint distr.
        hist_normalised = hist.flatten(start_dim=1, end_dim=-1).sum(dim=1) + 1e-5
        return hist / hist_normalised.view(-1, 1, 1)

    def forward(self, x, y):
        if self.sample_ratio < 1.0:
            # random spatial sampling with the same number of pixels/voxels chosen for every sample in the batch
            numel_ = np.prod(x.size()[2:])
            idx_th = int(self.sample_ratio * numel_)
            idx_choice = torch.randperm(int(numel_))[:idx_th]

            x = x.view(x.size()[0], 1, -1)[:, :, idx_choice]
            y = y.view(y.size()[0], 1, -1)[:, :, idx_choice]

        # make sure the sizes are (N, 1, prod(sizes))
        x = x.flatten(start_dim=2, end_dim=-1)
        y = y.flatten(start_dim=2, end_dim=-1)

        # compute joint distribution
        p_joint = self._compute_joint_prob(x, y)

        # marginalise the joint distribution to get marginal distributions, (batch_size, x_bins, y_bins)
        p_x = torch.sum(p_joint, dim=2)
        p_y = torch.sum(p_joint, dim=1)

        # calculate entropy
        ent_x = - torch.sum(p_x * torch.log(p_x + 1e-5), dim=1)  # (N,1)
        ent_y = - torch.sum(p_y * torch.log(p_y + 1e-5), dim=1)  # (N,1)
        ent_joint = - torch.sum(p_joint * torch.log(p_joint + 1e-5), dim=(1, 2))  # (N,1)

        if self.normalised:
            return -1.0 * torch.mean((ent_x + ent_y) / ent_joint)

        return -1.0 * torch.mean(ent_x + ent_y - ent_joint)


def SSD(im_fixed, im_moving, mask=None, reduction='mean'):
    z = (im_fixed - im_moving) ** 2

    if mask is not None:
        z = z[mask]

    if reduction == 'mean':
        return z.mean()
    elif reduction == 'sum':
        return z.sum()

    raise NotImplementedError
