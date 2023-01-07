import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch.nn.utils import spectral_norm

from .loss import MI


def cubic_B_spline_1D_value(x):
    """
    evaluate a 1D cubic B-spline
    """

    t = abs(x)

    if t >= 2:  # outside the local support region
        return 0

    if t < 1:
        return 2.0 / 3.0 + (0.5 * t - 1.0) * t ** 2

    return -1.0 * ((t - 2.0) ** 3) / 6.0


def B_spline_1D_kernel(stride):
    kernel = torch.ones(4 * stride - 1)
    radius = kernel.shape[0] // 2

    for i in range(kernel.shape[0]):
        kernel[i] = cubic_B_spline_1D_value((i - radius) / stride)

    return kernel


def conv1D(x, kernel, dim=-1, stride=1, dilation=1, padding=0, transpose=False):
    """
    convolve data with 1-dimensional kernel along specified dimension
    """

    x = x.type(kernel.dtype)  # (N, ndim, *sizes)
    x = x.transpose(dim, -1)  # (N, ndim, *other_sizes, sizes[dim])
    shape_ = x.size()

    # reshape into channel (N, ndim * other_sizes, sizes[dim])
    groups = int(torch.prod(torch.tensor(shape_[1:-1])))
    weight = kernel.expand(groups, 1, kernel.shape[-1])  # (ndim*other_sizes, 1, kernel_size)
    x = x.reshape(shape_[0], groups, shape_[-1])  # (N, ndim*other_sizes, sizes[dim])
    conv_fn = F.conv_transpose1d if transpose else F.conv1d

    x = conv_fn(x, weight, stride=stride, dilation=dilation, padding=padding, groups=groups)
    x = x.reshape(shape_[0:-1] + x.shape[-1:])  # (N, ndim, *other_sizes, size[dim])

    return x.transpose(-1, dim)  # (N, ndim, *sizes)


class Cubic_B_spline_FFD_3D(nn.Module):
    def __init__(self, dims, cps):
        """
        compute dense velocity field of the cubic B-spline FFD transformation model from input control point parameters
        :param cps: control point spacing
        """

        super(Cubic_B_spline_FFD_3D, self).__init__()

        self.dims = dims
        self.stride = cps
        self.kernels, self.padding = list(), list()

        for idx, s in enumerate(self.stride):
            kernel = B_spline_1D_kernel(s)
            
            self.register_buffer('kernel' + str(idx), kernel, persistent=False)
            self.kernels.append(getattr(self, 'kernel' + str(idx)))
            self.padding.append((len(kernel) - 1) // 2)

    def forward(self, v):
        # compute B-spline tensor product via separable 1D convolutions
        for i, (k, s, p) in enumerate(zip(self.kernels, self.stride, self.padding)):
            v = conv1D(v, dim=i + 2, kernel=k, stride=s, padding=p, transpose=True)

        #  crop the output to image size
        slicer = (slice(0, v.shape[0]), slice(0, v.shape[1])) + tuple(slice(s, s + self.dims[i]) for i, s in enumerate(self.stride))
        return v[slicer]


def transform(src, grid, interpolation='bilinear', padding='border'):
    return F.grid_sample(src, grid, mode=interpolation, align_corners=True, padding_mode=padding)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, s=1, activation_fn=nn.Identity()):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernel_size=2 * s + 1, stride=1, padding=s, padding_mode='replicate')
        self.activation_fn = activation_fn

        with torch.no_grad():
            self.main.weight.multiply_(1e-7)
            self.main.weight[..., s, s, s] = F.normalize(torch.rand_like(self.main.weight[..., s, s, s]), p=1, dim=0)

            nn.init.zeros_(self.main.bias)
            self.main.bias.add_(torch.randn_like(self.main.bias).multiply(1e-7))

    def forward(self, x):
        out = self.main(x)
        return self.activation_fn(out)


class UNetEncoder(nn.Module):
    def __init__(self, no_features, s=1, activation_fn=nn.Identity()):
        super(UNetEncoder, self).__init__()
        self.enc = nn.ModuleList()
        self.no_features = no_features
        prev_no_features = 1

        for no_features in self.no_features:
            self.enc.append(ConvBlock(prev_no_features, no_features, s=s, activation_fn=activation_fn))
            prev_no_features = no_features

    def forward(self, im):
        im_enc = [im]

        for layer in self.enc:
            im_enc.append(layer(im_enc[-1]))

        return im_enc[-1]


class BaseModel(nn.Module):
    """
    base class for all models
    """

    def __init__(self, input_size, no_features=None):
        super(BaseModel, self).__init__()
        
        activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)

        self.no_features = no_features
        self.enc = UNetEncoder(no_features, activation_fn=activation_fn)
        self.agg = nn.Conv1d(no_features[-1], 1, kernel_size=1, stride=1, bias=False)

        with torch.no_grad():
            nn.init.ones_(self.agg.weight)
            self.agg.weight.add_(torch.randn_like(self.agg.weight).multiply(1e-7))
        
        default_mask = torch.ones((1, 1, *input_size), dtype=torch.bool)
        self.register_buffer('mask', default_mask, persistent=False)
        self.disable_grads()

    def enable_grads(self):
        for param in self.parameters():
            param.requires_grad_(True)

        self.train()

    def disable_grads(self):
        for param in self.parameters():
            param.requires_grad_(False)

        self.eval()

    @abstractmethod
    def encode(self, im_fixed, im_moving):
        pass

    def feature_extraction(self, im_fixed, im_moving):
        z_fixed = self.enc(im_fixed)
        N, C, D, H, W = z_fixed.shape
        z_fixed = self.agg(z_fixed.view(N, C, -1)).view(N, 1, D, H, W)

        z_moving = self.enc(im_moving)
        z_moving = self.agg(z_moving.view(N, C, -1)).view(N, 1, D, H, W)

        return z_fixed, z_moving

    def forward(self, input, mask=None, reduction='mean'):
        """
        forward pass logic

        :return: Model output
        """

        im_fixed = input[:, 1:2]
        im_moving = input[:, 0:1]

        z_fixed, z_moving = self.feature_extraction(im_fixed, im_moving)
        z = self.encode(z_fixed, z_moving)

        if mask is None:
            mask = self.mask

        return z.sum(dim=(1, 2, 3, 4)) / mask.sum(dim=(1, 2, 3, 4))

    def __str__(self):
        """
        model prints with number of trainable parameters
        """

        model_parameters = list(self.parameters())
        no_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\ntrainable parameters: {}\n'.format(no_trainable_params)


class SimilarityMetricOld(BaseModel):
    def __init__(self, input_size, no_features=None):
        super(SimilarityMetricOld, self).__init__(input_size, no_features)

    def encode(self, im_fixed, im_moving):
        return (im_fixed - im_moving) ** 2


class Model(ABC):
    def enable_grads(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def disable_grads(self):
        for p in self.parameters():
            p.requires_grad_(False)

    
class SimilarityMetric(nn.Module, Model):
    def __init__(self, activation_fn="tanh", enable_spectral_norm=True, init='ssd', use_bias=True, use_strided_conv=True):
        super(SimilarityMetric, self).__init__()

        if activation_fn == 'tanh':
            self.activation_fn = lambda x: torch.tanh(x)
        elif activation_fn == 'leaky_relu':
            self.activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)
        elif activation_fn == 'none':
            self.activation_fn = lambda x: x

        f = lambda x: spectral_norm(x) if enable_spectral_norm else x
        self.stride = 2 if use_strided_conv else 1

        self.conv1 = f(nn.Conv3d(1, 4, kernel_size=3, padding=1, bias=use_bias))
        self.conv2 = f(nn.Conv3d(4, 4, kernel_size=3, padding=1, stride=self.stride, bias=use_bias))
        self.conv3 = f(nn.Conv3d(4, 8, kernel_size=3, padding=1, bias=use_bias))
        self.conv4 = f(nn.Conv3d(8, 8, kernel_size=3, padding=1, stride=self.stride, bias=use_bias))
        self.conv5 = f(nn.Conv3d(8, 16, kernel_size=3, padding=1, bias=use_bias))
        self.conv6 = f(nn.Conv3d(16, 16, kernel_size=3, padding=1, stride=self.stride, bias=use_bias))
        
        self.init = init

        if self.init == 'mi':
            self.mi_module = MI()

    def _forward(self, input):
        y1 = self.activation_fn(self.conv1(input))
        y2 = self.activation_fn(self.conv2(y1))
        y3 = self.activation_fn(self.conv3(y2))
        y4 = self.activation_fn(self.conv4(y3))
        y5 = self.activation_fn(self.conv5(y4))
        y6 = self.activation_fn(self.conv6(y5))
        
        return y6

    def forward(self, input, mask=None, reduction='mean'):
        im_fixed = input[:, 1:2]
        im_moving_warped = input[:, 0:1]
        
        if self.init == 'ssd':
            diff = (im_fixed - im_moving_warped) ** 2
            z = self._forward(diff)
        elif self.init == 'mi':
            z_fixed, z_moving_warped = self._forward(im_fixed), self._forward(im_moving_warped)
            z = self.mi_module(z_fixed, z_moving_warped)

        if reduction == 'mean':
            return z.mean()
        elif reduction == 'sum':
            return z.sum()

        raise NotImplementedError


class Encoder(nn.Module, Model):
    def __init__(self):
        super(Encoder, self).__init__()

        input_channels, no_dims = 2, 3
        use_bias = True

        self.activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)

        self.conv1 = nn.Conv3d(input_channels, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)
        self.conv5 = nn.Conv3d(32, 32, kernel_size=3, padding=1, stride=2, bias=use_bias)

    def forward(self, x):
        x2 = self.activation_fn(self.conv1(x))
        x3 = self.activation_fn(self.conv2(x2))
        x4 = self.activation_fn(self.conv3(x3))
        x5 = self.activation_fn(self.conv4(x4))
        x6 = self.activation_fn(self.conv5(x5))

        return x6, x5, x4, x3, x2


class Decoder(nn.Module, Model):
    def __init__(self, input_size, cps=None):
        super(Decoder, self).__init__()

        self.activation_fn = lambda x: F.leaky_relu(x, negative_slope=0.2)
        self.cps = cps
        self.register_buffer('grid', self.get_normalized_grid(input_size))

        input_channels, no_dims = 2, 3
        use_bias = True

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv1 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=use_bias)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=use_bias)

        if self.cps is not None:
            self.evaluate_cubic_bspline_ffd = Cubic_B_spline_FFD_3D(dims=input_size, cps=cps)
            self.conv4 = nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=use_bias)
        else:
            self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            self.conv4 = nn.Conv3d(48, 32, kernel_size=3, padding=1, bias=use_bias)

        self.conv5 = nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv6 = nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=use_bias)
        self.conv7 = nn.Conv3d(16, no_dims, kernel_size=3, padding=1, bias=use_bias)

    def compute_disp(self, flow):
        # rescale flow to range of normalized grid
        size = flow.shape[2:]
        ndim = len(size)

        for i in range(ndim):
            flow[:, i] = 2. * flow[:, i] / (2. * size[i] - 1)

        # integrate velocity field
        disp = self.integrate(flow, 6)
        disp_inv = self.integrate(flow * -1, 6)

        return disp, disp_inv

    def get_normalized_grid(self, size):
        ndim = len(size)
        grid = torch.stack(torch.meshgrid([torch.arange(s) for s in size]), 0).float()

        for i in range(ndim):
            grid[i] = 2. * grid[i] / (size[i] - 1) - 1.

        return grid.unsqueeze(0)

    def integrate(self, vel, nb_steps):
        disp = vel / (2 ** nb_steps)

        for _ in range(nb_steps):
            warped_disp = transform(disp, self.move_grid_dims(self.grid + disp), padding='border')
            disp = disp + warped_disp

        return disp

    def moveaxis(self, x, src, dst):
        ndims = x.dim()
        dims = list(range(ndims))
        dims.pop(src)

        if dst < 0:
            dst = ndims + dst
        dims.insert(dst, src)

        return x.permute(dims)

    def move_grid_dims(self, grid):
        size = grid.shape[2:]
        ndim = len(size)
        assert ndim == grid.shape[1]

        grid = self.moveaxis(grid, 1, -1)  # [batch, z?, y, x, dims]
        grid = grid[..., list(range(ndim))[::-1]]  # reverse dims

        return grid

    def regularizer(self, penalty='l2'):
        dy = torch.abs(self.disp[:, :, 1:, :, :] - self.disp[:, :, :-1, :, :])
        dx = torch.abs(self.disp[:, :, :, 1:, :] - self.disp[:, :, :, :-1, :])
        dz = torch.abs(self.disp[:, :, :, :, 1:] - self.disp[:, :, :, :, :-1])

        if penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        return d / 3.0

    def enable_grads_T(self):
        self.T.requires_grad_(True)

    def warp_image(self, img, disp=None, interpolation='bilinear', padding='border'):
        if disp is None:
            return transform(img, self.T, interpolation=interpolation, padding=padding)

        T = self.move_grid_dims(self.grid + disp)
        return transform(img, T, interpolation=interpolation, padding=padding)
       

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        wrp = transform(img, self.T_inv, interpolation=interpolation, padding=padding)
        return wrp

    def forward(self, x, x6, x5, x4, x3, x2):
        x5 = torch.cat([self.up1(x6), x5], dim=1)
        x5 = self.activation_fn(self.conv1(x5))
        x4 = torch.cat([self.up2(x5), x4], dim=1)
        x4 = self.activation_fn(self.conv2(x4))
        x3 = torch.cat([self.up3(x4), x3], dim=1)
        x3 = self.activation_fn(self.conv3(x3))
        x2 = x3 if self.cps is not None else torch.cat([self.up4(x3), x2], dim=1)
        x2 = self.activation_fn(self.conv4(x2))
        x1 = self.activation_fn(self.conv5(x2))
        x1 = self.activation_fn(self.conv6(x1))
        flow = self.conv7(x1)

        if self.cps is not None:
            flow = self.evaluate_cubic_bspline_ffd(flow)
        
        disp, disp_inv = self.compute_disp(flow)

        self.register_buffer('disp', disp, persistent=False)
        self.register_buffer('disp_inv', disp_inv, persistent=False)
        
        T2 = self.grid + disp  # ugly hack
        T = self.move_grid_dims(T2)
        T_inv = self.move_grid_dims(self.grid + disp_inv)

        self.register_buffer('T', T, persistent=False)
        self.register_buffer('T2', T2, persistent=False)
        self.register_buffer('T_inv', T_inv, persistent=False)

        # extract first channel for warping
        im = x.narrow(dim=1, start=0, length=1)

        # warp image
        return self.warp_image(im)


class UNet(nn.Module):
    def __init__(self, input_size, activation_fn_sim='tanh', cps=None, enable_spectral_norm=False, init='ssd', old=False, use_strided_conv=True):
        super(UNet, self).__init__()

        encoder, decoder = Encoder(), Decoder(input_size, cps=cps)

        if old:
            sim = SimilarityMetricOld(input_size=input_size, no_features=[4, 8, 8])
        else:
            sim = SimilarityMetric(activation_fn=activation_fn_sim, enable_spectral_norm=enable_spectral_norm, init=init, use_strided_conv=use_strided_conv)

        self.submodules = nn.ModuleDict({'enc': encoder, 'dec': decoder, 'sim': sim})

    def forward(self, x):
        x6, x5, x4, x3, x2 = self.submodules['enc'](x)
        return self.submodules['dec'](x, x6, x5, x4, x3, x2)

    def get_disp(self):
        return self.submodules['dec'].disp

    def get_disp_inv(self):
        return self.submodules['dec'].disp_inv

    def get_T(self):
        return self.submodules['dec'].T

    def get_T2(self):
        return self.submodules['dec'].T2

    def get_T_inv(self):
        return self.submodules['dec'].T_inv

    @property
    def no_params(self):
        no_params_sim = sum(p.numel() for p in self.submodules['sim'].parameters() if p.requires_grad)
        no_params_enc = sum(p.numel() for p in self.submodules['enc'].parameters() if p.requires_grad)
        no_params_dec = sum(p.numel() for p in self.submodules['dec'].parameters() if p.requires_grad)

        return no_params_sim, no_params_enc, no_params_dec

    def regularizer(self):
        return self.submodules['dec'].regularizer()

    def transform(self, src, grid, interpolation='bilinear', padding='border'):
        return self.submodules['dec'].transform(src, grid, interpolation=interpolation, padding=padding)

    def warp_image(self, img, disp=None, interpolation='bilinear', padding='border'):
        return self.submodules['dec'].warp_image(img, interpolation=interpolation, padding=padding)

    def warp_inv_image(self, img, interpolation='bilinear', padding='border'):
        return self.submodules['dec'].warp_inv_image(img, interpolation=interpolation, padding=padding)
