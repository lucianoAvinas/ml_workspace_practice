## Code very slightly modified from https://github.com/jvanvugt/pytorch-unet

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, init_exp, max_exp, normalization, use_bilinear):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            init_exp (int): number of filters in the first layer is 2**init_exp
            max_exp (int): max number of filters is 2**max_exp
            normalization (str): one of 'none', 'batch', or 'instance'
                            'none' will use no normalization after activation.
                           'batch' will use BatchNorm after activation.
                           'instance' will use InstanceNorm after activation.
            use_bilinear (bool): If true up modules will use bilinear 
                                 upsampling, else transposed convolutions will
                                 be used for learned upsampling.
        """
        super(UNet, self).__init__()
        assert normalization in ('none', 'batch', 'instance')
        self.depth = depth
        prev_features = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            curr_exp = min(init_exp + i, max_exp)
            self.down_path.append(
                UNetConvBlock(prev_features, 2 ** curr_exp, normalization)
            )
            prev_features = 2 ** curr_exp

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            curr_exp = min(init_exp + i, max_exp)
            self.up_path.append(
                UNetUpBlock(prev_features, 2 ** curr_exp, use_bilinear, normalization)
            )
            prev_features = 2 ** curr_exp

        self.last = nn.Conv2d(prev_features, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for down in self.down_path[:-1]:
            x = down(x)
            blocks.append(x)
            x = F.max_pool2d(x, 2)

        x = self.down_path[-1](x)
        
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, normalization):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        if normalization == 'batch':
            block.append(nn.BatchNorm2d(out_size))
        elif normalization == 'instance':
            block.append(nn.InstanceNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1))
        block.append(nn.ReLU())
        if normalization == 'batch':
            block.append(nn.BatchNorm2d(out_size))
        elif normalization == 'instance':
            block.append(nn.Instance2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, use_bilinear, no_padding, normalization):
        super(UNetUpBlock, self).__init__()
        if use_bilinear:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        self.conv_block = UNetConvBlock(in_size, out_size, normalization)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), 
                     diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out