# CycleGAN Patch Discriminator
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

import torch
from torch import nn
import torch.nn.functional as F


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, max_exp, ndf, n_layers, norm_layer):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer (str)     -- normalization layer. choices in ['batch', 'instance']
        """
        super(NLayerDiscriminator, self).__init__()
        if norm_layer == 'batch':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            raise Exception('not supported norm layer')

        use_bias = (norm_layer == nn.InstanceNorm2d)

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = 2**min(n, max_exp)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = 2**min(n_layers, max_exp)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)