

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def get_nonspade_norm_layer():
    def get_out_channel(layer):
        if hasattr(layer, 'out_channels'):
            return getattr(layer, 'out_channels')
        return layer.weight.size(0)

    def add_norm_layer(layer):
        layer = spectral_norm(layer)
        if getattr(layer, 'bias', None) is not None:
            delattr(layer, 'bias')
            layer.register_parameter('bias', None)
        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        return nn.Sequential(layer, norm_layer)

    return add_norm_layer




class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        if opt.view_encoding_input_type == 'label':
            in_dim = opt.semantic_nc
        elif opt.view_encoding_input_type == 'pseudo_image':
            in_dim = 3
        else:
            raise ValueError()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        norm_layer = get_nonspade_norm_layer()
        self.layer1 = norm_layer(nn.Conv2d(in_dim, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
