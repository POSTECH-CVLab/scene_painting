import torch.nn as nn
import models.norms as norms
import torch
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

import math
import numpy as np

class PosEncodingNeRF(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, num_frequencies=10, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = num_frequencies
        elif self.in_features == 2:
            self.num_frequencies = num_frequencies
#            assert sidelength is not None
#            if isinstance(sidelength, int):
#                sidelength = (sidelength, sidelength)
#            self.num_frequencies = 4
#            if use_nyquist:
#                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        #coords = coords.view(coords.shape[0], -1, self.in_features)
        img_size = coords.shape[-1]
        coords = coords.reshape(coords.shape[0], self.in_features, -1)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[:, j, :]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), 1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), 1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=1)

        return coords_pos_enc.reshape(coords.shape[0], self.out_dim, img_size, img_size)


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        if not self.opt.no_pos_encoding:
            self.pos_encoding = PosEncodingNeRF(in_features=3)
            self.coordinate_embedding = nn.Conv2d(3, self.pos_encoding.out_dim, kernel_size=1)

            in_dim = self.pos_encoding.out_dim * 2 + opt.semantic_nc + self.opt.z_dim
        else:
            self.pos_encoding = None
            in_dim = 3 + opt.semantic_nc + self.opt.z_dim

        hdim = opt.hidden_dim
        
        self.num_layers = opt.G_num_layers
        self.mlp = []
        self.rgb_mlp = []
        if not opt.use_G_spectral_norm:
            self.first_mlp = nn.Sequential(
                nn.Conv2d(in_dim, hdim, kernel_size=1),
                nn.LeakyReLU(inplace=True),)

            for i in range(self.num_layers):
                self.mlp.append(nn.Sequential(
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                ))

            for i in range(self.num_layers):
                self.rgb_mlp.append(
                    nn.Conv2d(hdim, 3, kernel_size=1)
                )

        else:
            self.mlp.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_dim, hdim, kernel_size=1)),
                nn.LeakyReLU(inplace=True),))
            for i in range(12):
                self.mlp.append(nn.Sequential(
                    spectral_norm(nn.Conv2d(hdim, hdim, kernel_size=1)),
                    nn.LeakyReLU(inplace=True),))
            self.mlp.append(
                nn.Conv2d(hdim, 3, kernel_size=1))

        self.mlp = nn.ModuleList(self.mlp)
        self.rgb_mlp = nn.ModuleList(self.rgb_mlp)

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, coord_image, z=None):

        seg = input

        if not self.opt.no_pos_encoding:
            coord_embedding = self.coordinate_embedding(coord_image)
            pos_encoding = self.pos_encoding(coord_image)
            coord_image = torch.cat([coord_embedding, pos_encoding], dim=1)

            #coord_image = self.pos_encoding(coord_image)

        #x = torch.cat([seg, coord_image], dim=1)

        if z is None:
            dev = seg.device
            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))

        x = torch.cat([z, seg, coord_image], dim=1)

        #x = self.mlp(x)
        #x = torch.tanh(x)
        
        x = self.first_mlp(x)

        x = self.mlp[0](x)
        rgb = self.rgb_mlp[0](x)

        for i in range(1, self.num_layers):
            x = self.mlp[i](x)
            cur_rgb = self.rgb_mlp[i](x)
            rgb = rgb + cur_rgb

        x = torch.tanh(rgb)
        return x


class Generator_Without_Label(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        if not self.opt.no_pos_encoding:
            self.pos_encoding = PosEncodingNeRF(in_features=2, sidelength=10, use_nyquist=False)
            self.coordinate_embedding = nn.Conv2d(2, self.pos_encoding.out_dim, kernel_size=1)

            in_dim = self.pos_encoding.out_dim * 2 + self.opt.z_dim
        else:
            self.pos_encoding = None
            in_dim = 2 + self.opt.z_dim

        hdim = opt.hidden_dim
        
        self.num_layers = opt.G_num_layers
        self.mlp = []
        self.rgb_mlp = []
        if not opt.use_G_spectral_norm:
            self.first_mlp = nn.Sequential(
                nn.Conv2d(in_dim, hdim, kernel_size=1),
                nn.LeakyReLU(inplace=True),)

            for i in range(self.num_layers):
                self.mlp.append(nn.Sequential(
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(hdim, hdim, kernel_size=1),
                    nn.LeakyReLU(inplace=True),
                ))

            for i in range(self.num_layers):
                self.rgb_mlp.append(
                    nn.Conv2d(hdim, 3, kernel_size=1)
                )

        else:
            pass
#            self.mlp.append(nn.Sequential(
#                spectral_norm(nn.Conv2d(in_dim, hdim, kernel_size=1)),
#                nn.LeakyReLU(inplace=True),))
#            for i in range(12):
#                self.mlp.append(nn.Sequential(
#                    spectral_norm(nn.Conv2d(hdim, hdim, kernel_size=1)),
#                    nn.LeakyReLU(inplace=True),))
#            self.mlp.append(
#                nn.Conv2d(hdim, 3, kernel_size=1))

        self.mlp = nn.ModuleList(self.mlp)
        self.rgb_mlp = nn.ModuleList(self.rgb_mlp)

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def forward(self, input, coord_image, z=None):

        if not self.opt.no_pos_encoding:
            coord_embedding = self.coordinate_embedding(coord_image)
            pos_encoding = self.pos_encoding(coord_image)
            coord_image = torch.cat([coord_embedding, pos_encoding], dim=1)

        if z is None:
            dev = coord_image.device
            z = torch.randn(coord_image.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
            z = z.expand(z.size(0), self.opt.z_dim, coord_image.size(2), coord_image.size(3))

        x = torch.cat([z, coord_image], dim=1)

        #x = self.mlp(x)
        #x = torch.tanh(x)
        
        x = self.first_mlp(x)

        x = self.mlp[0](x)
        rgb = self.rgb_mlp[0](x)

        for i in range(1, self.num_layers):
            x = self.mlp[i](x)
            cur_rgb = self.rgb_mlp[i](x)
            rgb = rgb + cur_rgb

        x = torch.tanh(rgb)
        return x

#class OASIS_SPADE_Generator(nn.Module):
#    def __init__(self, opt):
#        super().__init__()
#        self.opt = opt
#        sp_norm = norms.get_spectral_norm(opt)
#        ch = opt.channels_G
#        self.channels = [16*ch, 16*ch, 16*ch, 8*ch, 4*ch, 2*ch, 1*ch]
#        self.init_W, self.init_H = self.compute_latent_vector_size(opt)
#        self.conv_img = nn.Conv2d(self.channels[-1], 3, 3, padding=1)
#        self.up = nn.Upsample(scale_factor=2)
#        self.body = nn.ModuleList([])
#        for i in range(len(self.channels)-1):
#            self.body.append(ResnetBlock_with_SPADE(self.channels[i], self.channels[i+1], opt))
#        if not self.opt.no_3dnoise:
#            self.fc = nn.Conv2d(self.opt.semantic_nc + self.opt.z_dim, 16 * ch, 3, padding=1)
#        else:
#            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * ch, 3, padding=1)
#
#    def compute_latent_vector_size(self, opt):
#        w = opt.crop_size // (2**(opt.num_res_blocks-1))
#        h = round(w / opt.aspect_ratio)
#        return h, w
#
#    def forward(self, input, coord_image, z=None):
#        seg = input
#        if self.opt.gpu_ids != "-1":
#            seg.cuda()
#        if not self.opt.no_3dnoise:
#            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
#            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
#            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
#            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))
#            seg = torch.cat((z, seg), dim = 1)
#
#        x = torch.cat([x, coord_image], dim=1)
#        x = F.interpolate(seg, size=(self.init_W, self.init_H))
#        x = self.fc(x)
#        for i in range(self.opt.num_res_blocks):
#            x = self.body[i](x, seg)
#            if i < self.opt.num_res_blocks-1:
#                x = self.up(x)
#        x = self.conv_img(F.leaky_relu(x, 2e-1))
#        x = torch.tanh(x)
#        return x


class ResnetBlock_with_SPADE(nn.Module):
    def __init__(self, fin, fout, opt):
        super().__init__()
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        sp_norm = norms.get_spectral_norm(opt)
        self.conv_0 = sp_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = sp_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))
        if self.learned_shortcut:
            self.conv_s = sp_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))

        spade_conditional_input_dims = opt.semantic_nc
        if not opt.no_3dnoise:
            spade_conditional_input_dims += opt.z_dim

        self.norm_0 = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.norm_1 = norms.SPADE(opt, fmiddle, spade_conditional_input_dims)
        if self.learned_shortcut:
            self.norm_s = norms.SPADE(opt, fin, spade_conditional_input_dims)
        self.activ = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        dx = self.conv_0(self.activ(self.norm_0(x, seg)))
        dx = self.conv_1(self.activ(self.norm_1(dx, seg)))
        out = x_s + dx
        return out
