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


class FourierFeature(nn.Module):
    def __init__(self, opt, embedding_size=256, embedding_scale=1):
        super().__init__()
        
        bval = np.load(f'{opt.dataroot}/fourier_bval_{embedding_size}.npy')
        #self.bval = torch.randn(3, embedding_size) * embedding_scale
        self.bval = torch.from_numpy(bval).float() * embedding_scale

        self.out_dim = embedding_size*2

    def forward(self, coords):
        bval = self.bval.to(coords.device)

        img_size = coords.shape[-1]
        batch_size = coords.shape[0]
        coords = coords.reshape(batch_size, 3, -1).permute(0, 2, 1)
        coords_pos_enc = torch.cat([
            torch.sin(torch.matmul(2.*np.pi*coords, bval)),
            torch.cos(torch.matmul(2.*np.pi*coords, bval))], dim=1)

        coords_pos_enc = coords_pos_enc.permute(0, 2, 1)
        return coords_pos_enc.reshape(batch_size, self.out_dim, img_size, img_size)


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        if opt.pos_encoding_model == 'none':
            self.pos_encoding = None
            pos_encoding_dim = 3
        elif opt.pos_encoding_model == 'nerf':
            self.pos_encoding = PosEncodingNeRF(in_features=3, num_frequencies=opt.pos_encoding_num_freq)
            pos_encoding_dim = self.pos_encoding.out_dim
        elif opt.pos_encoding_model == 'fourier':
            self.pos_encoding = FourierFeature(opt=opt, embedding_size=opt.fourier_dim, embedding_scale=opt.fourier_scale)
            pos_encoding_dim = self.pos_encoding.out_dim
        else:
            raise ValueError('no pos_encoding_model')

        if opt.coordinate_embedding_model == 'one_linear':
            self.coordinate_embedding = nn.Conv2d(3, opt.coordinate_embedding_dim, kernel_size=1)
        elif opt.coordinate_embedding_model == 'two_linear':
            self.coordinate_embedding = nn.Sequential(
                nn.Conv2d(3, opt.coordinate_embedding_dim // 4, kernel_size=1),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(opt.coordinate_embedding_dim // 4, opt.coordinate_embedding_dim, kernel_size=1))
        elif opt.coordinate_embedding_model == 'none':
            self.coordinate_embedding = None
        else:
            raise ValueError('no coordinate_embedding_model')

#        if opt.pos_encoding_model == 'none':
#            in_dim = pos_encoding_dim + opt.semantic_nc + opt.z_dim
#        else:
#            in_dim = pos_encoding_dim + opt.coordinate_embedding_dim + opt.semantic_nc + opt.z_dim

        in_dim = opt.semantic_nc + opt.z_dim
        if not opt.pos_encoding_model == 'none'and not opt.coordinate_embedding_model == 'none':
            in_dim += pos_encoding_dim + opt.coordinate_embedding_dim
        elif not opt.pos_encoding_model == 'none':
            in_dim += pos_encoding_dim
        elif not opt.coordinate_embedding_model == 'none':
            in_dim += opt.coordinate_embedding_dim
        else:
            in_dim += 3

        hdim = opt.hidden_dim
        self.num_layers = opt.G_num_layers

        self.first_layer = nn.Conv2d(in_dim, hdim, kernel_size=1)
        self.models = []
        for i in range(self.num_layers):
            self.models.append(
                nn.Sequential(
                    #nn.Conv2d(hdim, hdim, kernel_size=1),
                    nn.Conv2d(hdim, hdim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.models = nn.ModuleList(self.models)
        self.last_layer = nn.Conv2d(hdim, 3, kernel_size=1)

        ### Refine network
        if self.opt.use_refine_net:
            self.refine_models = []
            for i in range(3):
                if i == 0:
                    self.refine_models.append(
                        nn.Sequential(
                            nn.Conv2d(3, hdim, kernel_size=3, padding=1),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
                else:
                    self.refine_models.append(
                        nn.Sequential(
                            nn.Conv2d(hdim, hdim, kernel_size=3, padding=1),
                            nn.LeakyReLU(0.2, inplace=True),
                        )
                    )
            self.refine_models = nn.ModuleList(self.refine_models)
            self.refine_last_layer = nn.Conv2d(hdim, 3, kernel_size=1)

    def forward(self, input, coord_image, z):

        seg = input

        if self.pos_encoding is not None:
            pos_encoding = self.pos_encoding(coord_image)
        else:
            pos_encoding =None

        if self.coordinate_embedding is not None and self.pos_encoding is not None:
            coord_embedding = self.coordinate_embedding(coord_image)
            coord_image = torch.cat([coord_embedding, pos_encoding], dim=1)
        elif self.pos_encoding is not None:
            coord_image = pos_encoding
        elif self.coordinate_embedding is not None:
            coord_embedding = self.coordinate_embedding(coord_image)
            coord_image = coord_embedding


#        if self.pos_encoding is not None:
#            pos_encoding = self.pos_encoding(coord_image)
#            coord_embedding = self.coordinate_embedding(coord_image)
#            coord_image = torch.cat([coord_embedding, pos_encoding], dim=1)

        #x = torch.cat([seg, coord_image], dim=1)
#        if z is None:
#            dev = seg.get_device() if self.opt.gpu_ids != "-1" else "cpu"
#            z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
#            z = z.view(z.size(0), self.opt.z_dim, 1, 1)
#            z = z.expand(z.size(0), self.opt.z_dim, seg.size(2), seg.size(3))

        z = z.view(z.size(0), z.size(1), 1, 1)
        z = z.expand(z.size(0), z.size(1), seg.size(2), seg.size(3))

        x = torch.cat([seg, coord_image, z], dim=1)
        x = self.first_layer(x)
        for i in range(self.num_layers):
            x = self.models[i](x)
        x = self.last_layer(x)

        x1 = torch.tanh(x)

        if not self.opt.use_refine_net:
            return x1
        else:

            tmpx = x
            for i in range(3):
                x = self.refine_models[i](x)
            x = self.refine_last_layer(x)
            x = x + tmpx

            x2 = torch.tanh(x)

            return x1, x2


