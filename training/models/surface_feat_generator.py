import torch.nn as nn 
import models.norms as norms
import torch
import torch.nn.functional as F

import math

#from models.cnn_style_recon_generator import PosEncodingNeRF
from models.cnn_style_recon_generator import FourierFeature 

from models.util import AdaptiveInstanceNorm2d 
from models.original_stylegan_v2_model import ModulatedConv2d
from models.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix
from models.conv_encoder import ConvEncoder


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(x)
        out += identity

        return out


class ResBlock_knk1(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=1, padding=0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.lrelu(out)

        out = self.conv2(x)
        out += identity

        return out
    

class AffineLayer(nn.Module):
    def __init__(self, in_channels, style_features):
        super().__init__()

        self.affine = nn.Linear(in_features=style_features, out_features=in_channels * 2)

        self.affine.bias.data[:in_channels] = 1  # initial gamma is 1
        self.affine.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.affine(style).unsqueeze(2).unsqueeze(3)  # (batch_size, 2*in_channels, 1, 1)
        gamma, beta = style.chunk(2, dim=1)

        x = input
        out = gamma * x + beta  # (batch_size, in_channels, height, width)

        return out

class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.))
        if not angular:
            emb = emb * math.pi

        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = nn.Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        img_size = x.shape[-1]
        x = x.reshape(x.shape[0], self.in_dim, -1)
        x = x.transpose(1,2)

        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))

        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
            
        feat_dim = x.shape[-1]
        x = x.transpose(1,2).reshape(x.shape[0], feat_dim, img_size, img_size)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr


class OASIS_Generator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.init_W, self.init_H = self.compute_latent_vector_size(opt)

        if self.opt.use_label_embedding:
            self.label_embedding = nn.Embedding(self.opt.semantic_nc, self.opt.label_embedding_dim)

        if self.opt.z_mapping_type == 'none':
            pass
        elif self.opt.z_mapping_type == 'mapping_net':
            self.mapping_net = nn.Sequential(
                            nn.Linear(opt.z_dim, opt.z_mapping_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(opt.z_mapping_dim, opt.z_mapping_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(opt.z_mapping_dim, opt.z_mapping_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(opt.z_mapping_dim, opt.z_mapping_dim),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(opt.z_mapping_dim, opt.z_mapping_dim),
                        )
        elif self.opt.z_mapping_type == 'clamp':
            pass
        else:
            raise ValueError()
        self.coord_mlp = MLPNet(opt)
        self.conv_img_output1 = nn.Conv2d(opt.mlp_hdim, 3, 1, padding=0)

        if self.opt.surface_feat_model_view_encoding:
            self.view_encoding_net = ConvEncoder(opt)
            self.view_z_mapping_net = nn.Sequential(
                            nn.Linear(256, 64),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Linear(64, 64),
                            )

    def compute_latent_vector_size(self, opt):
        w = opt.crop_size // (2**(opt.num_res_blocks-1))
        h = round(w / opt.aspect_ratio)
        return h, w

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def encode_view_z(self, input):
        mu, logvar = self.view_encoding_net(input)
        view_z = self.reparameterize(mu, logvar)
        return view_z, mu, logvar

    def forward(self, input, embed_idx_map, z=None, feat=None, view_img=None, pseudo_image=None):

        seg = input
        if not self.opt.no_3dnoise:
            dev = seg.device
            if z is None:
                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)

            if self.opt.z_mapping_type == 'none':
                pass
            elif self.opt.z_mapping_type == 'mapping_net':
                style = self.mapping_net(z)
            elif self.opt.z_mapping_type == 'clamp':
                style = torch.clamp(z, -1, 1)

        if self.opt.use_label_embedding:
            label_idx_map = torch.argmax(input, dim=1)
            seg = self.label_embedding(label_idx_map)
            seg = seg.transpose(2,3).transpose(1,2).contiguous()

        view_z = None
        if self.opt.surface_feat_model_view_encoding:
            if view_img is None:
                if self.opt.view_encoding_input_type == 'label':
                    view_img = seg
                elif self.opt.view_encoding_input_type == 'pseudo_image':
                    view_img = pseudo_image
            view_z, mu, logvar = self.encode_view_z(view_img)
            view_z = self.view_z_mapping_net(view_z)

        output1 = self.coord_mlp(embed_idx_map, seg, style, feat, view_z, z)
        
        output1 = self.conv_img_output1(F.leaky_relu(output1, 2e-1))
        output1 = torch.tanh(output1)

        if self.opt.surface_feat_model_view_encoding:
            return output1, output1, mu, logvar
        else:
            return output1, output1 

#    def get_feature_map(self, input, embed_idx_map, z=None, feat=None):
#        seg = input
#        if not self.opt.no_3dnoise:
#            dev = seg.device
#            if z is None:
#                z = torch.randn(seg.size(0), self.opt.z_dim, dtype=torch.float32, device=dev)
#
#            if self.opt.z_mapping_type == 'none':
#                pass
#            elif self.opt.z_mapping_type == 'mapping_net':
#                z = self.mapping_net(z)
#            elif self.opt.z_mapping_type == 'clamp':
#                z = torch.clamp(z, -1, 1)
#
#        if self.opt.use_label_embedding:
#            label_idx_map = torch.argmax(input, dim=1)
#            seg = self.label_embedding(label_idx_map)
#            seg = seg.transpose(2,3).transpose(1,2).contiguous()
#
#        feat = self.coord_mlp(embed_idx_map, seg, z, feat)
##        output1 = self.conv_img_output1(F.leaky_relu(output1, 2e-1))
##        output1 = torch.tanh(output1)
#
#        return feat 


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        # self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        # out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class MLPNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        if self.opt.use_point_embedding:
            self.embedding = nn.Embedding(self.opt.point_embedding_num, 
                                          self.opt.point_embedding_dim)

        if opt.pos_encoding_model == 'none':
            self.pos_encoding = None
            pos_encoding_dim = 3
        elif opt.pos_encoding_model == 'nerf':
            if self.opt.use_point_embedding:
                self.pos_encoding = NeRFPosEmbLinear(self.opt.point_embedding_dim, 
                        self.opt.point_embedding_dim*2*opt.pos_encoding_num_freq, 
                        angular=False, no_linear=False, cat_input=True)
                pos_encoding_dim = self.opt.point_embedding_dim*2*opt.pos_encoding_num_freq + self.opt.point_embedding_dim
            else:
                self.pos_encoding = NeRFPosEmbLinear(3, 3*2*opt.pos_encoding_num_freq, 
                        angular=False, no_linear=True, cat_input=True)
                pos_encoding_dim = 3*2*opt.pos_encoding_num_freq + 3

#        elif opt.pos_encoding_model == 'fourier':
#            self.pos_encoding = FourierFeature(embedding_size=opt.fourier_dim, embedding_scale=12.)
#            pos_encoding_dim = self.pos_encoding.out_dim
        else:
            raise ValueError('no pos_encoding_model')

        hdim = opt.mlp_hdim 
        self.num_layers = 7 

        if self.opt.z_mapping_type == 'none' or self.opt.z_mapping_type == 'clamp':
            style_features = opt.z_dim
        elif self.opt.z_mapping_type == 'mapping_net':
            style_features = opt.z_mapping_dim

        if self.opt.use_label_embedding:
            in_dim = pos_encoding_dim + opt.label_embedding_dim
        else:
            in_dim = pos_encoding_dim + opt.semantic_nc


        if self.opt.use_3dfeat:
            in_dim += opt.feat_dim
        else:
            pass

        if self.opt.surface_feat_model_3dnoise == 'map_z':
            in_dim += opt.z_mapping_dim
        elif self.opt.surface_feat_model_3dnoise == 'raw_z':
            in_dim += opt.z_dim
        else:
            pass

        if self.opt.surface_feat_model_view_encoding:
            if self.opt.view_encoding_use_type == 'style_cat':
                style_features += 64 
            elif self.opt.view_encoding_use_type == 'input_cat':
                in_dim += 64
            else:
                raise ValueError()

        self.models = []
        out_ch = hdim
        for i in range(self.num_layers):
            if i == 0:
                in_ch = in_dim 
            else:
                in_ch = hdim
            self.models.append(
                StyledConv(in_ch, out_ch, kernel_size=1, 
                           style_dim=style_features),
            )

        self.models = nn.ModuleList(self.models)
        
        if self.opt.surface_feat_model_convblock_type == 'k3k3':
            self.conv1 = nn.Conv2d(hdim, hdim, kernel_size=1, padding=0)
            self.lrelu = nn.LeakyReLU(inplace=True)
            self.resblock1 = ResBlock(hdim, hdim, kernel_size=3, padding=1)
            self.affine1 = AffineLayer(hdim, opt.z_mapping_dim)
            self.resblock2 = ResBlock(hdim, hdim, kernel_size=3, padding=1)
            self.affine2 = AffineLayer(hdim, opt.z_mapping_dim)
            self.resblock3 = ResBlock(hdim, hdim, kernel_size=1, padding=0)
        elif self.opt.surface_feat_model_convblock_type == 'k3k1':
            self.conv1 = nn.Conv2d(hdim, hdim, kernel_size=1, padding=0)
            self.lrelu = nn.LeakyReLU(inplace=True)
            self.resblock1 = ResBlock_knk1(hdim, hdim, kernel_size=3, padding=1)
            self.affine1 = AffineLayer(hdim, opt.z_mapping_dim)
            self.resblock2 = ResBlock_knk1(hdim, hdim, kernel_size=3, padding=1)
            self.affine2 = AffineLayer(hdim, opt.z_mapping_dim)
            self.resblock3 = ResBlock_knk1(hdim, hdim, kernel_size=1, padding=0)
        elif self.opt.surface_feat_model_convblock_type == 'None':
            self.conv1 = nn.Conv2d(hdim, hdim, kernel_size=1, padding=0)
        else:
            raise ValueError('')


    def forward(self, x, seg, style, feat=None, view_z=None, raw_z=None):
        
        if self.opt.use_point_embedding:
            x = self.embedding(x)
            x = x.transpose(2,3).transpose(1,2).contiguous()

        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        if self.opt.use_3dfeat:
            x = torch.cat([x, seg, feat], dim=1)
        else:
            x = torch.cat([x, seg], dim=1)

        if self.opt.surface_feat_model_3dnoise == 'map_z':
            z_dim = style.shape[1]
            z = style.view(seg.size(0), z_dim, 1, 1)
            z = z.expand(seg.size(0), z_dim, seg.size(2), seg.size(3))
            x = torch.cat([x, z], dim=1)
        elif self.opt.surface_feat_model_3dnoise == 'raw_z':
            z_dim = raw_z.shape[1]
            z = raw_z.view(seg.size(0), z_dim, 1, 1)
            z = z.expand(seg.size(0), z_dim, seg.size(2), seg.size(3))
            x = torch.cat([x, z], dim=1)

        if self.opt.surface_feat_model_view_encoding:
            if self.opt.view_encoding_use_type == 'input_cat':
                view_z = view_z.view(seg.size(0), 64, 1, 1)
                view_z = view_z.expand(seg.size(0), 64, seg.size(2), seg.size(3))
                x = torch.cat([x, view_z], dim=1)
            elif self.opt.view_encoding_use_type == 'style_cat':
                style = torch.cat([style, view_z], dim=1)
            else:
                raise ValueError()

        for i in range(self.num_layers):
            x = self.models[i](x, style)
        
        if self.opt.surface_feat_model_convblock_type == 'None':
            x = self.conv1(x)
        else:
            x = self.conv1(x)
            x = self.lrelu(x)
            x = self.resblock1(x)
            x = self.affine1(x, style)
            x = self.lrelu(x)
            x = self.resblock2(x)
            x = self.affine2(x, style)
            x = self.lrelu(x)
            x = self.resblock3(x)

        return x

