import torch
import torch.nn as nn
import models.norms as norms


class OASIS_Discriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        output_channel = opt.semantic_nc + 1 # for N+1 loss
        #self.channels = [3, 128, 128, 256, 256, 512, 512]
        self.channels = self.opt.Discriminator_channels
        if self.opt.use_D_input_cat_label:
            self.channels[0] += opt.semantic_nc
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])
        # encoder part
        for i in range(opt.num_res_blocks):
            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
        # decoder part
        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
        for i in range(1, opt.num_res_blocks-1):
            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input):
        x = input
        #encoder
        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)
        #decoder
        x = self.body_up[0](x)
        for i in range(1, len(self.body_down)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))
        ans = self.layer_up_last(x)
        return ans

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride, opt):
        super().__init__()
        sp_norm = norms.get_spectral_norm(opt)
        self.conv = nn.Sequential(
            sp_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            sp_norm(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1)),
        )
        self.stride = stride
        if stride == 2:
            self.skip = sp_norm(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False))
        self.actvn = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, input):
        out = self.conv(input)
        if self.stride == 2:
            skip = self.skip(input)
            out = out + skip
        else:
            out = out + input 
        out = self.actvn(out)
        return out

class Discriminator_Global_Binary(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        sp_norm = norms.get_spectral_norm(opt)
        
        self.in_dims = [3, 128, 128, 256, 256, 512]
        self.out_dims = [128, 128, 256, 256, 512, 512]
        
        self.blocks = []
        for i in range(len(self.in_dims)):
            self.blocks.append(
                nn.Sequential(
                    ResBlock(self.in_dims[i], self.out_dims[i], stride=2, opt=opt),
                    #ResBlock(self.out_dims[i], self.out_dims[i], stride=1, opt=opt)
                )
            )
#                nn.Sequential(
#                    sp_norm(nn.Conv2d(self.in_dims[i], self.out_dims[i], kernel_size=3, stride=1, padding=1)),
#                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
#                    sp_norm(nn.Conv2d(self.out_dims[i], self.out_dims[i], kernel_size=4, stride=2, padding=1)),
#                    nn.LeakyReLU(negative_slope=0.2, inplace=False),
#                ))

        self.blocks = nn.ModuleList(self.blocks)
        self.conv = sp_norm(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.actvn = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.linear = nn.Sequential(
            sp_norm(nn.Linear(256, 64)),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            #sp_norm(nn.Linear(64, 1)),
            nn.Linear(64, 1),
        )

#        output_channel = opt.semantic_nc + 1 # for N+1 loss
#        #self.channels = [3, 128, 128, 256, 256, 512, 512]
#        self.channels = self.opt.Discriminator_channels
#        if self.opt.use_D_input_cat_label:
#            self.channels[0] += opt.semantic_nc
##        self.body_up   = nn.ModuleList([])
#        self.body_down = nn.ModuleList([])
#        # encoder part
#        for i in range(opt.num_res_blocks):
#            self.body_down.append(residual_block_D(self.channels[i], self.channels[i+1], opt, -1, first=(i==0)))
#        self.last_layer = nn.Sequential(
#            nn.LeakyReLU(0.2, False),
#            nn.Conv2d(self.channels[-1], 1, kernel_size=1, stride=1, padding=0)
#        )
#        # decoder part
#        self.body_up.append(residual_block_D(self.channels[-1], self.channels[-2], opt, 1))
#        for i in range(1, opt.num_res_blocks-1):
#            self.body_up.append(residual_block_D(2*self.channels[-1-i], self.channels[-2-i], opt, 1))
#        self.body_up.append(residual_block_D(2*self.channels[1], 64, opt, 1))
#        self.layer_up_last = nn.Conv2d(64, output_channel, 1, 1, 0)

    def forward(self, input):
        x = input
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        x = self.conv(x)
        x = self.actvn(x) 
        x = torch.sum(x, dim=[2,3])
        ans = torch.squeeze(self.linear(x), -1)

#        for i in range(len(self.body_down)):
#            x = self.body_down[i](x)
#        ans = self.last_layer(x)
        return ans

class residual_block_D(nn.Module):
    def __init__(self, fin, fout, opt, up_or_down, first=False):
        super().__init__()
        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout
        norm_layer = norms.get_spectral_norm(opt)
        if first:
            self.conv1 = nn.Sequential(norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), nn.Upsample(scale_factor=2), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
        self.conv2 = nn.Sequential(nn.LeakyReLU(0.2, False), norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))
        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))
        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):
        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s
