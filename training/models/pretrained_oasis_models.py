from models.sync_batchnorm import DataParallelWithCallback
import models.original_generator as generators
import models.original_discriminator as discriminators
from models.util import generate_labelmix
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses

import argparse

class OASIS_model(nn.Module):
    def __init__(self, train_opt):
        super(OASIS_model, self).__init__()

        opt = argparse.Namespace()
        opt.no_spectral_norm = False
        opt.param_free_norm = 'syncbatch'
        opt.spade_ks = 3
        opt.channels_G = 64
        opt.semantic_nc = 151
        opt.z_dim = 64
        opt.crop_size = 256
        opt.aspect_ratio = 1.0
        opt.num_res_blocks = 6
        opt.no_3dnoise = False
        opt.no_EMA = False
        opt.phase = 'test'

        opt.ckpt_iter = 'best'
        opt.checkpoints_dir = train_opt.pretrained_oasis_checkpoints_dir
        opt.name = 'oasis_ade20k_pretrained'

        opt.gpu_ids = train_opt.gpu_ids


        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.OASIS_Generator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()

    def forward(self, image, label, mode, losses_computer, z=None):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(label)
            output_D = self.netD(fake)
            loss_G_adv = losses_computer.loss_multi(output_D, label, for_real=True)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None
            return loss_G, {"Generator": loss_G_adv, "Vgg": loss_G_vgg}

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss_multi(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss_multi(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, {"D_fake": loss_D_fake, "D_real": loss_D_real, "LabelMix": loss_D_lm}

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label, z=z)
                else:
                    fake = self.netEMA(label, z=z)
            return fake

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                init.xavier_normal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


  


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
    else:
        input_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], input_semantics



