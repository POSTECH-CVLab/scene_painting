from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses

from utils.diff_aug import DiffAugment


class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.OASIS_Generator(opt)
        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
        self.print_parameter_count()
        self.init_networks()
        #--- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        #--- load previous checkpoints if needed ---
        self.load_checkpoints()
        #--- perceptual loss ---#
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss()

    def forward(self, fake_label, coord_image, real_image, real_label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            fake = self.netG(fake_label, coord_image)
            if self.opt.use_D_input_cat_label:
                output_D = self.netD(torch.cat([fake, fake_label], dim=1))
            else:
                output_D = self.netD(fake)

            loss_G_adv = losses_computer.loss_multi(output_D, fake_label, for_real=True)
            loss_G += loss_G_adv
#            if self.opt.add_vgg_loss:
#                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
#                loss_G += loss_G_vgg
#            else:
#                loss_G_vgg = None
            loss_G_vgg = None
            return loss_G, {"Generator": loss_G_adv, "Vgg": loss_G_vgg}

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(fake_label, coord_image)

            if self.opt.use_diff_aug:
                fake = DiffAugment(fake, policy="color") # policy="color,translation,cutout"
                real_image = DiffAugment(real_image, policy="color")

            if self.opt.use_D_dontcare_zero_mask:
                mask = (real_label[:,0] == 1)
                real_mask = torch.stack([mask, mask, mask], dim=1) 
                real_image[real_mask] = 0

            if self.opt.use_D_dontcare_fake_mask:
                mask = (real_label[:,0] == 1)
                real_mask = torch.stack([mask, mask, mask], dim=1) 
                real_image[real_mask] = fake[real_mask]

            if self.opt.use_D_input_cat_label:
                output_D_fake = self.netD(torch.cat([fake, fake_label], dim=1))
            else:
                output_D_fake = self.netD(fake)

            loss_D_fake = losses_computer.loss_multi(output_D_fake, fake_label, for_real=False)
            loss_D += loss_D_fake
            if self.opt.use_D_input_cat_label:
                output_D_real = self.netD(torch.cat([real_image, real_label], dim=1))
            else:
                output_D_real = self.netD(real_image)

            loss_D_real = losses_computer.loss(output_D_real, real_label, for_real=True)
            loss_D += loss_D_real
#            if not self.opt.no_labelmix:
#                mixed_inp, mask = generate_labelmix(label, fake, image)
#                output_D_mixed = self.netD(mixed_inp)
#                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
#                                                                                output_D_real)
#                loss_D += loss_D_lm
#            else:
#                loss_D_lm = None
            loss_D_lm = None
            return loss_D, {"D_fake": loss_D_fake, "D_real": loss_D_real, "LabelMix": loss_D_lm}

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(fake_label, coord_image)
                else:
                    fake = self.netEMA(fake_label, coord_image)
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
    data['real_label'] = data['real_label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['coord_image'] = data['coord_image'].cuda()
        data['real_image'] = data['real_image'].cuda()
        data['real_label'] = data['real_label'].cuda()

    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
    else:
        input_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    real_label_map = data['real_label']
    bs, _, h, w = real_label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        real_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
    else:
        real_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
    real_semantics = real_label.scatter_(1, real_label_map, 1.0)

    return  input_semantics, data['coord_image'], data['real_image'], real_semantics



