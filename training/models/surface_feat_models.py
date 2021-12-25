from models.sync_batchnorm import DataParallelWithCallback
import models.surface_feat_generator as generators
import models.original_discriminator as discriminators
import models.proj_discriminator as proj_discriminators
from models.util import generate_labelmix
import os
import copy
import torch
import torch.nn as nn
from torch.nn import init
import models.losses as losses


class OASIS_model(nn.Module):
    def __init__(self, opt):
        super(OASIS_model, self).__init__()
        self.opt = opt
        #--- generator and discriminator ---
        self.netG = generators.OASIS_Generator(opt)
        if opt.phase == "train":
            if opt.use_netD_output1:
                if self.opt.discriminator == 'oasis':
                    self.netD_output1 = discriminators.OASIS_Discriminator(opt)
                elif self.opt.discriminator == 'proj':
                    self.netD_output1 = proj_discriminators.OASIS_Discriminator(opt)

            if opt.use_output2:
                self.netD_output2 = discriminators.OASIS_Discriminator(opt)

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

    def forward(self, pseudo_image, label, coord, real_image, real_label, embed_idx_map,
                mode, losses_computer, z=None, feat_map=None, view=None):
        # Branching is applied to be compatible with DataParallel
        if mode == "losses_G":
            loss_G = 0
            if self.opt.use_point_embedding:
                fake1, fake2 = self.netG(label, embed_idx_map, z=z, feat=feat_map)
            else:
                if self.opt.surface_feat_model_view_encoding:
                    fake1, fake2, mu, logvar = self.netG(label, coord, z=z, feat=feat_map, pseudo_image=pseudo_image)
                else:
                    fake1, fake2 = self.netG(label, coord, z=z, feat=feat_map)

            loss_KLD = None
            if self.opt.surface_feat_model_view_encoding:
                lambda_kld = 0.05
                loss_KLD = lambda_kld * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            loss_G_real_adv_output2 = None
            if self.opt.add_real_adv_loss_output2 or self.opt.add_pseudo_adv_loss_output2: 
                output_D = self.netD_output2(fake2)
                loss_G_real_adv_output2 = losses_computer.loss_multi(output_D, label, for_real=True)
                loss_G += loss_G_real_adv_output2

            # binary gan loss
            loss_G_binary_output2 = None
            if self.opt.add_real_binary_gan_loss_output2:
                output_D = self.netD_output2(fake2)
                loss_G_binary_output2 = losses_computer.loss_binary(fake=output_D[:,0,:,:], for_D=False)
                loss_G += loss_G_binary_output2

            loss_G_adv_output1 = None
            if self.opt.add_pseudo_adv_loss_output1 or self.opt.add_real_adv_loss_output1: 
                assert self.opt.use_netD_output1
                
                output_D_output1 = self.netD_output1(fake1)
                if self.opt.discriminator == 'oasis':
                    loss_G_adv_output1 = (self.opt.lambda_G_real_output1
                             * losses_computer.loss_multi(output_D_output1, label, for_real=True))
                elif self.opt.discriminator == 'proj':
                    loss_G_adv_output1 = (self.opt.lambda_G_real_output1
                             * losses_computer.proj_loss(output_D_output1, label, loss_type='g_loss_fake'))

                loss_G += loss_G_adv_output1

            # binary gan loss
            loss_G_binary_output1_for_pseudo = None
            if self.opt.add_pseudo_binary_gan_loss_output1:
                assert self.opt.use_netD_output1
                output_D = self.netD_output1(fake1)
                loss_G_binary_output1_for_pseudo = losses_computer.loss_binary(fake=output_D[:,0,:,:], for_D=False)
                loss_G += loss_G_binary_output1_for_pseudo

            # binary gan loss
            loss_G_binary_output1_for_real = None
            if self.opt.add_real_binary_gan_loss_output1:
                assert self.opt.use_netD_output1
                output_D = self.netD_output1(fake1)
                loss_G_binary_output1_for_real = losses_computer.loss_binary(fake=output_D[:,0,:,:], for_D=False)
                loss_G += loss_G_binary_output1_for_real

            loss_G_vgg = None
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake1, pseudo_image)
                loss_G += loss_G_vgg

            if self.opt.surface_feat_model_defocal_weight:
                #defocal_weight = 1 / (torch.abs(fake1.detach().clone() - pseudo_image) + 1)**3
                batch_size = label.shape[0]
                num_label = label.shape[1]
                defocal_weight = torch.zeros((batch_size, label.shape[2], label.shape[3]), dtype=torch.float32, device=label.device)
                for b in range(batch_size):
                    b_pseudo = pseudo_image[b].transpose(0,1).transpose(1,2)
                    b_label = label[b]
                    for i in range(num_label):

                        bi_label = b_label[i].bool()
                        if not bi_label.any(): continue
                        value = b_pseudo[bi_label]
                        dist = torch.norm(value - torch.mean(value, dim=0), dim=1)
                        defocal_weight[b][bi_label] = torch.exp(-(dist**2/self.opt.surface_feat_model_defocal_lambda))

                defocal_weight_l1 = defocal_weight.unsqueeze(1)
                defocal_weight_l1 = defocal_weight_l1.expand(label.shape[0], 3, label.shape[2], label.shape[2])

            loss_G_pseudo_recon_l1 = None
            if self.opt.add_pseudo_recon_l1_loss: 
                if self.opt.surface_feat_model_defocal_weight:
                    loss_G_pseudo_recon_l1 = self.opt.lambda_pseudo_recon_l1 * \
                        torch.mean(defocal_weight_l1 * torch.abs(fake1 - pseudo_image))

                else:
                    loss_G_pseudo_recon_l1 = self.opt.lambda_pseudo_recon_l1 * torch.abs(fake1 - pseudo_image).mean()
                loss_G += loss_G_pseudo_recon_l1

            loss_G_pseudo_recon_l2 = None
            if self.opt.add_pseudo_recon_l2_loss: 
                if self.opt.surface_feat_model_l2_use_norm:
                    if self.opt.surface_feat_model_defocal_weight:
                        loss_G_pseudo_recon_l2 = self.opt.lambda_pseudo_recon_l2 * \
                            torch.mean(defocal_weight * torch.norm(fake1 - pseudo_image, dim=1))
                    else:
                        loss_G_pseudo_recon_l2 = self.opt.lambda_pseudo_recon_l2 * (torch.norm(fake1 - pseudo_image, dim=1)).mean()
                else:
                    if self.opt.surface_feat_model_defocal_weight:
                        loss_G_pseudo_recon_l2 = self.opt.lambda_pseudo_recon_l2 * \
                            torch.mean(defocal_weight_l1 * (fake1 - pseudo_image)**2)
                    else:
                        loss_G_pseudo_recon_l2 = self.opt.lambda_pseudo_recon_l2 * ((fake1 - pseudo_image)**2).mean()
                loss_G += loss_G_pseudo_recon_l2

            loss_G_output12_recon_l1 = None
            if self.opt.add_output12_recon_l1_loss: 
                loss_G_output12_recon_l1 = self.opt.lambda_output12_recon_l1 * torch.abs(fake1 - fake2).mean()
                loss_G += loss_G_output12_recon_l1
            
            if self.opt.use_output_log:
                if loss_G.device.type == 'cuda' and loss_G.device.index == 0:# fixme
                    print(
                        f'fake1 min: {fake1.min():.3f} | '
                        f'fake1 max: {fake1.max():.3f} | '
                        f'fake2 min: {fake2.min():.3f} | '
                        f'fake2 max: {fake2.max():.3f} | '
                    )

            return loss_G, {"Generator": loss_G_real_adv_output2, "Vgg": loss_G_vgg,
                            "Pseudo_recon_l1": loss_G_pseudo_recon_l1, "Pseudo_recon_l2": loss_G_pseudo_recon_l2,
                            "Output12_recon_l1": loss_G_output12_recon_l1,
                            "Generator_output1": loss_G_adv_output1,
                            "G_binary_output2": loss_G_binary_output2,
                            "G_binary_output1_for_pseudo": loss_G_binary_output1_for_pseudo,
                            "G_binary_output1_for_real": loss_G_binary_output1_for_real,
                            "KLD": loss_KLD, }


        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                if self.opt.use_point_embedding:
                    fake1, fake2 = self.netG(label, embed_idx_map, z=z, feat=feat_map)
                else:
                    if self.opt.surface_feat_model_view_encoding:
                        fake1, fake2, mu, logvar = self.netG(label, coord, z=z, feat=feat_map, pseudo_image=pseudo_image)
                    else:
                        fake1, fake2 = self.netG(label, coord, z=z, feat=feat_map)

            loss_D_fake_output2 = None
            loss_D_real_output2 = None
            if self.opt.add_real_adv_loss_output2:
                # auxiliary gan loss
                assert self.opt.use_output2
                output_D_fake = self.netD_output2(fake2)
                output_D_real = self.netD_output2(real_image)

                loss_D_fake_output2 = losses_computer.loss_multi(output_D_fake, label, for_real=False)
                loss_D += loss_D_fake_output2

                loss_D_real_output2 = losses_computer.loss_multi(output_D_real, real_label, for_real=True)
                loss_D += loss_D_real_output2

            # binary gan loss
            loss_D_binary_output2 = None
            if self.opt.add_real_binary_gan_loss_output2:
                assert self.opt.use_output2
                output_D_fake = self.netD_output2(fake2)
                output_D_real = self.netD_output2(real_image)

                loss_D_binary_output2 = losses_computer.loss_binary(output_D_real[:,0,:,:], output_D_fake[:,0,:,:], for_D=True)
                loss_D += loss_D_binary_output2

            loss_D_pseudo = None
            if self.opt.add_pseudo_adv_loss_output2: 
                output_D_pseudo = self.netD_output2(pseudo_image)
                loss_D_pseudo = losses_computer.loss_multi(output_D_pseudo, label, for_real=True)
                loss_D += loss_D_pseudo

            loss_D_fake_output1 = None
            loss_D_pseudo_output1 = None
            loss_D_real_output1 = None
            if self.opt.add_pseudo_adv_loss_output1 or self.opt.add_real_adv_loss_output1: 
                assert self.opt.use_netD_output1

                output_D_fake_output1 = self.netD_output1(fake1)
                if self.opt.discriminator == 'oasis':
                    loss_D_fake_output1 = (self.opt.lambda_D_fake_output1
                            * losses_computer.loss_multi(
                                output_D_fake_output1, label, for_real=False))
                elif self.opt.discriminator == 'proj':
                    loss_D_fake_output1 = (self.opt.lambda_D_fake_output1
                            * losses_computer.proj_loss(output_D_fake_output1, label, 'd_loss_fake'))
                loss_D += loss_D_fake_output1


                if self.opt.add_pseudo_adv_loss_output1:
                    output_D_pseudo_output1 = self.netD_output1(pseudo_image)
                    if self.opt.discriminator == 'oasis':
                        loss_D_pseudo_output1 = (self.opt.lambda_D_pseudo_output1
                            * losses_computer.loss_multi(
                                output_D_pseudo_output1, label, for_real=True))
                    elif self.opt.discriminator == 'proj':
                        loss_D_pseudo_output1 = (self.opt.lambda_D_pseudo_output1
                            * losses_computer.proj_loss(output_D_pseudo_output1, label, 'd_loss_real'))
                    loss_D += loss_D_pseudo_output1

                if self.opt.add_real_adv_loss_output1:
                    output_D_real_output1 = self.netD_output1(real_image)
                    if self.opt.discriminator == 'oasis':
                        loss_D_real_output1 = (self.opt.lambda_D_real_output1
                            * losses_computer.loss_multi(
                                output_D_real_output1, real_label, for_real=True))
                    elif self.opt.discriminator == 'proj':
                        loss_D_real_output1 = (self.opt.lambda_D_real_output1
                            * losses_computer.proj_loss(output_D_real_output1, real_label, 'd_loss_real'))
                    loss_D += loss_D_real_output1

            loss_D_binary_output1_for_pseudo = None
            if self.opt.add_pseudo_binary_gan_loss_output1:
                assert self.opt.use_netD_output1
                output_D_fake_output1 = self.netD_output1(fake1)
                output_D_real_output1 = self.netD_output1(pseudo_image)

                loss_D_binary_output1_for_pseudo = losses_computer.loss_binary(output_D_real_output1[:,0,:,:], 
                                                            output_D_fake_output1[:,0,:,:], for_D=True)
                loss_D += loss_D_binary_output1_for_pseudo

            loss_D_binary_output1_for_real = None
            if self.opt.add_real_binary_gan_loss_output1:
                assert self.opt.use_netD_output1
                output_D_fake_output1 = self.netD_output1(fake1)
                output_D_real_output1 = self.netD_output1(real_image)

                loss_D_binary_output1_for_real = losses_computer.loss_binary(output_D_real_output1[:,0,:,:], 
                                                            output_D_fake_output1[:,0,:,:], for_D=True)
                loss_D += loss_D_binary_output1_for_real


            loss_D_lm = None
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake1, pseudo_image)
                output_D_mixed = self.netD_output1(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * \
                    losses_computer.loss_labelmix(mask, output_D_mixed, 
                                                  output_D_fake_output1, output_D_pseudo_output1)
                loss_D += loss_D_lm

            return loss_D, {"D_fake": loss_D_fake_output2, "D_real": loss_D_real_output2,
                            "D_pseudo": loss_D_pseudo, "LabelMix": loss_D_lm,
                            "D_fake_output1": loss_D_fake_output1, 
                            "D_pseudo_output1": loss_D_pseudo_output1,
                            "D_real_output1": loss_D_real_output1,
                            "D_binary": loss_D_binary_output2, 
                            "D_binary_output1_for_pseudo": loss_D_binary_output1_for_pseudo,
                            "D_binary_output1_for_real": loss_D_binary_output1_for_real,}

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    if self.opt.use_point_embedding:
                        fake1, fake2 = self.netG(label, embed_idx_map, z=z, feat=feat_map, view_img=view)
                    else:
                        if self.opt.surface_feat_model_view_encoding:
                            fake1, fake2, mu, logvar = self.netG(label, coord, z=z, feat=feat_map, view_img=view, pseudo_image=pseudo_image)
                        else:
                            fake1, fake2 = self.netG(label, coord, z=z, feat=feat_map)
                else:
                    fake1, fake2 = self.netEMA(label, coord, z=z)
            return fake1, fake2

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
            if self.opt.use_netD_output1:
                self.netD_output1.load_state_dict(torch.load(path + "D_output1.pth"))
            if self.opt.use_output2:
                self.netD_output2.load_state_dict(torch.load(path + "D_output2.pth"))
 
            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG]
            if self.opt.use_netD_output1:
                networks += [self.netD_output1]
            if self.opt.use_output2:
                networks += [self.netD_output2]
        else:
            networks = [self.netG]

        from models.surface_feat_generator import StyledConv
        for network in networks:
            param_count = 0
            for name, module in network.named_modules():
                if (isinstance(module, nn.Conv2d)
                        or isinstance(module, nn.Linear)
                        or isinstance(module, nn.Embedding)
                        or isinstance(module, StyledConv)
                        ):
                    param_count += sum([p.data.nelement() for p in module.parameters()])
            print('Created', network.__class__.__name__, "with %d parameters" % param_count)

    def init_networks(self):
        def init_weights(m, gain=torch.nn.init.calculate_gain('leaky_relu', 0.2)):
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
            networks = [self.netG]
            if self.opt.use_netD_output1:
                networks += [self.netD_output1]
            if self.opt.use_output2:
                networks += [self.netD_output2]
        else:
            networks = [self.netG]
        
        if self.opt.init_type == 'none':
            pass
        elif self.opt.init_type == 'lrelu':
            for net in networks:
                net.apply(init_weights)
        else:
            raise ValueError()

#        for name, m in self.netG.named_modules():
##            if 'coord_mlp' in name and m.__class__.__name__=='Conv2d':
##                init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('leaky_relu', 0.2))
##                if hasattr(m, 'bias') and m.bias is not None:
##                    init.constant_(m.bias.data, 0.0)
#            if m.__class__.__name__=='Conv2d':
#                init.xavier_normal_(m.weight.data, gain=torch.nn.init.calculate_gain('leaky_relu', 0.2))
#                if hasattr(m, 'bias') and m.bias is not None:
#                    init.constant_(m.bias.data, 0.0)
#
#        for name, m in self.netD.named_modules():
#            classname = m.__class__.__name__
#            if classname.find('BatchNorm2d') != -1:
#                if hasattr(m, 'weight') and m.weight is not None:
#                    init.normal_(m.weight.data, 1.0, 0.02)
#                if hasattr(m, 'bias') and m.bias is not None:
#                    init.constant_(m.bias.data, 0.0)
#            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#                init.xavier_normal_(m.weight.data, gain=0.02)
#                if hasattr(m, 'bias') and m.bias is not None:
#                    init.constant_(m.bias.data, 0.0)
#
#        if self.opt.use_netD_output1:
#            for name, m in self.netD_output1.named_modules():
#                classname = m.__class__.__name__
#                if classname.find('BatchNorm2d') != -1:
#                    if hasattr(m, 'weight') and m.weight is not None:
#                        init.normal_(m.weight.data, 1.0, 0.02)
#                    if hasattr(m, 'bias') and m.bias is not None:
#                        init.constant_(m.bias.data, 0.0)
#                elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#                    init.xavier_normal_(m.weight.data, gain=0.02)
#                    if hasattr(m, 'bias') and m.bias is not None:
#                        init.constant_(m.bias.data, 0.0)


  


def preprocess_input(opt, data):

    data['label'] = data['label'].long()
    data['real_label'] = data['real_label'].long()
    data['pseudo_label'] = data['pseudo_label'].long()
    if opt.use_point_embedding:
        data['embed_idx_map'] = data['embed_idx_map'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['coord_image'] = data['coord_image'].cuda()
        data['real_image'] = data['real_image'].cuda()
        data['real_label'] = data['real_label'].cuda()
        data['pseudo_label'] = data['pseudo_label'].cuda()
        if opt.use_point_embedding:
            data['embed_idx_map'] = data['embed_idx_map'].cuda()
        if opt.use_3dfeat:
            data['feat_map'] = data['feat_map'].cuda()

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

    pseudo_label_map = data['pseudo_label']
    bs, _, h, w = pseudo_label_map.size()
    nc = 151
    if opt.gpu_ids != "-1":
        pseudo_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
    else:
        pseudo_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
    pseudo_semantics = pseudo_label.scatter_(1, pseudo_label_map, 1.0)

    if not opt.use_point_embedding:
        data['embed_idx_map'] = None

    if opt.use_3dfeat:
        return  input_semantics, data['coord_image'], data['real_image'], real_semantics, pseudo_semantics, data['embed_idx_map'], data['feat_map']
    else: 
        return  input_semantics, data['coord_image'], data['real_image'], real_semantics, pseudo_semantics, data['embed_idx_map']



