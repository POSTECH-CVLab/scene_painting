import torch
import models.losses as losses
import models
import models.surface_feat_models as surface_feat_models 
import models.pretrained_oasis_models as pretrained_oasis_models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
from utils.fid_scores import fid_pytorch
import config

from time import time
import random
import numpy as np

from train_etc_util import (style_recon_model_list, oasis_3dcoord_model_list, 
                            surface_feat_model_list, usis_model_list, omni_model_list,
                            class_specific_model_list)
from models.util import ortho


def main(opt):
    global model, loss
    # --- create utils ---#
    timer = utils.timer(opt)
    if opt.model in ['recon', 'style_recon', 'cnn_style_recon', 'fusion']:
        visualizer_losses = utils.losses_saver(opt, ["Generator", "Recon", "D_fake", "D_real", "LabelMix"])
    elif opt.model in oasis_3dcoord_model_list or opt.model in surface_feat_model_list:
        visualizer_losses = utils.losses_saver(opt,
                                               ["Generator", "Vgg", "Pseudo_recon_l1", "Pseudo_recon_l2",
                                                "Output12_recon_l1", "Generator_output1",
                                                "G_binary_output2",
                                                "G_binary_output1_for_pseudo", "G_binary_output1_for_real",
                                                "D_fake", "D_real", "D_pseudo", "LabelMix",
                                                "D_fake_output1", "D_pseudo_output1",
                                                "D_real_output1",
                                                "D_binary",
                                                "D_binary_output1_for_pseudo", "D_binary_output1_for_real",
                                                "KLD",
                                                ])
    elif opt.model in usis_model_list:
        visualizer_losses = utils.losses_saver(opt, ["G_adv", "Seg", "D_fake", "D_real", "R1"])
    elif opt.model in omni_model_list:
        visualizer_losses = utils.losses_saver(opt, ["G_adv", "Pseudo_recon_l1", "Pseudo_recon_l2", "Vgg",
                                                     "D_fake", "D_pseudo", "D_real", "LabelMix"])
    elif opt.model in class_specific_model_list:
        visualizer_losses = utils.losses_saver(opt, ["G_adv", "D_fake", "D_real", "LabelMix"])
    else:
        visualizer_losses = utils.losses_saver(opt, ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"])
    losses_computer = losses.losses_computer(opt)
    dataloader, dataloader_val = dataloaders.get_dataloaders(opt)
    im_saver = utils.image_saver(opt)
    fid_computer = fid_pytorch(opt, dataloader_val, dataloader)
    # --- create models ---#

    use_pretrained_oasis_model = False
    if opt.model == 'surface_feat':
        model = surface_feat_models.OASIS_model(opt)
        preprocess_input_func = surface_feat_models.preprocess_input
        use_pretrained_oasis_model = True
    else:
        raise ValueError('No model')
    model = models.util.put_on_multi_gpus(model, opt)

    # --- create pretrained OASIS model ---#
    pretrained_oasis_model = None
    if use_pretrained_oasis_model:
        pretrained_oasis_model = pretrained_oasis_models.OASIS_model(opt)
        pretrained_oasis_model = models.util.put_on_multi_gpus(pretrained_oasis_model, opt)
        pretrained_oasis_model.eval()

    # --- create optimizers ---#
    if opt.model in class_specific_model_list:
        netG_params = []
        netD_params = []
        for idx, label in enumerate(model.module.class_specific_label_list):
            netG_params += list(model.module.class_specific_netG_list[idx].parameters())
            netD_params += list(model.module.class_specific_netD_list[idx].parameters())
        if opt.optim == 'adam':
            optimizerG = torch.optim.Adam(netG_params,
                                          lr=opt.lr_g, betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.g_weight_decay)
            optimizerD = torch.optim.Adam(netD_params, 
                                          lr=opt.lr_d, betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.d_weight_decay)
        else:
            raise ValueError('no optim')
    else:
        netD_params = []
        if opt.use_netD_output1:
            netD_params += list(model.module.netD_output1.parameters())
        if opt.use_output2:
            netD_params += list(model.module.netD_output2.parameters())

        if opt.optim == 'adam':
            optimizerG = torch.optim.Adam(model.module.netG.parameters(), 
                                          lr=opt.lr_g, betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.g_weight_decay)
            optimizerD = torch.optim.Adam(netD_params, 
                                          lr=opt.lr_d, betas=(opt.beta1, opt.beta2),
                                          weight_decay=opt.d_weight_decay)
            if opt.model in usis_model_list:
                optimizer_seg = torch.optim.Adam(model.module.unet.parameters(), lr=opt.lr_g, betas=(opt.beta1, opt.beta2))
#    elif opt.optim == 'rmsprop':
#        optimizerG = torch.optim.RMSprop(model.module.netG.parameters(), lr=opt.lr_g)
#        optimizerD = torch.optim.Adam(netD_params, lr=opt.lr_d, betas=(opt.beta1, opt.beta2))
        else:
            raise ValueError('no optim')

    # ---  z_vec ---
    if 'style_interpolation' in opt.model and opt.use_fixed_z_vec:
        torch.manual_seed(0)
        z_list = []
        num_z_list = 9
        for i in range(num_z_list):
            z = torch.randn(opt.z_dim, dtype=torch.float32)
            z_list.append(z)
        for i in range(num_z_list):
            print(z_list[i][0])

    # --- the training loop ---#
    already_started = False
    start_epoch, start_iter = utils.get_start_iters(opt.loaded_latest_iter, len(dataloader))

    s_epoch = time()
    for epoch in range(start_epoch, opt.num_epochs):

        np.random.seed()  # reset seed

        s_time = time()
        for i, data_i in enumerate(dataloader):
            if not already_started and i < start_iter:
                continue
            already_started = True
            cur_iter = epoch * len(dataloader) + i

            # --- unpack data ---
            if opt.model in style_recon_model_list:
                # fake_label, coord_image, real_image, real_label, z_vec = preprocess_input_func(opt, data_i)
                data_dict = preprocess_input_func(opt, data_i)
                fake_label = data_dict['fake_label']
                coord_image = data_dict['coord_image']
                real_image = data_dict['real_image']
                real_label = data_dict['real_label']
                z_vec = data_dict['z_vec']

            elif 'style_interpolation' in opt.model:
                # real_label: ade20k_label
                # In Discriminator, we use only fake_label (blender_label)
                fake_label, coord_image, _, real_label = preprocess_input_func(opt, data_i)

                if opt.use_fixed_z_vec:
                    z_vec = random.choices(z_list, k=fake_label.size(0))
                    z_vec = torch.stack(z_vec, 0).to(fake_label.device)
                else:
                    # Sampling z_vec
                    z_vec = torch.randn(real_label.shape[0], opt.z_dim, dtype=torch.float32,
                                        device=fake_label.device)

                # Generate real_image from fake_label by using pre-trained OASIS
                with torch.no_grad():
                    real_image = pretrained_oasis_model(None, real_label, "generate", None, z_vec)

            #                # Un-normalize coordinate
            #                if opt.un_normalize_coord:
            #                    coord_image = coord_image * 4
            #                    print(coord_image.min(), coord_image.max())
            elif opt.model in oasis_3dcoord_model_list:
                fake_label, coord_image, real_image, real_label, pseudo_label = preprocess_input_func(opt, data_i)

                # Sampling z_vec
                z_vec = torch.randn(fake_label.shape[0], opt.z_dim, dtype=torch.float32,
                                    device=fake_label.device)

                # Generate real_image from fake_label by using pre-trained OASIS
                with torch.no_grad():
                    pseudo_image = pretrained_oasis_model(None, pseudo_label, "generate", None, z_vec)

            elif opt.model in surface_feat_model_list or opt.model in omni_model_list:
                feat_map = None
                if opt.use_3dfeat:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map, feat_map = preprocess_input_func(opt, data_i)
                else:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map = preprocess_input_func(opt, data_i)

                # Sampling z_vec
                z_vec = torch.randn(fake_label.shape[0], opt.z_dim, dtype=torch.float32,
                                    device=fake_label.device)

                # Generate real_image from fake_label by using pre-trained OASIS
                with torch.no_grad():
                    pseudo_image = pretrained_oasis_model(None, pseudo_label, "generate", None, z_vec)

            elif opt.model in usis_model_list:
                fake_label, coord_image, real_image, embed_idx_map = preprocess_input_func(opt, data_i)

                # Sampling z_vec
                z_vec = torch.randn(fake_label.shape[0], opt.z_dim, dtype=torch.float32,
                                    device=fake_label.device)
                pseudo_image = None

            elif opt.model in class_specific_model_list:
                feat_map = None
                if opt.use_3dfeat:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map, feat_map, cs_real_image, cs_real_label = preprocess_input_func(opt, data_i)
                else:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map, cs_real_image, cs_real_label = preprocess_input_func(opt, data_i)

                # Sampling z_vec
                z_vec = torch.randn(fake_label.shape[0], opt.z_dim, dtype=torch.float32,
                                    device=fake_label.device)
            else:
                fake_label, coord_image, real_image, real_label = preprocess_input_func(opt, data_i)

            # Normalize z_vec
            if opt.normalize_z_vec:
                if 'style_interpolation' in opt.model or opt.model in oasis_3dcoord_model_list:
                    z_vec = z_vec / 4.
                else:
                    z_vec = z_vec / torch.abs(z_vec).max(1)[0].unsqueeze(1)

            # --- generator update ---#
            losses_G_dict = None
            if i % opt.D_steps_per_G == 0:
                if opt.model in 'class_specific':
                    for net in model.module.class_specific_netG_list:
                        net.zero_grad()
                else:
                    model.module.netG.zero_grad()
                if opt.model in usis_model_list:
                    model.module.unet.zero_grad()

                if opt.model in style_recon_model_list or 'style_interpolation' in opt.model:
                    loss_G, losses_G_dict = model(fake_label, coord_image, real_image, real_label, z_vec, "losses_G",
                                                  losses_computer)
                elif opt.model in oasis_3dcoord_model_list:
                    loss_G, losses_G_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label,
                                                  "losses_G",
                                                  losses_computer, z_vec)
                elif opt.model in surface_feat_model_list:
                    loss_G, losses_G_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  "losses_G",
                                                  losses_computer, z_vec, feat_map)
                elif opt.model in usis_model_list:
                    loss_G, losses_G_dict = model(fake_label, coord_image,
                                                  real_image, embed_idx_map,
                                                  "losses_G",
                                                  losses_computer, z_vec)
                elif opt.model in omni_model_list:
                    loss_G, losses_G_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  "losses_G",
                                                  losses_computer, z_vec, feat_map)
                elif opt.model in class_specific_model_list:
                    loss_G, losses_G_dict = model(fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  cs_real_image, cs_real_label,
                                                  "losses_G",
                                                  losses_computer, z_vec, feat_map)
                else:
                    loss_G, losses_G_dict = model(fake_label, coord_image, real_image, real_label, "losses_G",
                                                  losses_computer)
                loss_G, losses_G_dict = loss_G.mean(), {name: loss.mean() if loss is not None else None for name, loss
                                                        in losses_G_dict.items()}
                loss_G.backward()
                optimizerG.step()
                if opt.model in usis_model_list:
                    optimizer_seg.step()

            # --- discriminator update ---#
            if opt.lr_d != 0.0:
                if opt.model in class_specific_model_list:
                    for net in model.module.class_specific_netD_list:
                        net.zero_grad()
                else:
                    if opt.use_netD_output1:
                        model.module.netD_output1.zero_grad()
                    if opt.use_output2:
                        model.module.netD_output2.zero_grad()

                if opt.model in style_recon_model_list or 'style_interpolation' in opt.model:
                    loss_D, losses_D_dict = model(fake_label, coord_image,
                                                  real_image, real_label, z_vec, "losses_D",
                                                  losses_computer)

                elif opt.model in oasis_3dcoord_model_list:
                    loss_D, losses_D_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label,
                                                  "losses_D",
                                                  losses_computer, z_vec)
                elif opt.model in surface_feat_model_list:
                    loss_D, losses_D_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  "losses_D",
                                                  losses_computer, z_vec, feat_map)
                elif opt.model in usis_model_list:
                    loss_D, losses_D_dict = model(fake_label, coord_image,
                                                  real_image, embed_idx_map,
                                                  "losses_D",
                                                  losses_computer, z_vec)
                elif opt.model in omni_model_list:
                    loss_D, losses_D_dict = model(pseudo_image, fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  "losses_D",
                                                  losses_computer, z_vec, feat_map)
                elif opt.model in class_specific_model_list:
                    loss_D, losses_D_dict = model(fake_label, coord_image,
                                                  real_image, real_label, embed_idx_map,
                                                  cs_real_image, cs_real_label,
                                                  "losses_D",
                                                  losses_computer, z_vec, feat_map)
                else:
                    loss_D, losses_D_dict = model(fake_label, coord_image, real_image, real_label, "losses_D",
                                                  losses_computer)
                loss_D, losses_D_dict = loss_D.mean(), {name: loss.mean() if loss is not None else None for name, loss
                                                        in
                                                        losses_D_dict.items()}
                loss_D.backward()
                if opt.add_ortho_regularize:
                    ortho(model.module.netD_output1, 1e-4)

                optimizerD.step()
            else:
                losses_D_dict = {}


            # --- stats update ---#
            if not opt.no_EMA:
                utils.update_EMA(model, cur_iter, dataloader, opt)
            # log every opt.freq_print iterations
            if cur_iter % opt.freq_print == 0:
                if opt.use_D_dontcare_zero_mask:
                    mask = (real_label[:, 0] == 1)
                    real_mask = torch.stack([mask, mask, mask], dim=1)
                    real_image[real_mask] = 0

                if opt.model in style_recon_model_list or 'style_interpolation' in opt.model:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, cur_iter, z_vec)
                elif opt.model in oasis_3dcoord_model_list:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, pseudo_image,
                                             cur_iter, z_vec)
                elif opt.model in surface_feat_model_list:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, pseudo_image,
                                             cur_iter, z_vec, embed_idx_map, feat_map)
                elif opt.model in usis_model_list:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, fake_label, real_image,
                                             cur_iter, z_vec, embed_idx_map)
                elif opt.model in omni_model_list:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, pseudo_image,
                                             cur_iter, z_vec, embed_idx_map, feat_map)
                elif opt.model in class_specific_model_list:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, None,
                                             cur_iter, z_vec, embed_idx_map, feat_map, cs_real_image, cs_real_label)
                else:
                    im_saver.visualize_batch(model, fake_label, coord_image, real_image, real_label, cur_iter)
                timer(epoch, cur_iter)
            # save every opt.freq_save_ckpt iterations
            if cur_iter % opt.freq_save_ckpt == 0:
                utils.save_networks(opt, cur_iter, model)
            # save every opt.freq_save_latest iterations
            if cur_iter % opt.freq_save_latest == 0:
                utils.save_networks(opt, cur_iter, model, latest=True)
            # compute fid every opt.freq_fid iterations
            if cur_iter % opt.freq_fid == 0 and cur_iter > 0:
                if opt.model in class_specific_model_list: 
                    pass
                else:
                    is_best = fid_computer.update(model, cur_iter, preprocess_input_func,
                                                  pretrained_oasis_model=pretrained_oasis_model)
                    if is_best:
                        utils.save_networks(opt, cur_iter, model, best=True)

            # plot and save every opt.freq_save_loss iterations
            visualizer_losses(cur_iter, {**losses_G_dict, **losses_D_dict})

            losses_G_dict = {name: np.round(x.detach().cpu().item(), 3) for name, x in losses_G_dict.items() if
                             x is not None}
            losses_D_dict = {name: np.round(x.detach().cpu().item(), 3) for name, x in losses_D_dict.items() if
                             x is not None}

            # log every 10 iterations
            e_time = time()
            if cur_iter % 10 == 0:
                print(f'iter [{i}/{epoch}/{cur_iter}] elp: {e_time - s_time:.3f}s '
                      f'loss_G: {loss_G.detach().cpu().numpy():.3f}'
                      # f'loss_G_adv: {losses_G_list[0]:.3f} \t'
                      # f'loss_G_pseudo_recon_l1: {losses_G_list[2]:.3f} \t'
                      # #f'loss_G_pseudo_recon_l2: {losses_G_list[3]:.3f} \t'
                      # f'loss_G_output12_recon_l1: {losses_G_list[4]:.3f} \n'
                      # f'loss_G_adv_output1: {losses_G_list[5]:.3f} \t'
                      # f'loss_G_adv_binary: {losses_G_list[6]:.3f} \n'
                      # f'loss_G_adv_binary_output1: {losses_G_list[7]:.3f} \n'
                      # f'\n'
                      # f'loss_D_fake: {losses_D_list[0]:.3f} \t'
                      # f'loss_D_real: {losses_D_list[1]:.3f} \t'
                      # f'loss_D_pseudo: {losses_D_list[2]:.3f} \n'
                      # #f'loss_D_lm: {losses_D_list[3]:.3f} \n'
                      # f'loss_D_fake_output1: {losses_D_list[4]:.3f} \t'
                      # f'loss_D_pseudo_output1: {losses_D_list[5]:.3f} \t'
                      # f'loss_D_binary: {losses_D_list[6]:.3f} \t'
                      # f'loss_D_binary_output1: {losses_D_list[7]:.3f} \t'
                      )
                print({**losses_G_dict, **losses_D_dict})
            s_time = time()

        # log every epoch
        e_epoch = time()
        print(f'epoch [{epoch}] elapsed time: {e_epoch - s_epoch:.3f}s')
        s_epoch = time()
    ##--- after training ---#
    # utils.update_EMA(model, cur_iter, dataloader, opt, force_run_stats=True)
    # utils.save_networks(opt, cur_iter, model)
    # utils.save_networks(opt, cur_iter, model, latest=True)
    # is_best = fid_computer.update(model, cur_iter)
    # if is_best:
    #    utils.save_networks(opt, cur_iter, model, best=True)
    print("The training has successfully finished")


if __name__ == '__main__':
    # --- read options ---#
    opt = config.read_arguments(train=True)

    main(opt)
