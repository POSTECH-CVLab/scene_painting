import models
import models.surface_feat_models as surface_feat_models 
import models.pretrained_oasis_models as pretrained_oasis_models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config

import torch

import os
import numpy as np
import random
from PIL import Image

from train_etc_util import style_recon_model_list, oasis_3dcoord_model_list, surface_feat_model_list, omni_model_list


def main(opt):
    # --- create dataloader ---#
    _, dataloader_val = dataloaders.get_dataloaders(opt)

    # --- create utils ---#
    if opt.model in style_recon_model_list:
        image_saver = utils.results_saver(opt, ['label', 'image'] + ['oasis_images_seed_' + str(i) for i in range(9)])
    else:
        image_saver = utils.results_saver(opt)

    # --- create models ---#
    use_pretrained_oasis_model = False
    if opt.model == 'surface_feat':
        model = surface_feat_models.OASIS_model(opt)
        preprocess_input_func = surface_feat_models.preprocess_input
        use_pretrained_oasis_model = True
    else:
        raise ValueError('No model')
    model = models.util.put_on_multi_gpus(model, opt)
    model.eval()

    # --- create pretrained OASIS model ---#
    pretrained_oasis_model = None
    if use_pretrained_oasis_model:
        pretrained_oasis_model = pretrained_oasis_models.OASIS_model(opt)
        pretrained_oasis_model = models.util.put_on_multi_gpus(pretrained_oasis_model, opt)
        pretrained_oasis_model.eval()

    # ---  z_vec ---
    if opt.use_fixed_z_vec:
        torch.manual_seed(0)
        z_list = []
        num_z_list = opt.num_z_vec
        for i in range(num_z_list):
            z = torch.randn(opt.z_dim, dtype=torch.float32)
            z_list.append(z)
        for i in range(num_z_list):
            print(z_list[i][0])

        selected_z_list = list(range(num_z_list))
        if not -1 in opt.z_list:
            selected_z_list = opt.z_list

    else:
        torch.manual_seed(0)
        #raise ValueError('Use use_fixed_z_vec')
        print('Not using use_fixed_z_vec option !')

    base_path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)

    # --- iterate over validation set ---#
    if opt.use_fixed_z_vec:
        for z_idx, z in enumerate(z_list):
            if not z_idx in selected_z_list: 
                continue
            label_save_dir = os.path.join(base_path, str(z_idx), 'label')
            os.makedirs(label_save_dir, exist_ok=True)
            gen_save_dir = os.path.join(base_path, str(z_idx), 'generated_image')
            os.makedirs(gen_save_dir, exist_ok=True)
            oasis_save_dir = os.path.join(base_path, str(z_idx), 'oasis_image')
            os.makedirs(oasis_save_dir, exist_ok=True)
            
            if opt.model in surface_feat_model_list:
                view = None
                if opt.surface_feat_model_view_encoding:
                    data_sample = dataloader_val.dataset[0]
                
                    if opt.view_encoding_input_type == 'label':
                        label_map = torch.unsqueeze(data_sample['label'], 0).long().cuda()
                        bs, _, h, w = label_map.size()
                        nc = opt.semantic_nc
                        if opt.gpu_ids != "-1":
                            input_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
                        else:
                            input_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
                        input_semantics = input_label.scatter_(1, label_map, 1.0)
                        view = input_semantics 
                    elif opt.view_encoding_input_type == 'pseudo_image':
                        pseudo_label_map = torch.unsqueeze(data_sample['pseudo_label'], 0).long().cuda()
                        bs, _, h, w = pseudo_label_map.size()
                        nc = 151
                        if opt.gpu_ids != "-1":
                            pseudo_label = torch.zeros(bs, nc, h, w, dtype=torch.float).cuda()
                        else:
                            pseudo_label = torch.zeros(bs, nc, h, w, dtype=torch.float)
                        pseudo_semantics = pseudo_label.scatter_(1, pseudo_label_map, 1.0)

                        z_vec = z.expand(1, opt.z_dim)
                        with torch.no_grad():
                            pseudo_image = pretrained_oasis_model(None, pseudo_semantics, "generate", None, z_vec)
                        view = pseudo_image

            for i, data_i in enumerate(dataloader_val):
                if opt.model in style_recon_model_list:
                    fake_label, coord_image, real_image, real_label, z_vec = preprocess_input_func(opt, data_i)
                elif opt.model in oasis_3dcoord_model_list:
                    fake_label, coord_image, real_image, real_label, pseudo_label = preprocess_input_func(opt, data_i)
                elif opt.model in surface_feat_model_list:
                    feat_map = None
                    if opt.use_3dfeat:
                        fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map, feat_map = preprocess_input_func(opt, data_i)
                    else:
                        fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map = preprocess_input_func(opt, data_i)
                else:
                    fake_label, coord_image, real_image, real_label = preprocess_input_func(opt, data_i)

                if 'style_interpolation' in opt.model:
                    z_vec = z.expand(fake_label.size(0), opt.z_dim)
                    with torch.no_grad():
                        real_image = pretrained_oasis_model(None, real_label, "generate", None, z_vec)
                elif opt.model in oasis_3dcoord_model_list:
                    z_vec = z.expand(fake_label.size(0), opt.z_dim)
                    with torch.no_grad():
                        real_image = pretrained_oasis_model(None, pseudo_label, "generate", None, z_vec)
                elif opt.model in surface_feat_model_list:
                    z_vec = z.expand(fake_label.size(0), opt.z_dim)
                    with torch.no_grad():
                        real_image = pretrained_oasis_model(None, pseudo_label, "generate", None, z_vec)

                if opt.normalize_z_vec:
                    z_vec = z_vec / 4

                if opt.model != 'original_oasis':
                    if opt.model in style_recon_model_list or 'style_interpolation' in opt.model:
                        generated = model(fake_label, coord_image, None, None, z_vec, "generate", None)
                    elif opt.model in oasis_3dcoord_model_list:
                        if opt.output_type == 'output1':
                            generated, _ = model(None, fake_label, coord_image, None, None, "generate", None, z_vec)
                        elif opt.output_type == 'output2':
                            _, generated = model(None, fake_label, coord_image, None, None, "generate", None, z_vec)
                        else:
                            raise ValueError('no output_type')
                    elif opt.model in surface_feat_model_list:
                        generated, _ = model(None, fake_label, coord_image, None, None, embed_idx_map, "generate", None, z_vec, feat_map, view=view)
                    else:
                        generated = model(fake_label, coord_image, None, None, "generate", None)
                else:
                    z_vec = z.expand(fake_label.size(0), opt.z_dim)
                    generated = model(None, fake_label, "generate", None, z_vec)

                if opt.model in style_recon_model_list:
                    floder_names = [os.path.split(os.path.split(fils_path)[0])[1] for fils_path in data_i['name']]
                    image_saver(fake_label, generated, data_i["name"], floder_names=floder_names)
                else:
                    image_saver(fake_label, generated, data_i["name"])

                for j in range(len(fake_label)):
                    save_name = os.path.split(data_i["name"][j])[-1]
                    label_lab = utils.tens_to_lab(fake_label[j], image_saver.num_cl)
                    label_lab = Image.fromarray(label_lab.astype(np.uint8))
                    label_lab.save(os.path.join(label_save_dir, save_name))

                    generated_im = utils.tens_to_im(generated[j]) * 255
                    generated_im = Image.fromarray(generated_im.astype(np.uint8))
                    generated_im.save(os.path.join(gen_save_dir, save_name))

                    oasis_im = utils.tens_to_im(real_image[j]) * 255
                    oasis_im = Image.fromarray(oasis_im.astype(np.uint8))
                    oasis_im.save(os.path.join(oasis_save_dir, save_name))
    else:
        z_idx = 0
        label_save_dir = os.path.join(base_path, str(z_idx), 'label')
        os.makedirs(label_save_dir, exist_ok=True)
        gen_save_dir = os.path.join(base_path, str(z_idx), 'generated_image')
        os.makedirs(gen_save_dir, exist_ok=True)
        oasis_save_dir = os.path.join(base_path, str(z_idx), 'oasis_image')
        os.makedirs(oasis_save_dir, exist_ok=True)

        for i, data_i in enumerate(dataloader_val):
            if opt.model in surface_feat_model_list:
                feat_map = None
                if opt.use_3dfeat:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map, feat_map = preprocess_input_func(opt, data_i)
                else:
                    fake_label, coord_image, real_image, real_label, pseudo_label, embed_idx_map = preprocess_input_func(opt, data_i)
            else:
                fake_label, coord_image, real_image, real_label = preprocess_input_func(opt, data_i)

            if opt.model in surface_feat_model_list:
                z = torch.randn(opt.z_dim, dtype=torch.float32)
                z_vec = z.expand(fake_label.size(0), opt.z_dim)
                with torch.no_grad():
                    real_image = pretrained_oasis_model(None, pseudo_label, "generate", None, z_vec)

            if opt.normalize_z_vec:
                z_vec = z_vec / 4

            if opt.model != 'original_oasis':
                if opt.model in surface_feat_model_list:
                    generated, _ = model(None, fake_label, coord_image, None, None, embed_idx_map, "generate", None, z_vec, feat_map)
                else:
                    generated = model(fake_label, coord_image, None, None, "generate", None)
            else:
                z_vec = z.expand(fake_label.size(0), opt.z_dim)
                generated = model(None, fake_label, "generate", None, z_vec)

            if opt.model in style_recon_model_list:
                floder_names = [os.path.split(os.path.split(fils_path)[0])[1] for fils_path in data_i['name']]
                image_saver(fake_label, generated, data_i["name"], floder_names=floder_names)
            else:
                image_saver(fake_label, generated, data_i["name"])

            for j in range(len(fake_label)):
                save_name = os.path.split(data_i["name"][j])[-1]
                label_lab = utils.tens_to_lab(fake_label[j], image_saver.num_cl)
                label_lab = Image.fromarray(label_lab.astype(np.uint8))
                label_lab.save(os.path.join(label_save_dir, save_name))

                generated_im = utils.tens_to_im(generated[j]) * 255
                generated_im = Image.fromarray(generated_im.astype(np.uint8))
                generated_im.save(os.path.join(gen_save_dir, save_name))

                oasis_im = utils.tens_to_im(real_image[j]) * 255
                oasis_im = Image.fromarray(oasis_im.astype(np.uint8))
                oasis_im.save(os.path.join(oasis_save_dir, save_name))

            if i % 100 == 0:
                print(f'[{i} / {len(dataloader_val)}]')


if __name__ == '__main__':
    # --- read options ---#
    opt = config.read_arguments(train=False)

    main(opt)
