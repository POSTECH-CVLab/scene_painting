import torch
import numpy as np
import random
import time
import os
import models.models as models
import matplotlib.pyplot as plt
from PIL import Image

from train_etc_util import (style_recon_model_list, oasis_3dcoord_model_list,
                            surface_feat_model_list, usis_model_list, omni_model_list,
                            class_specific_model_list)


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_start_iters(start_iter, dataset_size):
    if start_iter == 0:
        return 0, 0
    start_epoch = (start_iter + 1) // dataset_size
    start_iter = (start_iter + 1) % dataset_size
    return start_epoch, start_iter


class results_saver():
    def __init__(self, opt, folder_names=('label', 'image')):
        self.num_cl = opt.label_nc + 2
        # self.is_style_model = opt.model in style_recon_model_list

        base_path = os.path.join(opt.results_dir, opt.name, opt.ckpt_iter)

        self.folder_names = folder_names
        # self.folder_names = ['label', 'image'] + \
        #                     ['oasis_images_seed_' + str(i) for i in range(9) if
        #                                           self.is_style_model]

        # self.path_to_save = {"label": self.path_label, "image": self.path_image}
        self.path_to_save = {name: os.path.join(base_path, name) for name in self.folder_names}

        for path in self.path_to_save.values():
            os.makedirs(path, exist_ok=True)

    def __call__(self, label, generated, file_paths, floder_names='image'):
        assert len(label) == len(generated)

        if isinstance(floder_names, str):
            floder_names = [floder_names] * len(file_paths)

        for i in range(len(label)):
            lable_lab = tens_to_lab(label[i], self.num_cl)

            file_name = os.path.split(file_paths[i])[-1]
            self.save_im(lable_lab, "label", file_name)
            generated_im = tens_to_im(generated[i]) * 255

            self.save_im(generated_im, floder_names[i], file_name)

    def save_im(self, im, folder_name, file_name):
        im = Image.fromarray(im.astype(np.uint8))
        im.save(os.path.join(self.path_to_save[folder_name], file_name.replace('.jpg', '.png')))


class timer():
    def __init__(self, opt):
        self.prev_time = time.time()
        self.prev_epoch = 0
        self.num_epochs = opt.num_epochs
        self.file_name = os.path.join(opt.checkpoints_dir, opt.name, "progress.txt")

    def __call__(self, epoch, cur_iter):
        if cur_iter != 0:
            avg = (time.time() - self.prev_time) / (cur_iter - self.prev_epoch)
        else:
            avg = 0
        self.prev_time = time.time()
        self.prev_epoch = cur_iter

        with open(self.file_name, "a") as log_file:
            log_file.write('[epoch %d/%d - iter %d], time:%.3f \n' % (epoch, self.num_epochs, cur_iter, avg))
        print('[epoch %d/%d - iter %d], time:%.3f' % (epoch, self.num_epochs, cur_iter, avg))
        return avg


class losses_saver():
    def __init__(self, opt, name_list):
        self.opt = opt
        self.freq_smooth_loss = opt.freq_smooth_loss
        self.freq_save_loss = opt.freq_save_loss

        self.losses_dict = {k: [] for k in name_list}
        self.cur_estimates_dict = {k: 0.0 for k in name_list}

        self.path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses")

        os.makedirs(self.path, exist_ok=True)
        for idx, k in enumerate(self.losses_dict.keys()):
            if opt.continue_train:
                self.losses_dict[k] = np.load(self.path + "/losses.npy", allow_pickle=True)[idx].tolist()
            else:
                self.losses_dict[k] = list()

    def __call__(self, epoch, losses_dict):
        # convert to numpy and accumulate cur_estimates
        for name in self.losses_dict.keys():
            loss = losses_dict.get(name, None)
            if loss is None:
                self.cur_estimates_dict[name] = None
            else:
                self.cur_estimates_dict[name] += loss.detach().cpu().numpy()

        # smoothing
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss - 1:
            for name, list_ in self.losses_dict.items():
                # for i in range(len(self.name_list)):
                if not self.cur_estimates_dict[name] is None:
                    list_.append(self.cur_estimates_dict[name] / self.opt.freq_smooth_loss)
                    self.cur_estimates_dict[name] = 0.0
                else:
                    list_.append(np.NAN)

        # plot
        if epoch % self.freq_save_loss == self.freq_save_loss - 1:
            self.plot_losses()
            print()
            np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", "losses"),
                    list(self.losses_dict.values()))

    def plot_losses(self):
        # plot each loss
        for name in self.losses_dict.keys():
            fig, ax = plt.subplots(1)
            n = np.array(range(len(self.losses_dict[name]))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], self.losses_dict[name][1:])
            plt.ylabel('loss')
            plt.xlabel('epochs')

            plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", '%s.png' % (name)), dpi=600)
            plt.close(fig)

        # plot combined loss
        fig, ax = plt.subplots(1)
        for name, list_ in self.losses_dict.items():
            if np.isnan(list_[0]):
                continue
            n = np.array(range(len(list_))) * self.opt.freq_smooth_loss
            plt.plot(n[1:], list_[1:], label=name)
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.opt.checkpoints_dir, self.opt.name, "losses", 'combined.png'), dpi=600)
        plt.close(fig)


def update_EMA(model, cur_iter, dataloader, opt, force_run_stats=False):
    model_ = model.module if isinstance(model, torch.nn.DataParallel) else model

    # update weights based on new generator weights
    with torch.no_grad():
        for key in model_.netEMA.state_dict():
            model_.netEMA.state_dict()[key].data.copy_(
                model_.netEMA.state_dict()[key].data * opt.EMA_decay +
                model_.netG.state_dict()[key].data * (1 - opt.EMA_decay)
            )
    # collect running stats for batchnorm before FID computation, image or network saving
    condition_run_stats = (force_run_stats or
                           cur_iter % opt.freq_print == 0 or
                           cur_iter % opt.freq_fid == 0 or
                           cur_iter % opt.freq_save_ckpt == 0 or
                           cur_iter % opt.freq_save_latest == 0
                           )
    if condition_run_stats:
        with torch.no_grad():
            num_upd = 0
            for i, data_i in enumerate(dataloader):
                fake_label, coord_image, real_image, real_label = models.preprocess_input(opt, data_i)
                fake = model_.netEMA(fake_label, coord_image)
                num_upd += 1
                if num_upd > 50:
                    break


def save_networks(opt, cur_iter, model, latest=False, best=False):
    path = os.path.join(opt.checkpoints_dir, opt.name, "models")
    os.makedirs(path, exist_ok=True)

    model_ = model.module if isinstance(model, torch.nn.DataParallel) else model

    if opt.model in class_specific_model_list:
        if latest:
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netG_list[idx].state_dict(), path + f"/latest_G_{idx:03d}.pth")
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netD_list[idx].state_dict(), path + f"/latest_D_{idx:03d}.pth")
            if not opt.no_EMA:
                import pdb; pdb.set_trace()
                torch.save(model_.netEMA.state_dict(), path + '/%s_EMA.pth' % ("latest"))
            with open(os.path.join(opt.checkpoints_dir, opt.name) + "/latest_iter.txt", "w") as f:
                f.write(str(cur_iter))
        elif best:
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netG_list[idx].state_dict(), path + f"/best_G_{idx:03d}.pth")
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netD_list[idx].state_dict(), path + f"/best_D_{idx:03d}.pth")
            if not opt.no_EMA:
                import pdb; pdb.set_trace()
                torch.save(model_.netEMA.state_dict(), path + '/%s_EMA.pth' % ("best"))
            with open(os.path.join(opt.checkpoints_dir, opt.name) + "/best_iter.txt", "w") as f:
                f.write(str(cur_iter))
        else:
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netG_list[idx].state_dict(), path + f"/{cur_iter}_G_{idx:03d}.pth")
            for idx, label in enumerate(model_.class_specific_label_list):
                torch.save(model_.class_specific_netD_list[idx].state_dict(), path + f"/{cur_iter}_D_{idx:03d}.pth")
            if not opt.no_EMA:
                import pdb; pdb.set_trace()
                torch.save(model_.netEMA.state_dict(), path + '/%d_EMA.pth' % (cur_iter))
    else:
        if latest:
            torch.save(model_.netG.state_dict(), path + '/%s_G.pth' % ("latest"))
            if opt.use_netD_output1:
                torch.save(model_.netD_output1.state_dict(), path + '/%s_D_output1.pth' % ("latest"))
            if opt.use_output2:
                torch.save(model_.netD_output2.state_dict(), path + '/%s_D_output2.pth' % ("latest"))
            if opt.model in usis_model_list:
                torch.save(model_.unet.state_dict(), path + '/%s_unet.pth' % ("latest"))
            if not opt.no_EMA:
                torch.save(model_.netEMA.state_dict(), path + '/%s_EMA.pth' % ("latest"))
            with open(os.path.join(opt.checkpoints_dir, opt.name) + "/latest_iter.txt", "w") as f:
                f.write(str(cur_iter))
        elif best:
            torch.save(model_.netG.state_dict(), path + '/%s_G.pth' % ("best"))
            if opt.use_netD_output1:
                torch.save(model_.netD_output1.state_dict(), path + '/%s_D_output1.pth' % ("best"))
            if opt.use_output2:
                torch.save(model_.netD_output2.state_dict(), path + '/%s_D_output2.pth' % ("best"))
            if opt.model in usis_model_list:
                torch.save(model_.unet.state_dict(), path + '/%s_unet.pth' % ("best"))
            if not opt.no_EMA:
                torch.save(model_.netEMA.state_dict(), path + '/%s_EMA.pth' % ("best"))
            with open(os.path.join(opt.checkpoints_dir, opt.name) + "/best_iter.txt", "w") as f:
                f.write(str(cur_iter))
        else:
            torch.save(model_.netG.state_dict(), path + '/%d_G.pth' % (cur_iter))
            if opt.use_netD_output1:
                torch.save(model_.netD_output1.state_dict(), path + '/%d_D_output1.pth' % (cur_iter))
            if opt.use_output2:
                torch.save(model_.netD_output2.state_dict(), path + '/%d_D_output2.pth' % (cur_iter))
            if opt.model in usis_model_list:
                torch.save(model_.unet.state_dict(), path + '/%d_unet.pth' % (cur_iter))
            if not opt.no_EMA:
                torch.save(model_.netEMA.state_dict(), path + '/%d_EMA.pth' % (cur_iter))


class image_saver():
    def __init__(self, opt):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.path = os.path.join(opt.checkpoints_dir, opt.name, "images") + "/"
        self.opt = opt
        self.num_cl = opt.label_nc + 2
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, fake_label, coord_image, real_image, real_label, pseudo_image, cur_iter,
                        z_vec=None, embed_idx_map=None, feat_map=None, cs_real_image=None, cs_real_label=None):
        self.save_images(fake_label, "fake_label", cur_iter, is_label=True)
        self.save_images(real_label, "real_label", cur_iter, is_label=True)
        with torch.no_grad():
            model.eval()
            if self.opt.model in style_recon_model_list or 'style_interpolation' in self.opt.model:
                fake = model(fake_label, coord_image, None, None, z_vec, "generate", None)
            elif self.opt.model in oasis_3dcoord_model_list:
                fake1, fake2 = model(None, fake_label, coord_image, None, None, "generate", None, z_vec)
            elif self.opt.model in surface_feat_model_list:
                fake1, fake2 = model(pseudo_image, fake_label, coord_image, None, None, embed_idx_map, "generate", None, z_vec, feat_map)
            elif self.opt.model in usis_model_list:
                fake, _ = model(fake_label, coord_image, None, embed_idx_map, "generate", None, z_vec)
            elif self.opt.model in omni_model_list:
                fake = model(None, fake_label, coord_image, None, None, embed_idx_map, "generate", None, z_vec, feat_map)
            elif self.opt.model in class_specific_model_list:
                fake = model(fake_label, coord_image, None, None, embed_idx_map, cs_real_image, cs_real_label, "generate", None, z_vec, feat_map)
                cs_fake_cr, cs_real_image_cr, cs_real_label_cr = model(fake_label, coord_image, None, None, embed_idx_map, cs_real_image, cs_real_label, "debug", None, z_vec, feat_map)
            else:
                fake = model(fake_label, coord_image, None, None, "generate", None)

            if self.opt.model in oasis_3dcoord_model_list or self.opt.model in surface_feat_model_list:
                self.save_images(fake1, "fake1", cur_iter)
                self.save_images(fake2, "fake2", cur_iter)
            elif self.opt.model in class_specific_model_list:
                self.save_images(fake, "fake", cur_iter)
                self.save_images(cs_fake_cr, "cs_fake", cur_iter)
                self.save_images(cs_real_image_cr, "cs_real_image", cur_iter)
                self.save_images(cs_real_label_cr, "cs_real_label", cur_iter, is_label=True)
            else:
                self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.opt.no_EMA:
                model.eval()
                model_ = model.module if isinstance(model, torch.nn.DataParallel) else model
                fake = model_.netEMA(fake_label, coord_image)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

        if self.opt.use_D_dontcare_fake_mask:
            mask = (real_label[:, 0] == 1)
            real_mask = torch.stack([mask, mask, mask], dim=1)
            real_image[real_mask] = fake[real_mask]
        self.save_images(real_image, "real_image", cur_iter)
        if not self.opt.model in class_specific_model_list:
            self.save_images(pseudo_image, "pseudo_image", cur_iter)

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i + 1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path + str(cur_iter) + "_" + name)
        plt.close()


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy



def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image


def labelcolormap(N):
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
