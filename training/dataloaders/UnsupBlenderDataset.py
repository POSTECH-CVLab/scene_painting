import glob
import random
import torch
import torchvision.transforms.functional as tvf

import os
from PIL import Image
import numpy as np

import OpenEXR
import Imath
import array


class UnsupBlenderDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
#        if opt.phase == "test" or for_metrics:
#            opt.load_size = 256
#        else:
#            opt.load_size = 286
        opt.load_size = 256 #  TODO: change for blender dataset (256)
        opt.crop_size = 256
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.label_dir = opt.label_dir
        self.coordinate_image_dir = opt.coordinate_image_dir
        self.pseudo_image_dir = opt.pseudo_image_dir
        self.pseudo_label_dir = opt.pseudo_label_dir

        self.opt = opt
        self.for_metrics = for_metrics
        self.labels, self.coord_images, self.pseudo_images, self.pseudo_labels = self.list_images()

        ### ade20k images
        ade20k_image_dir = os.path.join(opt.ade20k_dir, 'images')
        self.ade20k_images = sorted(glob.glob(
            os.path.join(ade20k_image_dir, '*.jpg')))

        if self.opt.use_point_embedding:
            embed_idx_map_dir = self.opt.point_embedding_dir
            self.embed_idx_maps = sorted(glob.glob(
                os.path.join(embed_idx_map_dir, '*.npy')))

    def __len__(self,):
        if self.for_metrics:
            return len(self.ade20k_images)
        else:
            return len(self.labels)

    def get_coordinate_image(self, path):
        file = OpenEXR.InputFile(path)

        dw = file.header()['dataWindow']
        sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        (R,G,B) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B") ]

        img = np.zeros((sz[1],sz[0],3), np.float64)
        img[:,:,0] = np.array(R).reshape(img.shape[0],-1)
        img[:,:,1] = np.array(G).reshape(img.shape[0],-1)
        img[:,:,2] = np.array(B).reshape(img.shape[0],-1)

        img = img.transpose(2, 0, 1)
        
        coordinate_image_tensor = torch.from_numpy(img).float()
        #coordainte_image_tensor = coordinate_image_tensor.transpose(2, 0, 1)
        return coordinate_image_tensor

    def __getitem__(self, idx):

        #fake_idx = idx % len(self.labels)
        if self.for_metrics:
            # ade20k 
            ade20k_image_path = self.ade20k_images[idx]
            ade20k_image = Image.open(ade20k_image_path).convert('RGB')
            ade20k_image = self.transforms_image(ade20k_image)

            return {"real_image": ade20k_image, "name": self.ade20k_images[idx]}

        fake_label = Image.open(self.labels[idx])
        fake_label = self.transforms_label(fake_label)
        coord_image = self.get_coordinate_image(self.coord_images[idx])

        # ade20k 
        ade20k_random_idcs = np.random.choice(len(self.ade20k_images), size=1)[0]
        ade20k_image_path = self.ade20k_images[ade20k_random_idcs]
        ade20k_image = Image.open(ade20k_image_path).convert('RGB')
        ade20k_image = self.transforms_image(ade20k_image)

        # point embedding
        if self.opt.use_point_embedding:
            embed_idx_map_path = self.embed_idx_maps[idx]
            embed_idx_map = np.load(embed_idx_map_path)
            embed_idx_map = torch.from_numpy(embed_idx_map).long()

            if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
                if random.random() < 0.5:
                    fake_label = fake_label.flip(-1)
                    coord_image = coord_image.flip(-1)
                    embed_idx_map = embed_idx_map.flip(-1)

            return {"label": fake_label, "coord_image": coord_image,
                    "real_image": ade20k_image, 
                    "name": self.labels[idx],
                    "idx": ade20k_random_idcs,
                    "embed_idx_map": embed_idx_map, }

        else:
            if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
                if random.random() < 0.5:
                    fake_label = fake_label.flip(-1)
                    coord_image = coord_image.flip(-1)

            return {"label": fake_label, "coord_image": coord_image,
                    "real_image": ade20k_image, 
                    "name": self.labels[idx],
                    "idx": ade20k_random_idcs}

    def list_images(self):
#        mode = "validation" if self.opt.phase == "test" or self.for_metrics else "training"
#        path_img = os.path.join(self.opt.dataroot, "images", mode)
#        path_lab = os.path.join(self.opt.dataroot, "annotations", mode)
#        img_list = os.listdir(path_img)
#        lab_list = os.listdir(path_lab)
#        img_list = [filename for filename in img_list if ".png" in filename or ".jpg" in filename]
#        lab_list = [filename for filename in lab_list if ".png" in filename or ".jpg" in filename]
#        images = sorted(img_list)
#        labels = sorted(lab_list)

        label_paths = sorted(glob.glob(self.label_dir + '/*'))
        coordinate_image_paths = sorted(glob.glob(self.coordinate_image_dir + '/*'))
        pseudo_image_paths = sorted(glob.glob(self.pseudo_image_dir + '/*'))
        pseudo_label_paths = sorted(glob.glob(self.pseudo_label_dir + '/*'))

        assert len(pseudo_image_paths)  == len(pseudo_label_paths), "different len of images and labels %s - %s" % (len(pseudo_image_paths), len(pseudo_label_paths))
        for i in range(len(pseudo_image_paths)):
            assert os.path.splitext(os.path.basename(pseudo_image_paths[i]))[0] == os.path.splitext(os.path.basename(pseudo_label_paths[i]))[0], '%s and %s are not matching' % (pseudo_image_paths[i], pseudo_label_paths[i])
        return label_paths, coordinate_image_paths, pseudo_image_paths, pseudo_label_paths

    def transforms_image(self, image):
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = tvf.resize(image, [new_width, new_height], tvf.InterpolationMode.BICUBIC)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
<<<<<<< HEAD
                image = tvf.hflip(image)
=======
                image = TR.functional.hflip(image)
>>>>>>> origin/feat/lsun_data
        # to tensor
        image = tvf.to_tensor(image)
        # normalize
        image = tvf.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image

    def transforms_label(self, label):
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        label = tvf.resize(label, [new_width, new_height], tvf.InterpolationMode.NEAREST)
        label = tvf.to_tensor(label)
        label = label * 255
        return label

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = tvf.resize(image, [new_width, new_height], tvf.InterpolationMode.BICUBIC)
        label = tvf.resize(label, [new_width, new_height], tvf.InterpolationMode.NEAREST)
#        # crop
#        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
#        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
#        image = image.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
#        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
#        # flip
#        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
#            if random.random() < 0.5:
<<<<<<< HEAD
#                image = tvf.hflip(image)
#                label = tvf.hflip(label)
=======
#                image = TR.functional.hflip(image)
#                label = TR.functional.hflip(label)
>>>>>>> origin/feat/lsun_data
        # to tensor
        image = tvf.to_tensor(image)
        label = tvf.to_tensor(label)
        # normalize
        image = tvf.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, label


