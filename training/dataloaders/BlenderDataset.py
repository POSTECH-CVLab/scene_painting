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

from train_etc_util import class_specific_model_list

class BlenderDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
        opt.load_size = 256 
        opt.crop_size = 256
        opt.contain_dontcare_label = True
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.label_dir = opt.label_dir
        self.coordinate_image_dir = opt.coordinate_image_dir
        self.pseudo_label_dir = opt.pseudo_label_dir

        self.opt = opt
        self.for_metrics = for_metrics
        self.labels, self.coord_images, self.pseudo_labels = self.list_images()

        ### real images
        real_image_dir = opt.real_image_dir
        real_label_dir = opt.real_label_dir
        self.real_images = sorted(glob.glob(
            os.path.join(real_image_dir, '*.jpg')))
        self.real_labels = sorted(glob.glob(
            os.path.join(real_label_dir, '*.png')))
        
        if self.opt.model in class_specific_model_list:
            class_specific_real_image_dir = opt.class_specific_real_image_dir
            class_specific_real_label_dir = opt.class_specific_real_label_dir
            self.class_specific_real_images = sorted(glob.glob(
                os.path.join(class_specific_real_image_dir, '*.jpg')))
            self.class_specific_real_labels = sorted(glob.glob(
                os.path.join(class_specific_real_label_dir, '*.png')))

        if self.opt.use_point_embedding:
            embed_idx_map_dir = self.opt.point_embedding_dir
            self.embed_idx_maps = sorted(glob.glob(
                os.path.join(embed_idx_map_dir, '*.npy')))

        if self.opt.use_3dfeat:
            feat_dir = self.opt.feat_dir
            self.feat_maps = sorted(glob.glob(
                os.path.join(feat_dir, '*.npy')))

    def __len__(self,):
        #return len(self.labels)
        if self.for_metrics:
            return len(self.real_images)
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

        if self.opt.surface_feat_model_quantize != -1:
            img = np.round(img, self.opt.surface_feat_model_quantize)
        else:
            pass
        
        coordinate_image_tensor = torch.from_numpy(img).float()
        return coordinate_image_tensor

    def __getitem__(self, idx):

        if self.for_metrics:
            # real 
            real_image_path = self.real_images[idx]
            real_label_path = self.real_labels[idx]
            real_image = Image.open(real_image_path).convert('RGB')
            real_label = Image.open(real_label_path)
            real_image, real_label = self.transforms(real_image, real_label)
            real_label = real_label * 255

            return {"real_image": real_image, "real_label": real_label,
                    "name": self.real_labels[idx]}

        fake_label = Image.open(self.labels[idx])
        fake_label = self.transforms_label(fake_label)

        coord_image = self.get_coordinate_image(self.coord_images[idx])

        pseudo_label = Image.open(self.pseudo_labels[idx])
        pseudo_label = self.transforms_label(pseudo_label)

        do_flip = False
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                do_flip = True
        
        if do_flip:
            fake_label = fake_label.flip(-1)
            coord_image = coord_image.flip(-1)
            pseudo_label = pseudo_label.flip(-1)

        result = {} 

        # point embedding
        if self.opt.use_point_embedding:
            embed_idx_map_path = self.embed_idx_maps[idx]
            embed_idx_map = np.load(embed_idx_map_path)
            embed_idx_map = torch.from_numpy(embed_idx_map).long()
            if do_flip:
                embed_idx_map = embed_idx_map.flip(-1)
            result["embed_idx_map"] = embed_idx_map
    
        # feat map
        if self.opt.use_3dfeat:
            feat_map_path = self.feat_maps[idx]
            feat_map = np.load(feat_map_path)
            feat_map = torch.from_numpy(feat_map).float()
            feat_map = feat_map.transpose(1,2).transpose(0,1)
            if do_flip:
                feat_map = feat_map.flip(-1)
            result["feat_map"] = feat_map

        # real 
        real_random_idcs = np.random.choice(len(self.real_images), size=1)[0]
        real_image_path = self.real_images[real_random_idcs]
        real_label_path = self.real_labels[real_random_idcs]
        real_image = Image.open(real_image_path).convert('RGB')
        real_label = Image.open(real_label_path)
        real_image, real_label = self.transforms(real_image, real_label)
        real_label = real_label * 255

        result["label"] = fake_label
        result["coord_image"] = coord_image 
        result["pseudo_label"] = pseudo_label
        result["real_image"] = real_image
        result["real_label"] = real_label
        result["name"] = self.labels[idx]
        result["real_idx"] = real_random_idcs

        if self.opt.model in class_specific_model_list:
            cs_real_random_idcs = np.random.choice(len(self.class_specific_real_images), size=1)[0]
            cs_real_image_path = self.class_specific_real_images[cs_real_random_idcs]
            cs_real_label_path = self.class_specific_real_labels[cs_real_random_idcs]
            cs_real_image = Image.open(cs_real_image_path).convert('RGB')
            cs_real_label = Image.open(cs_real_label_path)
            cs_real_image, cs_real_label = self.transforms(cs_real_image, cs_real_label)
            cs_real_label = cs_real_label * 255
            result["cs_real_image"] = cs_real_image
            result["cs_real_label"] = cs_real_label

        return result

    def list_images(self):

        label_paths = sorted(glob.glob(self.label_dir + '/*'))
        coordinate_image_paths = sorted(glob.glob(self.coordinate_image_dir + '/*'))
        pseudo_label_paths = sorted(glob.glob(self.pseudo_label_dir + '/*'))

        assert len(label_paths)  == len(pseudo_label_paths), "different len of labels and pseudo_labels %s - %s" % (len(label_paths), len(pseudo_label_paths))
        assert len(label_paths)  == len(coordinate_image_paths), "different len of labels and coord_image %s - %s" % (len(label_paths), len(coordinate_image_paths))

        for i in range(len(label_paths)):
            assert os.path.splitext(os.path.basename(label_paths[i]))[0] == os.path.splitext(os.path.basename(pseudo_label_paths[i]))[0], '%s and %s are not matching' % (label_paths[i], pseudo_label_paths[i])

        for i in range(len(label_paths)):
            assert os.path.splitext(os.path.basename(label_paths[i]))[0] == (os.path.basename(coordinate_image_paths[i])).split('.')[0], '%s and %s are not matching' % (label_paths[i], coordinate_image_paths[i])

        return label_paths, coordinate_image_paths, pseudo_label_paths

    def transforms(self, image, label):
        assert image.size == label.size
        # resize
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = tvf.resize(image, [new_width, new_height], tvf.InterpolationMode.BICUBIC)
        label = tvf.resize(label, [new_width, new_height], tvf.InterpolationMode.NEAREST)

        # to tensor
        image = tvf.to_tensor(image)
        label = tvf.to_tensor(label)
        # normalize
        image = tvf.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, label

    def transforms_image(self, image):
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        image = tvf.resize(image, [new_width, new_height], tvf.InterpolationMode.BICUBIC)
        # flip
        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
            if random.random() < 0.5:
                image = tvf.hflip(image)
                image = TR.functional.hflip(image)
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
