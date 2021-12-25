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


class StyleBlenderDataset(torch.utils.data.Dataset):
    def __init__(self, opt, for_metrics):
#        if opt.phase == "test" or for_metrics:
#            opt.load_size = 256
#        else:
#            opt.load_size = 286
        opt.load_size = 256 #  TODO: change for blender dataset (256)
        opt.crop_size = 256
        #opt.label_nc = 150
        #opt.label_nc = 13
        opt.contain_dontcare_label = True
        #opt.semantic_nc = 151 # label_nc + unknown
        #opt.semantic_nc = 14 # label_nc + unknown
        opt.cache_filelist_read = False
        opt.cache_filelist_write = False
        opt.aspect_ratio = 1.0

        self.label_dir = opt.label_dir
        self.coordinate_image_dir = opt.coordinate_image_dir
        #self.real_image_dir = opt.real_image_dir
        self.real_label_dir = opt.real_label_dir

        self.opt = opt
        self.for_metrics = for_metrics
        self.labels, self.coord_images, self.real_images, self.real_labels, self.z_paths = self.list_images()

    def __len__(self,):
        #return len(self.labels)
        return len(self.real_images)

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

        fake_label = Image.open(self.labels[idx])
        new_width, new_height = (self.opt.load_size, self.opt.load_size)
        fake_label = tvf.resize(fake_label, [new_width, new_height], tvf.InterpolationMode.NEAREST)
        fake_label = tvf.to_tensor(fake_label)
        fake_label = fake_label * 255

        coord_image = self.get_coordinate_image(self.coord_images[idx])

        #real_idx = np.random.randint(0, len(self.real_images))
        real_image = Image.open(self.real_images[idx]).convert('RGB')
        real_label = Image.open(self.real_labels[idx])

        real_image, real_label = self.transforms(real_image, real_label)
        real_label = real_label * 255

        z_vec = torch.tensor(np.load(self.z_paths[idx]), dtype=torch.float)

        return {"label": fake_label, "coord_image": coord_image,
                "real_image": real_image, "real_label": real_label, 
                "z_vec": z_vec, "name": self.real_images[idx]}

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


        oasis_image_folder_list = sorted(glob.glob(self.opt.dataroot + '/oasis_images_seed_*'))
        real_image_paths = []
        for oasis_image_folder in oasis_image_folder_list:
            real_image_paths += sorted(glob.glob(oasis_image_folder + '/*.png'))

        label_paths = []
        coordinate_image_paths = []
        real_label_paths = []
        for oasis_image_folder in oasis_image_folder_list:
            real_label_paths += sorted(glob.glob(self.real_label_dir + '/*.png'))
            label_paths += sorted(glob.glob(self.label_dir + '/*.png'))
            coordinate_image_paths += sorted(glob.glob(self.coordinate_image_dir + '/*.exr'))

        z_vecs = sorted(glob.glob(self.opt.dataroot + '/z_vecs/*.npy'))

        z_paths = []
        for i, oasis_image_folder in enumerate(oasis_image_folder_list):
            count = len(sorted(glob.glob(oasis_image_folder + '/*.png')))
            z_paths += [z_vecs[i] for _ in range(count)]

        assert len(real_image_paths)  == len(real_label_paths), "different len of images and labels %s - %s" % (len(real_image_paths), len(real_label_paths))
        for i in range(len(real_image_paths)):
            assert os.path.splitext(os.path.basename(real_image_paths[i]))[0] == os.path.splitext(os.path.basename(real_label_paths[i]))[0], '%s and %s are not matching' % (real_image_paths[i], real_label_paths[i])
        return label_paths, coordinate_image_paths, real_image_paths, real_label_paths, z_paths

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
#                image = tvf.hflip(image)
#                label = tvf.hflip(label)
        # to tensor
        image = tvf.to_tensor(image)
        label = tvf.to_tensor(label)
        # normalize
        image = tvf.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        return image, label

    def transforms_coord_and_label(self, coord_image, label):
        # resize
#        new_width, new_height = (self.opt.load_size, self.opt.load_size)
#        coord_image = torch.nn.functional.interpolate(coord_image, size=(new_width, new_height), 
#                                                      mode='bicubic', align_corners=False)
#        label = tvf.resize(label, [new_width, new_height], tvf.InterpolationMode.NEAREST)
        # crop
#        crop_x = random.randint(0, np.maximum(0, new_width -  self.opt.crop_size))
#        crop_y = random.randint(0, np.maximum(0, new_height - self.opt.crop_size))
#        coord_image = coord_image[crop_y:crop_y+self.opt.crop_size, crop_x:crop_x+self.opt.crop_size]
#        label = label.crop((crop_x, crop_y, crop_x + self.opt.crop_size, crop_y + self.opt.crop_size))
#        # flip
#        if not (self.opt.phase == "test" or self.opt.no_flip or self.for_metrics):
#            if random.random() < 0.5:
#                coord_image = torch.fliplr(coord_image)
#                label = tvf.hflip(label) # left right

        # to tensor
#        label = tvf.to_tensor(label)

        return label

