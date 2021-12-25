import os
import time

import torch
import numpy as np


def get_dataset_name(mode):
    if mode == "ade20k":
        return "Ade20kDataset"
    if mode == "cityscapes":
        return "CityscapesDataset"
    if mode == "coco":
        return "CocoStuffDataset"
    if mode == "blender":
        return "BlenderDataset"
    if mode == "unsup_blender":
        return "UnsupBlenderDataset"
    if mode == "style_blender":
        return "StyleBlenderDataset"
    if mode == "style_blender_with_ade":
        return "StyleBlenderWithADEDataset"
    else:
        ValueError("There is no such dataset regime as %s" % mode)


def worker_init_fn(worker_id):
    np.random.seed((np.random.get_state()[1][0] + worker_id + time.time_ns()) % 2 ** 32)


def get_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders." + dataset_name)
    dataset_train = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    dataset_val = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=True)
    print("Created %s, size train: %d, size val: %d" % (dataset_name, len(dataset_train), len(dataset_val)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                   num_workers=opt.num_workers, shuffle=True, drop_last=True,
                                                   worker_init_fn=worker_init_fn)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.val_batch_size, num_workers=8,
                                                 shuffle=False, drop_last=False)
    
    if opt.phase == 'test':
        dataloader_test = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size,
                                                      num_workers=opt.num_workers, shuffle=False, drop_last=False,
                                                      worker_init_fn=worker_init_fn)

        return None, dataloader_test

    return dataloader_train, dataloader_val


def get_test_dataloaders(opt):
    dataset_name = get_dataset_name(opt.dataset_mode)

    file = __import__("dataloaders." + dataset_name)
    dataset_test = file.__dict__[dataset_name].__dict__[dataset_name](opt, for_metrics=False)
    print("Created %s, size test: %d" % (dataset_name, len(dataset_test)))

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size,
                                                  num_workers=opt.num_workers, shuffle=False, drop_last=False,
                                                  worker_init_fn=worker_init_fn)
    return dataloader_test
