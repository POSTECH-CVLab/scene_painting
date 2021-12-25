import argparse
import glob
import os

import numpy as np
import cv2
import pickle


Dontcare_label = 0 

def get_ade_to_scenenet_dict(label_dir, target_dir, ade_to_scenenet_path):
    ade_to_scenenet = {}
    scenenet_set = set()
    scenenet_label_list = sorted(glob.glob(label_dir + '/Segmentation*.png'))
    for label_idx, label_path in enumerate(scenenet_label_list):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        scenenet_set.update(np.unique(label).tolist())
        if 0 in np.unique(label).tolist():
            print(label_idx)
    scenenet_object_label_list = sorted(list(scenenet_set))

    if 0 in scenenet_object_label_list:
        print('Error: 0 in labels')
        import pdb; pdb.set_trace()

    for idx, object_label in enumerate(scenenet_object_label_list):
        ade_to_scenenet[object_label] = idx + 1 

    print('label_nc:', len(ade_to_scenenet))

    with open(ade_to_scenenet_path, 'wb') as fw:
        pickle.dump(ade_to_scenenet, fw)

    return ade_to_scenenet

def convert_label(ade_to_scenenet, label_dir, target_dir):
    ade_used_labels = [k for k, _ in ade_to_scenenet.items()]
    
    label_list = sorted(glob.glob(label_dir + '/Segmentation*.png'))

    for label_idx, label_path in enumerate(label_list):
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    
        mask_list = []
        for ade_label, blender_label in ade_to_scenenet.items():
            cur_mask = (label == ade_label)
            mask_list.append(cur_mask)
        
        for idx, (ade_label, blender_label) in enumerate(ade_to_scenenet.items()):
            cur_mask = mask_list[idx]
            label[cur_mask] = blender_label

        total_used_mask = np.zeros_like(label).astype(np.bool)
        for cur_mask in mask_list:
            total_used_mask = np.logical_or(total_used_mask, cur_mask)
        not_used_mask = np.logical_not(total_used_mask)

        label[not_used_mask] = Dontcare_label

        save_path = os.path.join(target_dir, os.path.basename(label_path))
        cv2.imwrite(save_path, label)

def convert_ade_to_scenenet_label(label_dir, target_dir, is_testset, ade_to_scenenet_path):
    os.makedirs(target_dir, exist_ok=True)

    if is_testset:
        with open(ade_to_scenenet_path, 'rb') as fr:
            ade_to_scenenet = pickle.load(fr)
    else:
        ade_to_scenenet = get_ade_to_scenenet_dict(label_dir, target_dir, ade_to_scenenet_path)

    convert_label(ade_to_scenenet, label_dir, target_dir)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir")
    parser.add_argument("--target_dir")
    parser.add_argument('--is_testset', action='store_true')
    parser.add_argument("--ade_to_scenenet_path", type=str)
    config = parser.parse_args()
    
    label_dir = config.label_dir
    target_dir = config.target_dir
    is_testset = config.is_testset
    ade_to_scenenet_path = config.ade_to_scenenet_path

    convert_ade_to_scenenet_label(label_dir, target_dir, is_testset, ade_to_scenenet_path)

    
