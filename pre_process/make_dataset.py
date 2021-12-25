import argparse
import os
from subprocess import check_output 

from make_coordinate_image import make_coord_image_and_stats
from convert_ade20k_to_scenenet_label import convert_ade_to_scenenet_label
from make_ade20k_labels import make_ade_labels


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--is_testset', action='store_true')
    parser.add_argument('--stats_path', type=str)
    parser.add_argument('--ade_to_scenenet_path', type=str)
    opt = parser.parse_args()

    # Make coordinate images and stats.npz
    data_root = opt.data_root
    save_dir = data_root[:-9]
    visualize_test = False
    is_testset = opt.is_testset
    if is_testset:
        stats_path = opt.stats_path
        assert stats_path is not None
    else:
        stats_path = None
    make_coord_image_and_stats(data_root, save_dir, visualize_test, is_testset, stats_path)

    # Make scenenet labels and ade_to_scenenet dictionary
    label_dir = os.path.join(data_root, 'labels')
    target_dir = os.path.join(save_dir, 'labels_scenenet')
    if is_testset:
        ade_to_scenenet_path = opt.ade_to_scenenet_path
        assert ade_to_scenenet_path is not None
    else:
        ade_to_scenenet_path = os.path.join(save_dir, 'ade_to_scenenet.pickle')
    convert_ade_to_scenenet_label(label_dir, target_dir, is_testset, ade_to_scenenet_path)
    
    # Make ade20k labels
    src_dir = os.path.join(data_root, 'labels')
    tgt_dir = os.path.join(save_dir, 'labels_ade20k')
    make_ade_labels(src_dir, tgt_dir)


