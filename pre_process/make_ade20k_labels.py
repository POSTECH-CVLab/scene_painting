import argparse
import glob
import os
import shutil

def make_ade_labels(src_dir, tgt_dir):
    os.makedirs(tgt_dir, exist_ok=True)

    src_list = sorted(glob.glob(os.path.join(src_dir, 'Segmentation*')))
    for path in src_list:
        filename = os.path.basename(path)
        tgt_path = os.path.join(tgt_dir, filename)
        shutil.copy(path, tgt_path) 


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir")
    parser.add_argument("--tgt_dir")
    opt = parser.parse_args()

    src_dir = opt.src_dir
    tgt_dir = opt.tgt_dir

    make_ade_labels(src_dir, tgt_dir)
    
