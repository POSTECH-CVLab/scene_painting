import argparse
import array
import glob
import os

import Imath
import OpenEXR
import numpy as np
import cv2
import open3d as o3d
import imageio

from utils.utils import exr2numpy
    

def get_coordinate_image(image_path, depth_path, camera_path, intrinsic_path, use_ret_image=False):

    image = cv2.imread(image_path)
    depth_data = exr2numpy(depth_path, maxvalue=15, normalize=False)
    camera_data = np.load(camera_path)
    rot = camera_data['rotation']
    translation = camera_data['translation']
    K = np.load(intrinsic_path)

    height, width = image.shape[:2]

    fx = K['fx']
    #fy = K['fy']
    fy = fx
    cx = K['cx']
    cy = K['cy']

    depth_data = depth_data.astype(np.float32)
    rot = rot.astype(np.float32)
    translation = translation.astype(np.float32)
    fx = fx.astype(np.float32)
    fy = fy.astype(np.float32)
    cx = cx.astype(np.float32)
    cy = cy.astype(np.float32)
    
    xs, ys= np.meshgrid(np.arange(width), np.arange(height))
    xy_points = np.stack([xs, ys], axis=2).astype(np.float32) # (y, x)
    xy_points = (depth_data[:,:,None] * (xy_points - np.array([cx, cy]))) / np.array([fx, fy])
    points = np.concatenate([xy_points, depth_data[:,:,None]], axis=-1)
    points = points.reshape(height*width, 3)
    points = rot.T @ (points - translation).T
    points = points.T

    if use_ret_image:
        points = points.reshape(height, width, 3)

    return points

def get_path_lists(data_root):
    depth_path_list = sorted(glob.glob(os.path.join(data_root, 'depths', 'Depth*')))
    image_path_list = sorted(glob.glob(os.path.join(data_root, 'labels', 'Segmentation*')))
    camera_path_list = sorted(glob.glob(os.path.join(data_root, 'cameras', 'Camera_*.npz')))
    intr_path_list = sorted(glob.glob(os.path.join(data_root, 'cameras', 'intrinsic_*.npz')))
    return depth_path_list, image_path_list, camera_path_list, intr_path_list

def test(depth_path_list, image_path_list, camera_path_list, intr_path_list):
    index_list = [1, 2, 3, 4, 5, 6, 7, 8]
    color_list = [[1, 0, 0], [1, 0.7, 0], [0.7, 1, 0], [0, 1, 0], [1, 0, 0.7], [0.7, 0, 1], [0, 0, 1], [0.7, 0.7, 0.7], [0.4, 0.7, 0.4], [0.4, 0.4, 0.7]]
    pcd_list = []
    for i, idx in enumerate(index_list):
        ipath = image_path_list[idx]
        dpath = depth_path_list[idx]
        cpath = camera_path_list[idx]
        int_path = intr_path_list[idx]
        coordinate_points = get_coordinate_image(ipath, dpath, cpath, int_path)
        points1 = coordinate_points
        
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        colors = []
        for cidx in range(points1.shape[0]):
            #colors.append([0.01*idx,0,0])
            colors.append(color_list[i])
        pcd1.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd1)

    zero_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    zero_sphere.paint_uniform_color([0,1,0])
    o3d.visualization.draw_geometries(pcd_list + [zero_sphere])

    pcd1 = pcd_list[0]
    pcd2 = pcd_list[1]
    points1 = np.array(pcd1.points).astype(np.float32)
    points2 = np.array(pcd2.points).astype(np.float32)

    res = 512
    points1 = np.round(points1 * res, 0) / res
    points2 = np.round(points2 * res, 0) / res 

    point_dict = {}
    dup_count = 0
    for point in points1:
        point_key = (point[0], point[1], point[2])
        if point_key in point_dict:
            point_dict[point_key] += 1
            dup_count += 1
        else:
            point_dict[point_key] = 0
    for point in points2:
        point_key = (point[0], point[1], point[2])
        if point_key in point_dict:
            point_dict[point_key] += 1
            dup_count += 1
        else:
            point_dict[point_key] = 0
    print('dup_count:', dup_count)

def calculate_and_save_statistics(stats_path, depth_path_list, image_path_list, camera_path_list, intr_path_list):
    total_point_list = []
    total_mean_list = []
    for ipath, dpath, cpath, int_path in zip(image_path_list, depth_path_list, camera_path_list, intr_path_list):
        coordinate_points = get_coordinate_image(ipath, dpath, cpath, int_path)
        total_point_list.append(coordinate_points)
        total_mean_list.append(np.mean(coordinate_points, axis=0))
    total_mean_list = np.array(total_mean_list)
    total_mean = np.mean(total_mean_list, axis=0)

    max_value = -np.inf 
    for points in total_point_list:
        cand = np.abs(points - total_mean).max()
        if max_value < cand:
            max_value = np.abs(points).max()
    total_max_value = max_value + 1e-3

    np.savez(stats_path, mean=total_mean, max_value=total_max_value)

    return total_mean, total_max_value

def save_coordinate_images(depth_path_list, image_path_list, camera_path_list, intr_path_list, total_mean, total_max_value, coord_dir):

    for ipath, dpath, cpath, int_path in zip(image_path_list, depth_path_list, camera_path_list, intr_path_list):
        print(ipath)
        print(dpath)
        print(cpath)
        coordinate_image_points = get_coordinate_image(ipath, dpath, cpath, int_path, use_ret_image=True)
        height, width = coordinate_image_points.shape[:2]
        coordinate_image_points = coordinate_image_points.reshape(height*width, 3)

        # Normalize
        coordinate_image_points = (coordinate_image_points - total_mean) / total_max_value

        coordinate_image = coordinate_image_points.reshape(height, width, 3)
        coordinate_iamge = np.ascontiguousarray(coordinate_image)
        coordinate_image = coordinate_image.astype('float32')
        
        save_name = os.path.basename(ipath)
        save_path = os.path.join(coord_dir, save_name + '.exr')
        imageio.imwrite(save_path, coordinate_image)
        
        print(f'Saving {save_path} ...')
        print('\n')

def make_coord_image_and_stats(data_root, save_dir, visualize_test, is_testset, stats_path):
    depth_path_list, image_path_list, camera_path_list, intr_path_list = get_path_lists(data_root)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if visualize_test:
        test(depth_path_list, image_path_list, camera_path_list, intr_path_list)
        exit()

    if is_testset:
        assert stats_path is not None
        stats = np.load(stats_path)
        total_mean = stats['mean']
        total_max_value = stats['max_value']
    else:
        assert stats_path is None
        stats_path = os.path.join(save_dir, 'stats.npz')
        total_mean, total_max_value = calculate_and_save_statistics(stats_path, depth_path_list, image_path_list, camera_path_list, intr_path_list)

    coord_dir = os.path.join(save_dir, 'coordinate_images')
    os.makedirs(coord_dir, exist_ok=True)
    save_coordinate_images(depth_path_list, image_path_list, camera_path_list, intr_path_list, total_mean, total_max_value, coord_dir)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--testset', action='store_true')
    parser.add_argument('--stats_path', type=str)
    config = parser.parse_args()

    data_root = config.data_root
    save_dir = config.save_dir
    visualize_test = config.visualize
    is_testset = config.testset
    stats_path = config.stats_path

    make_coord_image_and_stats(data_root, save_dir, visualize_test, is_testset, stats_path)


