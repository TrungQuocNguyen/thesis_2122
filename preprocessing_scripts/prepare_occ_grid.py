'''
Input: ScanNet gt and label PLY files
Output: GT Binary occupancy voxel grid and label grid
'''
import sys
sys.path.append('.')
import argparse
import torch

import numpy as np
from scipy.spatial.distance import cdist

from tqdm import tqdm

import os, os.path as osp
from pathlib import Path

import trimesh
from datasets.scannet.common import load_ply

def get_label_grid(input_grid, gt_vertices, gt_vtx_labels,voxel_size=None, method='nearest', dist_thres=0.2):
    '''
    input_grid:  the input trimesh.VoxelGrid (l, h, b)
    gt_vertices: (n, 3) vertices of the GT mesh 
    gt_vtx_labels: (n, ) labels of these vertices

    return: (l, h, b) array of labels for each grid cell
    '''
    centers = input_grid.points
    indices = input_grid.points_to_indices(centers)
    pairs = list(zip(centers, indices))
    label_grid = -np.ones_like(input_grid.matrix, dtype=np.int16)

    for center, ndx in tqdm(pairs, leave=False, desc='nearest_point'):
        if method == 'nearest':
            # distance from this voxel center to all vertices
            dist = cdist(np.expand_dims(center, 0), gt_vertices).flatten()
            # closest vertex
            closest_vtx_ndx = dist.argmin()
            # label of this vertex
            if dist.min() < dist_thres: 
                voxel_label = gt_vtx_labels[closest_vtx_ndx]
            else: 
                voxel_label = 0
        elif method == 'voting':
            # find indices all vertices within this voxel
            low, high = center - voxel_size, center + voxel_size
            vtx_in_voxel = np.all(np.logical_and((gt_vertices >= low), (gt_vertices <= high)), axis=1)
            # labels of these vertices
            labels = gt_vtx_labels[vtx_in_voxel]
            # most common label
            try:
                voxel_label = np.bincount(labels).argmax()
            except ValueError:
                voxel_label = None
        
        label_grid[ndx[0], ndx[1], ndx[2]] = voxel_label


    return label_grid

def main(args):
    root = Path(args.scannet_dir)
    output = Path(args.output_dir)
    voxel_size = args.voxel_size
    print(f'Using voxel size: {voxel_size}')
    print(f'Read labels?: {not args.no_label}')
    for scan_id in tqdm(sorted(os.listdir(root)), desc='scan'):
        scan_dir = root / scan_id

        input_file = f'{scan_id}_vh_clean_2.ply' 
        gt_file = f'{scan_id}_vh_clean_2.labels.ply' 

        # read input mesh and voxelize
        input_mesh = trimesh.load(scan_dir / input_file)
        input_grid = input_mesh.voxelized(pitch=voxel_size) 
        
        # read GT mesh, get vertex coordinates and labels
        coords, _, _ = load_ply(scan_dir / input_file)

        if args.no_label:
            # no labels, zeros
            label_grid = np.zeros_like(input_grid.matrix, dtype=np.int16)
        else:
            # read coords and labels from GT file
            _, _, labels = load_ply(scan_dir / gt_file, read_label=True)
            # get label grid
            label_grid = get_label_grid(input_grid, coords, labels)
        
        x, y = input_grid.matrix, label_grid
        out_file = f'{scan_id}_occ_grid.pth'

        data = {'x': x, 'y': y, 'translation': input_grid.translation, 
                'start_ndx': input_grid.translation / voxel_size}
        save_path = output / scan_id
        save_path.mkdir(parents = True, exist_ok = True)
        torch.save(data, save_path / out_file)

if __name__ == '__main__':
    # params
    parser = argparse.ArgumentParser()
    # data paths
    parser.add_argument('scannet_dir', help='path to scannet root dir file to read')
    parser.add_argument('output_dir', help = 'path to output file ')
    parser.add_argument('--voxel-size', type=float, dest='voxel_size', default=0.05)
    parser.add_argument('--no-label', action='store_true', default=False, dest='no_label', 
                        help='No labels (test set)')
    

    args = parser.parse_args()

    main(args)