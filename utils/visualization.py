"""
    Simple Usage example (with 3 images)
"""
import os
import math 
import numpy as np
import argparse
import pickle
import sys
sys.path.append('.')
from plyfile import PlyData

# color palette for nyu40 labels
def create_color_palette():
    return [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]


def read_ply(ply_file):
    with open(ply_file, 'rb') as read_file:
        ply_data = PlyData.read(read_file)

    points = []
    colors = []
    indices = []
    for x,y,z,r,g,b,a in ply_data['vertex']:
        points.append([x,y,z])
        colors.append([r,g,b])
    for face in ply_data['face']:
        indices.append([face[0][0], face[0][1], face[0][2]])
    points = np.array(points)
    colors = np.array(colors)
    indices = np.array(indices)
    return points, indices, colors

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_mask_pointcloud(mask, output_file):
    """
    mask: numpy array (x,y,z), in which instance/label id
    output_file: string
    """
    def make_voxel_mesh(box_min, box_max, color): 
        vertices = [
            np.array([box_max[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_min[1], box_min[2]])
        ]
        return vertices

    scale = 1
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for z in range(mask.shape[2]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]):
                if mask[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) - 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts = make_voxel_mesh(box_min, box_max, np.array(create_color_palette()[int(mask[x,y,z]%41)])/255.0)
                    verts.extend(box_verts)
    write_ply(verts, None, None, output_file)

def write_mask(mask, output_file):
    """
    mask: numpy array (x,y,z), in which instance/label id
    output_file: string
    """
    def make_voxel_mesh(box_min, box_max, color): 
        vertices = [
            np.array([box_max[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_max[1], box_max[2]]),
            np.array([box_min[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_min[1], box_max[2]]),
            np.array([box_max[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_max[1], box_min[2]]),
            np.array([box_min[0], box_min[1], box_min[2]]),
            np.array([box_max[0], box_min[1], box_min[2]])
        ]

        colors = [
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]]),
            np.array([color[0], color[1], color[2]])
        ]
        indices = [
            np.array([1, 2, 3], dtype=np.uint32),
            np.array([1, 3, 0], dtype=np.uint32),
            np.array([0, 3, 7], dtype=np.uint32),
            np.array([0, 7, 4], dtype=np.uint32),
            np.array([3, 2, 6], dtype=np.uint32),
            np.array([3, 6, 7], dtype=np.uint32),
            np.array([1, 6, 2], dtype=np.uint32),
            np.array([1, 5, 6], dtype=np.uint32),
            np.array([0, 5, 1], dtype=np.uint32),
            np.array([0, 4, 5], dtype=np.uint32),
            np.array([6, 5, 4], dtype=np.uint32),
            np.array([6, 4, 7], dtype=np.uint32)
        ]
        return vertices, colors, indices

    scale = 1
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []
    for z in range(mask.shape[2]):
        for y in range(mask.shape[1]):
            for x in range(mask.shape[0]):
                if mask[x, y, z] > 0:
                    box_min = (np.array([x, y, z]) - 0.05)*scale + offset
                    box_max = (np.array([x, y, z]) + 0.95)*scale + offset
                    box_verts, box_color, box_ind = make_voxel_mesh(box_min, box_max, np.array(create_color_palette()[int(mask[x,y,z]%41)])/255.0)
                    cur_num_verts = len(verts)
                    box_ind = [x + cur_num_verts for x in box_ind]
                    verts.extend(box_verts)
                    indices.extend(box_ind)
                    colors.extend(box_color)
    write_ply(verts, colors, indices, output_file)

def parse_args():
    """parse input arguments"""
    parser = argparse.ArgumentParser(description='3D-SIS')
    parser.add_argument('--path', type=str, default='../results/')
    parser.add_argument('--mode', type=str, default='npy')

    args = parser.parse_args()
    return args

