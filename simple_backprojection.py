import argparse
import numpy as np
import os, math
from PIL import Image
import torchvision.transforms as transforms
def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', required=True, help='path to 2d train data') # "/mnt/raid/tnguyen/scannet_frames_25k"
parser.add_argument('--scan_name', required=True, help='scan to load')  # "scene0255_01"
parser.add_argument('--intrinsic_image_width', type=int, default=640, help='2d image width')
parser.add_argument('--intrinsic_image_height', type=int, default=480, help='2d image height')
parser.add_argument('--fx', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--fy', type=float, default=577.870605, help='intrinsics')
parser.add_argument('--mx', type=float, default=319.5, help='intrinsics')
parser.add_argument('--my', type=float, default=239.5, help='intrinsics')
proj_image_dims = [640, 480]
opt = parser.parse_args()
intrinsic = make_intrinsic(opt.fx, opt.fy, opt.mx, opt.my)
def main(): 
    base_dir = os.path.join(opt.data_path, opt.scan_name)
    for filename in os.listdir(os.path.join(base_dir, 'color')): 
        print('Processing file %s...'%(filename[:6]))
        depth_image, color_image, pose = load_frames_multi(base_dir, filename[:6], proj_image_dims)
        #depth_image: np array[480, 640]
        #color_image: np array[480, 640,3]
        #pose: np array [4,4], camera to world 
        coordinates = np.empty([proj_image_dims[1], proj_image_dims[0],4,1])  #[480, 640, 4,1]
        for ux in range(proj_image_dims[0]): 
            for uy in range(proj_image_dims[1]): 
                coordinates[uy, ux, :,:] = depth_to_skeleton(ux, uy, depth_image[uy, ux]).reshape(4,1)
      
        coordinates = np.matmul(pose, coordinates)  #[480, 640, 4,1]
        
        vertices = np.concatenate((coordinates.squeeze()[:,:,:3], color_image), axis = 2).reshape(-1, 6) # [480 *640,6]
        print('Writing to mesh file...')
        writeMesh(vertices, filename[:6])
        print('Done.')        

def writeMesh(vertices, ply_filename): 
    num_vertices = vertices.shape[0]
    ply_head = 'ply\n' \
               'format ascii 1.0\n' \
               'element vertex {}\n' \
               'property float x\n' \
               'property float y\n' \
               'property float z\n' \
               'property uchar red\n' \
               'property uchar green\n' \
               'property uchar blue\n' \
               'end_header'.format(num_vertices)
    np.savetxt(os.path.join('ply_folder', ply_filename + '.ply'), vertices, fmt="%.3f %.3f %.3f %d %d %d", header=ply_head, comments='')  

    return intrinsic
def load_frames_multi(base_dir, filename, proj_image_dims): 
    #base_dir: "/mnt/raid/tnguyen/scannet_frames_25k/scene0255_01"
    #filename: "000100.jpg"

    depth_file = os.path.join(base_dir, 'depth', filename + '.png')
    color_file = os.path.join(base_dir, 'color', filename + '.jpg')
    pose_file = os.path.join(base_dir, 'pose', filename + '.txt')

    depth_image_dims = proj_image_dims  #[640, 480]
    color_image_dims = depth_image_dims

    depth_image, color_image, pose = load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims)
    return depth_image, color_image, pose



def resize_crop_image(image, new_image_dims):
    #image: [480, 640]
    #new_image_dims: [640, 480]
    image_dims = [image.shape[1], image.shape[0]] # [640, 480]
    if image_dims == new_image_dims:
        return image
    resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
    image = transforms.Resize([new_image_dims[1], resize_width], interpolation=Image.NEAREST)(Image.fromarray(image))
    image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
    image = np.array(image)
    return image
def load_depth_label_pose(depth_file, color_file, pose_file, depth_image_dims, color_image_dims):
    #depth_image_dims: [640, 480]
    #color_image_dims: [640, 480]
    depth_image = np.array(Image.open(depth_file)) #  np array of [480, 640]
    color_image = np.array(Image.open(color_file)) # np array [968, 1296,3]
    pose = load_pose(pose_file)  #np array of [4,4]
    # preprocess
    depth_image = resize_crop_image(depth_image, depth_image_dims) #[480, 640]
    color_image = resize_crop_image(color_image, color_image_dims) # [480, 640,3]
    depth_image = depth_image.astype(np.float32)/1000.0# convert to mm to m ? (because value in depth frame > 10000, e.g 1853)
    return depth_image, color_image, pose
def load_pose(filename):
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
    return np.asarray(lines).astype(np.float32)
def depth_to_skeleton(ux, uy, depth):
    x = (ux - intrinsic[0][2]) / intrinsic[0][0]
    y = (uy - intrinsic[1][2]) / intrinsic[1][1]
    return np.array([depth*x, depth*y, depth, 1])  #return [X, Y, Z, 1]
if __name__ == '__main__': 
    main()