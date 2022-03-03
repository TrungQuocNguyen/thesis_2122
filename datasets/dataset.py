import os
import math
import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf
import random 
from .BinaryReader import BinaryReader
class ScanNet2D(Dataset): 
    def __init__(self, cfg): 
        self.cfg = cfg
        self.img_dir = cfg["img_dir"]
        self.label_dir =cfg["label_dir"]
        self.img_list = sorted(os.listdir(self.img_dir))
        self.label_list = sorted(os.listdir(self.label_dir))
        self.img_size = cfg["img_size"]
        self.is_transform = cfg["is_transform"]
        self.augmentation = cfg["augmentation"]
        self.normalize = cfg["normalize"]
        self.mean = cfg["mean"]
        self.std = cfg["std"]
    def __len__(self): 
        return len(self.img_list)
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_name)

        label_name = os.path.join(self.label_dir, self.label_list[idx])
        target = Image.open(label_name)

        if self.is_transform: 
            img, target  = self.transform(img, target)
        return img, target
    def transform(self, img, target):
        #resize
        #random crop
        #to tensor
        #flipping

        resize_img = transforms.Resize(self.img_size, interpolation = InterpolationMode.BILINEAR)
        resize_target = transforms.Resize(self.img_size, interpolation = InterpolationMode.NEAREST)
        totensor = transforms.ToTensor()
        normalize = transforms.Normalize(self.mean, self.std)

        img = resize_img(img)
        target = resize_target(target)

        if self.augmentation: 
            if random.random() > 0.5: 
                img = tf.vflip(img)
                target = tf.vflip(target)

            if random.random() > 0.5: 
                img = tf.hflip(img)
                target = tf.hflip(target)
    

        img = totensor(img)
        if self.normalize: 
            img = normalize(img)

        target = np.array(target, dtype = np.int64)
        target = torch.from_numpy(target)

        return img, target
        
class ScanNet2D3D(Dataset):
    """
    SUNCG/ScanNet DATASET
    """
    def __init__(self, cfg, data_location, mode):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.mode = mode 

        if os.path.isfile(data_location):
            datalist = open(data_location, 'r')
            self.scenes = [x.strip() for x in datalist.readlines()]


    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):

        #---------------------------
        # read sdf
        #---------------------------
        reader = BinaryReader(self.scenes[idx])
        dimX, dimY, dimZ = reader.read('UINT64', 3) # 96, 48, 96
        data = reader.read('float', dimX * dimY * dimZ) # tuple
        data = np.expand_dims(np.reshape(data, (dimX, dimY, dimZ), order='F'), 0).astype(np.float32) # np float array of [1,96, 48, 96]

        if self.cfg["FLIP_TSDF"]:
            trunc_data = np.clip(data, -self.cfg["TRUNCATED"], self.cfg["TRUNCATED"])
            trunc_abs_data = np.abs(trunc_data)
            trunc_abs_data_flip = self.cfg["TRUNCATED"] - trunc_abs_data
            trunc_abs_data_flip[data < 0] *= -1
            data = trunc_abs_data_flip
            #data = np.concatenate([trunc_abs_data_flip, np.greater(data, -1)], 0)
        elif self.cfg["LOG_TSDF"]:
            trunc_data = np.clip(data, -self.cfg["TRUNCATED"], self.cfg["TRUNCATED"])
            trunc_abs_data = np.abs(trunc_data)
            trunc_abs_data_log = np.log(trunc_abs_data)
            trunc_abs_data_log[data < 0] *=-1
            data = trunc_abs_data_log
            #data = np.concatenate([trunc_abs_data_log, np.greater(data, -1)], 0)
        else:
            trunc_data = np.clip(data, -self.cfg["TRUNCATED"], self.cfg["TRUNCATED"])
            mask = abs(trunc_data)< 1
            data = np.zeros((1, dimX, dimY, dimZ)).astype(np.float32) # (1, 96, 48, 96)
            data[mask] = 1.0
            #data = trunc_data
        #    trunc_abs_data = np.abs(trunc_data)
        #    data = np.concatenate([trunc_abs_data, np.greater(data, 0)], 0) # (2,96,48,96)
        #----------------------------
        # read redundant data (used for segmentation task, not for reconstruction)
        #----------------------------
        (num_box,) = reader.read('uint32')
        for i in range(num_box):
            _, _, _ , _, _, _ = reader.read('float', 6)
            _ = reader.read('uint32')
        (num_mask,) = reader.read('uint32')
        for i in range(num_mask):
                _ = reader.read('uint32')
                dimX, dimY, dimZ = reader.read('UINT64', 3) # for e.g(just one sample) (8,3,9)
                _ = reader.read('uint16', dimX * dimY * dimZ)
        (num_box,) = reader.read('uint32')
        for i in range(num_box):
                _ = reader.read('float')
        #----------------------------
        # read images
        #----------------------------
         # read image, depth image, pose, world2grid
        depths = []
        images = []
        poses = []
        frameids = []
        nearest_images = {}
        image_files = []
        world2grid = np.linalg.inv(np.transpose(np.reshape(reader.read('float', 16), (4, 4), order='F')).astype(np.float32)) # np array [4,4]
        (num_images,) = reader.read('uint32') # #5 RGB images that associated with this chunk

        if self.cfg["BASE_IMAGE_PATH"].endswith('square') or self.cfg["BASE_IMAGE_PATH"].endswith('square/'):
            scene_name = os.path.basename(self.scenes[idx]).split('__')[0] # e.g scene0528_00
        else:
            raise NotImplementedError

        if self.mode != 'chunk':
                num_images = os.listdir(os.path.join(self.cfg["BASE_IMAGE_PATH"], scene_name, 'depth'))
                # reload correct world2grid for scene
                world2grid = self.load_pose(os.path.join(self.cfg["BASE_IMAGE_PATH"], scene_name, 'world2grid.txt'))
                # padding substraction
                world2grid[0][3] = world2grid[0][3] - 10
                world2grid[1][3] = world2grid[1][3] - 16
                world2grid[2][3] = world2grid[2][3] - 10
        else:
            num_images = range(num_images)

        for i in num_images:
            if self.mode != 'chunk':
                frameid = i.split('.')[0]
            else:
                (frameid,) = reader.read('uint32')

            depth_file = os.path.join(self.cfg["BASE_IMAGE_PATH"], scene_name, 'depth', str(frameid) + '.png')
            image_file = os.path.join(self.cfg["BASE_IMAGE_PATH"], scene_name, self.cfg["IMAGE_TYPE"], str(frameid) + self.cfg["IMAGE_EXT"])
            pose_file = os.path.join(self.cfg["BASE_IMAGE_PATH"], scene_name, 'pose', str(frameid) + '.txt')
            poses.append(self.load_pose(pose_file)) # list of  5 np array [4,4], each array: camera to world pose
            depths.append(self.load_depth(depth_file, self.cfg["DEPTH_SHAPE"])) # list of 5 np array [256, 328]
            im_pre = self.load_image(image_file, self.cfg["IMAGE_SHAPE"]) # torch tensor size [3, 256, 328]

            images.append(im_pre) # list of 5 torch tensor size [3, 256, 328], value approximately in [-1.7,1.8]
            frameids.append(frameid) # list of 5 image id(id inside that specific scene e.g scene0528_00) corresponds to this chunk
            image_files.append(image_file) # list of 5 paths to images
            
        nearest_images = {'depths': depths, 'images': images, 'poses': poses, 'world2grid': world2grid, 'frameids': frameids}

        #---------------------------
        # crop max height
        #---------------------------
        if self.mode == 'benchmark':
            maxHeight = 480
        else:
            maxHeight = 48

        data = data[:,:,:maxHeight,:]

        # dict return
        dict_return = {
            'id': self.scenes[idx], # string: path to .chunk file of this chunk 
            'data': data,  # np float array [2,96, 48, 96]
            'nearest_images': nearest_images, # dict of {'depths': # list of 5 np array [32, 41], 'images':  # list of 5 torch tensor size [3, 256, 328], value approximately in [-1.7,1.8], 'poses':# list of  5 np array [4,4], 'world2grid': np array 4x4, 'frameids':  list of 5 image id}
            'image_files': image_files # list of 5 paths to images
        }
        reader.close()

        return dict_return


    def load_pose(self, filename):
        pose = np.zeros((4, 4))
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]
        return np.asarray(lines).astype(np.float32)

    def resize_crop_image(self, image, new_image_dims):
        #image: [240, 320]
        #new_image_dims: [328, 256]
        image_dims = [image.shape[1], image.shape[0]] # [320, 240]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=InterpolationMode.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image) # [256, 328]
        return image

    def load_depth(self, file, image_dims):
        #image_dims: [328, 256]
        depth_image = np.array(Image.open(file)) # [240, 320]
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims) # (256, 328)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image

    def load_image(self, file, image_dims):
        image = np.array(Image.open(file)) # [240, 320]
        # preprocess
        image = self.resize_crop_image(image, image_dims) # (256, 328,3)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=self.cfg["COLOR_MEAN"], std=self.cfg["COLOR_STD"])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image

