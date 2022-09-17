import os
import math
import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf
import random 
from .BinaryReader import BinaryReader
import h5py
class ScanNet2D(Dataset): 
    def __init__(self, cfg): 
        self.cfg = cfg
        self.img_dir = cfg["img_dir"]
        self.label_dir =cfg["label_dir"]
        self.img_list = sorted(os.listdir(self.img_dir))
        self.label_list = sorted(os.listdir(self.label_dir))
        self.img_size = cfg["img_size"]
        self.label_img_size = cfg["label_img_size"]
        self.augmentation = cfg["augmentation"]
        self.mean = cfg["mean"]
        self.std = cfg["std"]
        self.transform = T.Compose([T.ColorJitter(brightness= 0.3, contrast= 0.3, saturation= 0.3, hue= 0.3), 
                                    T.GaussianBlur(kernel_size=(3,7), sigma = (0.1, 4))])
    def __len__(self): 
        return len(self.img_list)
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        random_hflip = random.random()

        img_name = os.path.join(self.img_dir, self.img_list[idx])
        image = self.load_image(img_name, self.img_size, random_hflip)

        label_name = os.path.join(self.label_dir, self.label_list[idx])
        target = self.load_image(label_name, self.label_img_size, random_hflip)

        return image, target
    def resize_crop_image(self, image, new_image_dims):
        #image: [240, 320]
        #new_image_dims: [328, 256]
        image_dims = [image.shape[1], image.shape[0]] # [320, 240]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = T.Resize([new_image_dims[1], resize_width], interpolation=InterpolationMode.NEAREST)(Image.fromarray(image))
        image = T.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        return image # [256, 328,3] or [256, 328] if turn to np.array
    def load_image(self, file, image_dims, random_set):
        image = np.array(Image.open(file)) # [240, 320]
        # preprocess
        image = self.resize_crop_image(image, image_dims) # (256, 328,3)

        if self.augmentation: 

            if random_set > 0.5: 
                image = tf.hflip(image)
            if image.mode == 'RGB': 
                image = self.transform(image)

        image = np.array(image)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = T.Normalize(mean=self.mean, std=self.std)(torch.Tensor(image.astype(np.float32) / 255.0))
    
        elif len(image.shape) == 2: # label image
            image = torch.from_numpy(image.astype(np.int64))
        else:
            raise
        return image

class ScanNet2D3D(Dataset):
    """
    overfit takes 3 options: "1", "10" or None 
    """
    def __init__(self, cfg, split, overfit = None):
        super(Dataset, self).__init__()
        self.cfg = cfg
        self.split = split
        if overfit != None:
            filename = split + '_overfit_' + overfit + '_chunks'
        else: 
            if cfg['augmented_3d'] and split == 'train': 
                filename = 'augmented_' + split
            else: 
                filename = split
        self.file = h5py.File(os.path.join(cfg['root'], 'data_chunks_from_tsdf', filename  + '.hdf5'), 'r')
    def __len__(self):
        return len(self.file['frames'])

    def __getitem__(self, idx):
        frames = self.file['frames'][idx] # 
        data = self.file['x'][idx] # np array of [32, 32, 64]
        label_grid = self.file['y'][idx] if self.cfg['return_label_grid'] else None # np array of [32, 32, 64]
        scene_id = self.file['scene_id'][idx]
        scan_id = self.file['scan_id'][idx]
        world2grid = self.file['world_to_grid'][idx] # np array [4,4], float 32
        if self.cfg["augmented_2d"]: 
            self.augmentation_factor = random.uniform(0.7, 1.3)
            self.hue_factor = random.uniform(-0.3, 0.3)
            self.gaussian_sigma = random.uniform(0.1, 4)
        depths = []
        images = []
        poses = []
        frameids = []
        label_images = []
        nearest_images = {}

        scene_name = 'scene' + '{:04d}'.format(scene_id) + '_' + '{:02d}'.format(scan_id) # e.g scene0528_00

        for frameid in frames: 
            if frameid >=0: 
                depth_file = os.path.join(self.cfg['root'], scene_name, 'depth', str(frameid) + '.png')
                image_file = os.path.join(self.cfg['root'], scene_name, 'color', str(frameid) + '.jpg')
                pose_file = os.path.join(self.cfg['root'], scene_name, 'pose', str(frameid) + '.txt')
                poses.append(self.load_pose(pose_file)) 
                depths.append(self.load_depth(depth_file, self.cfg["depth_shape"]))
                im_pre = self.load_image(image_file, self.cfg["image_shape"])
                images.append(im_pre)
                frameids.append(frameid)
                if self.cfg['model_2d']['proxy_loss']: 
                    label_file = os.path.join(self.cfg['root'], scene_name, 'label', str(frameid) + '.png')
                    label_images.append(self.load_image(label_file, self.cfg["depth_shape"])) # we train 2D proxy loss with image size [328, 256], which is different from 3DMV, who train with image size [41, 32]. We may consider this option later on 

            
        nearest_images = {'depths': depths, 'images': images, 'poses': poses, 'world2grid': world2grid, 'frameids': frameids}
        if self.cfg['model_2d']['proxy_loss']: 
            nearest_images['label_images'] = label_images # list of 5 torch tensor size [image_shape[1], image_shape[0]]
        # dict return
        dict_return = {
            'data': data,  # np float array [32, 32, 64]
            'label': label_grid, # np float array of [32, 32, 64]
            'nearest_images': nearest_images, # dict of {'depths': # list of 5 np array [depth_shape[1], depth_shape[0]], 'images':  # list of 5 torch tensor size [3, image_shape[1], image_shape[0]], value approximately in [-1.7,1.8], 'poses':# list of  5 np array [4,4], 'world2grid': np array 4x4, 'frameids':  list of 5 image id}
            'scan_name': scene_name
        }

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
        image = T.Resize([new_image_dims[1], resize_width], interpolation=InterpolationMode.NEAREST)(Image.fromarray(image))
        image = T.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        return image

    def load_depth(self, file, image_dims):
        #image_dims: [328, 256]
        depth_image = np.array(Image.open(file)) # [240, 320]
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims) # (256, 328)
        depth_image = np.array(depth_image)
        depth_image = depth_image.astype(np.float32) / 1000.0
        return depth_image

    def load_image(self, file, image_dims):
        image = np.array(Image.open(file)) # [240, 320]
        # preprocess
        image = self.resize_crop_image(image, image_dims) # (256, 328,3)
        ########################## 2D image augmentation ###############################
        if self.cfg["augmented_2d"] and self.split == 'train': 
            if image.mode == 'RGB': 
                image = tf.adjust_hue(tf.adjust_saturation(tf.adjust_contrast(tf.adjust_brightness(image, self.augmentation_factor), self.augmentation_factor), self.augmentation_factor), self.hue_factor)
                image = tf.gaussian_blur(image, kernel_size= (3,7), sigma = self.gaussian_sigma)
        ################################################################################
        image = np.array(image) # [256, 328]
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = T.Normalize(mean=self.cfg["color_mean"], std=self.cfg["color_std"])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = torch.from_numpy(image.astype(np.int64))
        else:
            raise
        return image