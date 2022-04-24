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
import h5py
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
    overfit takes 3 options: "1", "10" or None 
    """
    def __init__(self, cfg, split, overfit = None):
        super(Dataset, self).__init__()
        self.cfg = cfg
        if overfit != None:
            filename = split + '_overfit_' + overfit + '_chunks'
        else: 
            if cfg['augmented'] and split == 'train': 
                filename = 'augmented_' + split + '_clean'
            else: 
                filename = split + '_clean'
        self.file = h5py.File(os.path.join(cfg['root'], 'data_chunks', filename  + '.hdf5'), 'r')
    def __len__(self):
        return len(self.file['frames'])

    def __getitem__(self, idx):
        frames = self.file['frames'][idx] # 
        data = self.file['x'][idx] # np array of [32, 32, 64]
        label_grid = self.file['y'][idx] if self.cfg['return_label'] else None # np array of [32, 32, 64]
        scene_id = self.file['scene_id'][idx]
        scan_id = self.file['scan_id'][idx]
        world2grid = self.file['world_to_grid'][idx] # np array [4,4], float 32

        depths = []
        images = []
        poses = []
        frameids = []
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

            
        nearest_images = {'depths': depths, 'images': images, 'poses': poses, 'world2grid': world2grid, 'frameids': frameids}
        # dict return
        dict_return = {
            'data': data,  # np float array [32, 32, 64]
            'label': label_grid, # np float array of [32, 32, 64]
            'nearest_images': nearest_images, # dict of {'depths': # list of 5 np array [256, 328], 'images':  # list of 5 torch tensor size [3, 256, 328], value approximately in [-1.7,1.8], 'poses':# list of  5 np array [4,4], 'world2grid': np array 4x4, 'frameids':  list of 5 image id}
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
            image = transforms.Normalize(mean=self.cfg["color_mean"], std=self.cfg["color_std"])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
            image = np.expand_dims(image, 0)
        else:
            raise
        return image