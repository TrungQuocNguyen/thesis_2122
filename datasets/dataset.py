import os
import numpy as np 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf
import random 
class ScanNet2D(Dataset): 
    def __init__(self, img_dir, label_dir, img_size, is_transform = True, augmentation = True): 
        self.img_dir = img_dir
        self.label_dir =label_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.label_list = sorted(os.listdir(label_dir))
        self.img_size = img_size
        self.is_transform = is_transform
        self.augmentation = augmentation
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
        target = np.array(target, dtype = np.int64)
        target = torch.from_numpy(target)

        return img, target
        

