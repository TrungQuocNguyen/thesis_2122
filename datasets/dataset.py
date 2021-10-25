import os
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as tf
import random 
class ScanNet2D(Dataset): 
    def __init__(self, img_dir, label_dir, img_size, is_transform = True): 
        self.img_dir = img_dir
        self.label_dir =label_dir
        self.img_list = sorted(os.listdir(img_dir))
        self.label_list = sorted(os.listdir(label_dir))
        self.img_size = img_size
        self.is_transform = is_transform
    def __len__(self): 
        return len(self.img_list)
    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.img_list[idx])
        img = Image.open(img_name)

        label_name = os.path.join(self.label_dir, self.label_list[idx])
        mask = Image.open(label_name)

        if self.is_transform: 
            img, mask  = self.transform(img, mask)
        return img, mask
    def transform(self, img, mask):
        #resize
        #random crop
        #to tensor
        #flipping

        resize_img = transforms.Resize(self.img_size, interpolation = InterpolationMode.BILINEAR)
        resize_mask = transforms.Resize(self.img_size, interpolation = InterpolationMode.NEAREST)
        totensor = transforms.ToTensor()

        img = resize_img(img)
        mask = resize_mask(mask)

        if random.random() > 0.5: 
            img = tf.vflip(img)
            mask = tf.vflip(mask)

        if random.random() > 0.5: 
            img = tf.hflip(img)
            mask = tf.hflip(mask)
    

        img = totensor(img)
        mask = np.array(mask, dtype = np.int64)
        mask = torch.from_numpy(mask)

        return img, mask
        

