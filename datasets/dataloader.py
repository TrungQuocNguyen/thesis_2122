import numpy as np
import torch
from torch.utils.data import DataLoader


class DataCollator(object):
    def __init__(self, cfg, batch ):
        """
        :param batch: list of dicts
        :return: dict
        """
        nearest_images = {}
        depths = []
        poses = []
        world2grid = []
        for b in batch:
            x = b['nearest_images']
            num_images = len(x['depths']) # 5
            max_num_images = cfg["NUM_IMAGES"]
            if max_num_images < num_images and cfg["MODE"] == 'train':
                num_images = max_num_images
                x['images'] = x['images'][:num_images]
                x['depths'] = x['depths'][:num_images]
                x['poses'] = x['poses'][:num_images]
            depths.append(torch.from_numpy(np.array(x['depths']))) # list of torch tensor size [max_num_images, 256, 328]
            poses.append(torch.from_numpy(np.array(x['poses']))) #list of torch tensor size [max_num_images, 4,4]
            world2grid.append(torch.from_numpy(x['world2grid']).expand(num_images, 4, 4)) #list of torch tensor size [max_num_images, 4,4](expanded from 1 to max_num_images)

        nearest_images = {
            'images': [torch.from_numpy(np.stack(x['nearest_images']['images'], 0).astype(np.float32)) for x in batch], # list of tensor, each tensor size [max_num_images, 3, 256, 328]
            'depths': depths, 'poses': poses, 'world2grid': world2grid
        }
        self.data = {
            'id': [x['id'] for x in batch], # list of strings, each string is path to one .chunk file
            'data': torch.stack([torch.from_numpy(x['data']) for x in batch], 0), # tensor [batch_size, 2,96,48,96]
            'nearest_images': nearest_images,
            'image_files': batch[0]['image_files']
        }
    def pin_memory(self):
        self.data['data']  = self.data['data'].pin_memory()
        self.data['nearest_images']['images'] = [x.pin_memory() for x in self.data['nearest_images']['images']]
        self.data['nearest_images']['depths'] = [x.pin_memory() for x in self.data['nearest_images']['depths']]
        self.data['nearest_images']['poses'] = [x.pin_memory() for x in self.data['nearest_images']['poses']]
        self.data['nearest_images']['world2grid'] = [x.clone().pin_memory() for x in self.data['nearest_images']['world2grid']]

        return self


def get_dataloader(cfg, dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory = True):
    def collate_wrapper(batch): 
        return DataCollator(cfg, batch)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_wrapper,
                             shuffle=shuffle, num_workers=num_workers, pin_memory= pin_memory)
    return data_loader