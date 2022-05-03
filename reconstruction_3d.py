import argparse
import json 
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from models import Dense3DNetwork, SurfaceNet, ResUNet, ResNeXtUNet, ConvNeXtUNet
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction
from utils.helpers import print_params, make_intrinsic, adjust_intrinsic, init_weights
#from projection import ProjectionHelper
from datasets.scannet.utils_3d import ProjectionHelper

#seed_value= 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
#os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
#random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
#np.random.seed(seed_value)
# 4. Set `pytorch` pseudo-random generator at a fixed value
#torch.manual_seed(seed_value)

class FixedCrossEntropyLoss(CrossEntropyLoss):
    """
    Standard CrossEntropyLoss with label_smoothing doesn't handle ignore_index properly, so we apply
    the mask ourselves. See https://github.com/pytorch/pytorch/issues/73205
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if not target.is_floating_point() and self.ignore_index is not None:
            input = input.permute(0,2,3,4,1)[target!=self.ignore_index]
            target = target[target != self.ignore_index]
        loss = super().forward(input, target)
        return loss
def train(cfg): 
    print('Training network for 3D reconstruction...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    print(device)

    dataset_train = ScanNet2D3D(cfg, split = 'train', overfit = cfg["overfit"])
    dataset_val = ScanNet2D3D(cfg, split = 'val', overfit = cfg["overfit"])
    num_images = len(dataset_train[0]["nearest_images"]["depths"]) # number of surrounding images of a chunk 
    if cfg["num_images"] < num_images: 
        num_images = cfg["num_images"]

    dataloader_train = get_dataloader(cfg, dataset_train, batch_size= cfg["batch_size"], shuffle = cfg["shuffle_train"], num_workers=cfg["num_workers"], pin_memory= cfg["pin_memory"])
    dataloader_val = get_dataloader(cfg, dataset_val, batch_size= cfg["batch_size"], shuffle= cfg["shuffle_val"], num_workers=cfg["num_workers"], pin_memory= cfg["pin_memory"])

    intrinsic = make_intrinsic(cfg["fx"], cfg["fy"], cfg["mx"], cfg["my"])
    intrinsic = adjust_intrinsic(intrinsic, [cfg["intrinsic_image_width"], cfg["intrinsic_image_height"]], cfg["depth_shape"])

    projector = ProjectionHelper(intrinsic, cfg["proj_depth_min"], cfg["proj_depth_max"], cfg["depth_shape"], cfg["subvol_size"], cfg["voxel_size"]).to(device)
    projector.update_intrinsic(intrinsic)
    
    model = ConvNeXtUNet(cfg, num_images)
    #model  = ResNeXtUNet(cfg, num_images)
    #model  = SurfaceNet(cfg, num_images)
    #model  = Dense3DNetwork(cfg, num_images)
    #model.apply(init_weights)
    print_params(model)
    model.to(device)

    #loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([44.5], device = 'cuda'))
    loss = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 13.0], device = 'cuda'), ignore_index = -100)
    #loss = FixedCrossEntropyLoss(weight = torch.tensor([1.0, 13.0], device = 'cuda'), ignore_index = -100, label_smoothing= 0.1)

    optimizer = optim.AdamW(model.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    #optimizer = optim.SGD(model.parameters(), lr  = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"], momentum = 0, nesterov= False)
    
    

    trainer = Trainer3DReconstruction(cfg, model, loss, dataloader_train, dataloader_val, projector, optimizer, device)
    trainer.train()
    
def test(cfg): 
    pass
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Training network for 3D reconstruction task')
    parser.add_argument('-c', '--config', default='experiments/cfgs/overfit_3d_reconstruction.json',type=str,
                        help='Path to the config file (default: train_3d_reconstruction.json)')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.mode == 'train': 
        cfg["mode"] = 'train'
        train(cfg)
    
    