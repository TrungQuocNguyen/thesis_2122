import argparse
import json
from multiprocessing.sharedctypes import Value 
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from models import SurfaceNet, ResNeXtUNet
from models import DeepLabv3
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction_v2
from utils.helpers import print_params, make_intrinsic, adjust_intrinsic
#from projection import ProjectionHelper
from datasets.scannet.utils_3d import ProjectionHelper
from metric.iou import IoU

seed_value= 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `pytorch` pseudo-random generator at a fixed value
torch.manual_seed(seed_value)

def train(cfg): 
    print('Training network for 3D reconstruction...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    print(device)

    dataset_train = ScanNet2D3D(cfg, split = 'train', overfit = cfg["overfit"])
    dataset_val = ScanNet2D3D(cfg, split = 'val_scenes_non_overlapping', overfit = cfg["overfit"])
    num_images = len(dataset_train[0]["nearest_images"]["depths"]) # number of surrounding images of a chunk 
    if cfg["num_images"] < num_images: 
        num_images = cfg["num_images"]

    dataloader_train = get_dataloader(cfg, dataset_train, batch_size= cfg["batch_size"], shuffle = cfg["shuffle_train"], num_workers=cfg["num_workers"], pin_memory= cfg["pin_memory"])
    dataloader_val = get_dataloader(cfg, dataset_val, batch_size= cfg["batch_size"], shuffle= cfg["shuffle_val"], num_workers=cfg["num_workers"], pin_memory= cfg["pin_memory"])

    intrinsic = make_intrinsic(cfg["fx"], cfg["fy"], cfg["mx"], cfg["my"])
    intrinsic = adjust_intrinsic(intrinsic, [cfg["intrinsic_image_width"], cfg["intrinsic_image_height"]], cfg["depth_shape"])

    projector = ProjectionHelper(intrinsic, cfg["proj_depth_min"], cfg["proj_depth_max"], cfg["depth_shape"], cfg["subvol_size"], cfg["voxel_size"]).to(device)
    projector.update_intrinsic(intrinsic)
    
    #model_3d  = ResNeXtUNet(cfg, num_images)
    model_3d  = SurfaceNet(cfg, num_images)
    print_params(model_3d)
    model_3d.to(device)

    optimizer = optim.AdamW(model_3d.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])

    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 13.0], device = 'cuda'), ignore_index = -100)
    if cfg["use_2d_feat_input"]: 
        print('Using DeepLabv3')
        model_2d = DeepLabv3(cfg["model_2d"]["num_classes"])
        model_2d.load_state_dict(torch.load(cfg["model_2d"]["load_path_2d"])["state_dict"])
        for param in model_2d.parameters():
            param.requires_grad = False
        model_2d.to(device)
        model_2d.eval()
    else: 
        raise ValueError('Use_2d_feat_input must be set to true.')

    metric_3d = IoU(num_classes=3, ignore_index=2) # ground truth of 3D grid has 3 values:0, 1, -100. Converting label -100 to 2 we have 3 values: 0,1,2

    trainer = Trainer3DReconstruction_v2(cfg, model_3d, criterion, dataloader_train, dataloader_val, projector, optimizer, device, metric_3d, model_2d = model_2d)
    trainer.train()

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Training network for 3D reconstruction task')
    parser.add_argument('-c', '--config', default='experiments/cfgs/pretrained_feat_input_3d_recon.json',type=str,
                        help='Path to the config file (default: pretrained_feat_input_3d_recon.json)')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.mode == 'train': 
        cfg["mode"] = 'train'
        train(cfg)