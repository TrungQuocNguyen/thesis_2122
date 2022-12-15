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
from trainer import Trainer3DSegmentation
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

SCANNET3D_CLASS_WEIGHTS = [1.3179, 5.0708, 5.3323, 5.3993, 5.4455, 5.3900, 5.4511, 5.4332, 5.4241,
        5.4455, 5.4347, 5.4756, 5.4731, 5.4836, 5.4488, 5.4275, 5.4555, 5.4764,
        5.4781, 5.4820, 5.4843, 5.4707, 5.4222, 5.4656, 5.4745, 5.4797, 5.4846,
        5.4789, 5.4700, 5.4705, 5.4733, 5.4839, 5.4808, 5.4722, 5.4784, 5.4814,
        5.4747, 5.4825, 5.4417, 5.4307, 5.4039] #  label 0: unoccupied, 1 wall, 2 floor, ..., 40: otherprop
def train(cfg): 
    print('Training network for 3D semantic segmentation...')
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
    
    model_3d  = SurfaceNet(cfg, num_images)
    if cfg["model_3d"]["use_pretrained_from_reconstruction"]: 
        print("Loading pretrained model from 3D reconstruction...")
        model_3d.load_state_dict(torch.load(cfg["model_3d"]["load_path_3d"])["state_dict"])
    model_3d.classifier = nn.Sequential(nn.Conv3d(100, cfg["model_3d"]["num_classes"], kernel_size= (1,1,1), padding= 0))
    print_params(model_3d)
    model_3d.to(device)


    if cfg["model_3d"]["use_pretrained_from_reconstruction"]:
        base, classifier = [], []
        module_list = list(model_3d.children())
        for module in module_list[:-2]: 
            base = base + list(module.parameters())
        for module in module_list[-2:]: 
            classifier = classifier + list(module.parameters())

        optimizer = optim.AdamW([{'params': classifier}, {'params': base, 'lr': 5e-5}], lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    else: 
        optimizer = optim.AdamW(model_3d.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])

    criterion_weights = torch.tensor(SCANNET3D_CLASS_WEIGHTS, device = 'cuda')
    criterion = nn.CrossEntropyLoss(weight = criterion_weights, ignore_index = -100)

    if cfg["use_2d_feat_input"]: 
        print('Using DeepLabv3 as 2D feature extractor')
        model_2d = DeepLabv3(cfg["model_2d"]["num_classes"])
        model_2d.load_state_dict(torch.load(cfg["model_2d"]["load_path_2d"])["state_dict"])
        for param in model_2d.parameters():
            param.requires_grad = False
        model_2d.to(device)
        #model_2d.eval()
    else: 
        model_2d = None

    #TODO change metric_3d for the task 3d segmentation
    #metric_3d = IoU(num_classes=3, ignore_index=2) # ground truth of 3D grid has 3 values:0, 1, -100. Converting label -100 to 2 we have 3 values: 0,1,2
    metric_3d = IoU(num_classes=42, ignore_index=41)
    #######################################################


    trainer = Trainer3DSegmentation(cfg, model_3d, criterion, dataloader_train, dataloader_val, projector, optimizer, device, metric_3d, model_2d = model_2d)
    trainer.train()

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Training network for 3D semantic segmentation...')
    parser.add_argument('-c', '--config', default='experiments/cfgs/segmentation_3d.json',type=str,
                        help='Path to the config file (default: segmentation_3d.json)')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    train(cfg)