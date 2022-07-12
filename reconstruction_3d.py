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
from models import ENet
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction
from utils.helpers import print_params, make_intrinsic, adjust_intrinsic, init_weights
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
    
    #model_3d = ConvNeXtUNet(cfg, num_images)
    model_3d  = ResNeXtUNet(cfg, num_images)
    #model_3d  = SurfaceNet(cfg, num_images)
    #model_3d  = Dense3DNetwork(cfg, num_images)
    print_params(model_3d)
    model_3d.to(device)

    optimizer = optim.RAdam(model_3d.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    #optimizer = optim.AdamW(model.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    #optimizer = optim.SGD(model_3d.parameters(), lr  = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"], momentum = 0.9, nesterov= False)

    #loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([44.5], device = 'cuda'))
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 8.0], device = 'cuda'), ignore_index = -100)
    #loss = FixedCrossEntropyLoss(weight = torch.tensor([1.0, 13.0], device = 'cuda'), ignore_index = -100, label_smoothing= 0.1)
    if cfg["trainer"]["add_figure_tensorboard"]: 
        assert cfg["model_2d"]["proxy_loss"], "add_figure_tensorboard is True but proxy_loss is False"
    if cfg["model_2d"]["proxy_loss"]: 
        assert cfg["use_2d_feat_input"], "proxy_loss is True but use_2d_feat_input is False"
    if cfg["use_2d_feat_input"]: 
        model_2d = ENet(cfg["model_2d"])
        checkpoint_2d_path = cfg["model_2d"]["load_path_2d"]
        assert checkpoint_2d_path, "load_path_2d is empty"
        assert os.path.isfile(checkpoint_2d_path), "path to 2D model checkpoint does not exist"
        model_2d_checkpoint = torch.load(checkpoint_2d_path)
        model_2d.load_state_dict(model_2d_checkpoint["state_dict"])
        for i, layer in enumerate(model_2d.children()): 
            if i < 15: 
                for param in layer.parameters():
                    param.requires_grad = False
        
        model_2d.to(device)
        model_2d.eval() # set all layer to evaluation mode, and later set trainable layer to train mode 
        optimizer2d = optim.RAdam(model_2d.parameters(), lr = cfg["optimizer_2d"]["learning_rate"], weight_decay= cfg["optimizer_2d"]["weight_decay"])
        #optimizer2d = optim.SGD(model_2d.parameters(), lr  = cfg["optimizer_2d"]["learning_rate"], weight_decay= cfg["optimizer_2d"]["weight_decay"], momentum = 0, nesterov= False)
        if cfg["model_2d"]["proxy_loss"]: 
            criterion2d = nn.CrossEntropyLoss(ignore_index = cfg["model_2d"]["ignore_index"])
        else: 
            criterion2d = None
    else: 
        model_2d = None
        optimizer2d = None
        criterion2d = None
    metric_3d = IoU(num_classes=3, ignore_index=2) # ground truth of 3D grid has 3 values:0, 1, -100. Converting label -100 to 2 we have 3 values: 0,1,2

    trainer = Trainer3DReconstruction(cfg, model_3d, criterion, dataloader_train, dataloader_val, projector, optimizer, device, metric_3d, model_2d = model_2d, optimizer2d = optimizer2d, criterion2d = criterion2d)
    trainer.train()
    
def test(cfg): 
    pass
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
    
    