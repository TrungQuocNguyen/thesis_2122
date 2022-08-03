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
from models import Model3DResNeXt
from models import ENet, create_enet_for_3d
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction
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

SCANNET2D_CLASS_WEIGHTS = [3.5664, 2.9007, 3.1645, 4.5708, 4.7439, 4.4247, 4.9056, 4.5828, 4.6723,
        5.1634, 5.2238, 5.3857, 5.3079, 5.4757, 5.0380, 5.1324, 5.2344, 5.3254,
        5.3597, 5.4598, 5.4675, 5.3584, 5.3784, 5.3338, 5.2996, 5.4221, 5.4798,
        5.4038, 5.3744, 5.3701, 5.3716, 5.4738, 5.4278, 5.3312, 5.3813, 5.4588,
        5.3192, 5.4527, 5.0615, 4.8410, 4.5946] #  label 0: unanottated, 40: otherprop

class FixedCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    Standard CrossEntropyLoss with label_smoothing doesn't handle ignore_index properly, so we apply
    the mask ourselves. See https://github.com/pytorch/pytorch/issues/73205
    """
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor: # input: [N, C, H, W], target: [N, H, W]
        if not target.is_floating_point() and self.ignore_index is not None:
            input = input.permute(0,2,3,1)[target!=self.ignore_index]
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
    #model_3d  = ResNeXtUNet(cfg, num_images)
    #model_3d  = SurfaceNet(cfg, num_images)
    #model_3d  = Dense3DNetwork(cfg, num_images)
    model_3d = Model3DResNeXt(cfg, num_images)
    print_params(model_3d)
    model_3d.to(device)

    #optimizer = optim.RAdam(model_3d.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    optimizer = optim.AdamW(model_3d.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    #optimizer = optim.SGD(model_3d.parameters(), lr  = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"], momentum = 0.9, nesterov= False)

    #loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([44.5], device = 'cuda'))
    criterion = nn.CrossEntropyLoss(weight = torch.tensor([1.0, 8.0], device = 'cuda'), ignore_index = -100)
    #loss = FixedCrossEntropyLoss(weight = torch.tensor([1.0, 13.0], device = 'cuda'), ignore_index = -100, label_smoothing= 0.1)
    if cfg["trainer"]["add_figure_tensorboard"]: 
        assert cfg["model_2d"]["proxy_loss"], "add_figure_tensorboard is True but proxy_loss is False"
    if cfg["model_2d"]["proxy_loss"]: 
        assert cfg["use_2d_feat_input"], "proxy_loss is True but use_2d_feat_input is False"
    if cfg["use_2d_feat_input"]: 
        #model_2d = ENet(cfg["model_2d"])
        #checkpoint_2d_path = cfg["model_2d"]["load_path_2d"]
        #assert checkpoint_2d_path, "load_path_2d is empty"
        #assert os.path.isfile(checkpoint_2d_path), "path to 2D model checkpoint does not exist"
        #model_2d_checkpoint = torch.load(checkpoint_2d_path)
        #model_2d.load_state_dict(model_2d_checkpoint["state_dict"])
        #for i, layer in enumerate(model_2d.children()): 
        #    if i < 15: 
        #        for param in layer.parameters():
        #            param.requires_grad = False
        
        #model_2d.to(device)
        #model_2d.eval() # set all layer to evaluation mode, and later set trainable layer to train mode 
        model_2d_fixed, model_2d_trainable, model_2d_classification = create_enet_for_3d(cfg["model_2d"]["num_classes"], cfg["model_2d"]["load_path_2d"])
        model_2d_fixed.to(device)
        model_2d_fixed.eval()
        model_2d_trainable.to(device)
        model_2d_classification.to(device)

        optimizer2d = optim.AdamW([{'params': model_2d_trainable.parameters()}, {'params': model_2d_classification.parameters()}], lr = cfg["optimizer_2d"]["learning_rate"], weight_decay= cfg["optimizer_2d"]["weight_decay"])
        #optimizer2d = optim.SGD(model_2d.parameters(), lr  = cfg["optimizer_2d"]["learning_rate"], weight_decay= cfg["optimizer_2d"]["weight_decay"], momentum = 0, nesterov= False)
        if cfg["model_2d"]["proxy_loss"]: 
            criterion_weights = torch.tensor(SCANNET2D_CLASS_WEIGHTS, device = 'cuda')
            #criterion2d = nn.CrossEntropyLoss(ignore_index = cfg["model_2d"]["ignore_index"])
            criterion2d = FixedCrossEntropyLoss(weight= criterion_weights, ignore_index = cfg["model_2d"]["ignore_index"], label_smoothing= 0.1)
            metric_2d = IoU(num_classes=cfg["model_2d"]["num_classes"], ignore_index=cfg["model_2d"]["IoU_ignore_index"])
            metric_2d_all_classes = IoU(num_classes=cfg["model_2d"]["num_classes"], ignore_index= cfg["model_2d"]["ignore_index"])
        else: 
            criterion2d = None
            metric_2d = None
            metric_2d_all_classes = None
    else: 
        model_2d_fixed = None
        model_2d_trainable = None
        model_2d_classification = None
        optimizer2d = None
        criterion2d = None
        metric_2d = None
        metric_2d_all_classes = None
    metric_3d = IoU(num_classes=3, ignore_index=2) # ground truth of 3D grid has 3 values:0, 1, -100. Converting label -100 to 2 we have 3 values: 0,1,2

    trainer = Trainer3DReconstruction(cfg, model_3d, criterion, dataloader_train, dataloader_val, projector, optimizer, device, metric_3d, model_2d_fixed = model_2d_fixed, model_2d_trainable = model_2d_trainable, model_2d_classification = model_2d_classification, optimizer2d = optimizer2d, criterion2d = criterion2d, metric_2d = metric_2d, metric_2d_all_classes = metric_2d_all_classes)
    trainer.train()
    
def test(cfg): 
    pass
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Training network for 3D reconstruction task')
    parser.add_argument('-c', '--config', default='experiments/cfgs/rgb_input_3d_recon.json',type=str,
                        help='Path to the config file (default: pretrained_feat_input_3d_recon.json)')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.mode == 'train': 
        cfg["mode"] = 'train'
        train(cfg)
    
    