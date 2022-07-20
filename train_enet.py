from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet, create_enet
import torch.nn as nn
import torch.optim as optim
from trainer import TrainerENet
from utils.helpers import print_params
import torch
import json 
import argparse
from metric.iou import IoU
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
def main(config):     
    print('Training ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    trainset = ScanNet2D(config["train_loader"])
    train_loader = DataLoader(trainset, batch_size = config["train_loader"]["batch_size"],shuffle = config["train_loader"]["shuffle"], num_workers = config["train_loader"]["num_workers"], pin_memory= True)

    valset = ScanNet2D(config["val_loader"])
    val_loader = DataLoader(valset, batch_size = config["val_loader"]["batch_size"],shuffle = config["val_loader"]["shuffle"], num_workers = config["val_loader"]["num_workers"], pin_memory= True)
    
    #model = ENet(config["models"])
    model = create_enet(config["models"]["num_classes"])
    print_params(model)
    model.to(device)

    criterion_weights = torch.tensor(SCANNET2D_CLASS_WEIGHTS, device = 'cuda')
    #loss = nn.CrossEntropyLoss(weight= criterion_weights, ignore_index = config["ignore_index"])
    loss = FixedCrossEntropyLoss(weight= criterion_weights, ignore_index = config["ignore_index"], label_smoothing= 0.1)
    optimizer = optim.AdamW(model.parameters(), lr = config["optimizer"]["learning_rate"], weight_decay= config["optimizer"]["weight_decay"])
    metric = IoU(num_classes=config["models"]["num_classes"], ignore_index=config["IoU_ignore_index"])
    metric_all_classes = IoU(num_classes=config["models"]["num_classes"], ignore_index= config["ignore_index"])

    trainer = TrainerENet(config, model, loss, train_loader, val_loader, optimizer, metric, metric_all_classes, device)
    trainer.train()


if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='experiments/cfgs/train_enet.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)