from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import torch.nn as nn
import torch.optim as optim
from trainer import TrainerENet
from utils.helpers import print_params
import torch
import json 
import argparse
from metric.iou import IoU
def main(config):     
    print('Training ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    trainset = ScanNet2D(config["train_loader"])
    train_loader = DataLoader(trainset, batch_size = config["train_loader"]["batch_size"],shuffle = config["train_loader"]["shuffle"], num_workers = config["train_loader"]["num_workers"], pin_memory= True)

    valset = ScanNet2D(config["val_loader"])
    val_loader = DataLoader(valset, batch_size = config["val_loader"]["batch_size"],shuffle = config["val_loader"]["shuffle"], num_workers = config["val_loader"]["num_workers"], pin_memory= True)
    
    model = ENet(config["models"])
    print_params(model)
    model.to(device)

    loss = nn.CrossEntropyLoss(ignore_index = config["ignore_index"])

    optimizer = optim.Adam(model.parameters(), lr = config["optimizer"]["learning_rate"], weight_decay= config["optimizer"]["weight_decay"])
    metric = IoU(num_classes=config["models"]["num_classes"], ignore_index=config["IoU_ignore_index"])

    trainer = TrainerENet(config, model, loss, train_loader, val_loader, optimizer, metric, device)
    trainer.train()


if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='experiments/cfgs/train_enet.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)