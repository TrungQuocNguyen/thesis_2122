from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import torch
import json 
import argparse
def main(config):     
    print('Training ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    trainset = ScanNet2D(config["train_loader"])
    train_loader = DataLoader(trainset, batch_size = config["train_loader"]["batch_size"],shuffle = config["train_loader"]["shuffle"], num_workers = config["train_loader"]["num_workers"])

    valset = ScanNet2D(config["val_loader"])
    val_loader = DataLoader(valset, batch_size = config["val_loader"]["batch_size"],shuffle = config["val_loader"]["shuffle"], num_workers = config["val_loader"]["num_workers"])
    
    model = ENet(config["models"])
    print_params(model)
    model.to(device)

    loss = nn.CrossEntropyLoss(ignore_index = config["ignore_index"])

    optimizer = optim.Adam(model.parameters(), lr = config["optimizer"]["learning_rate"])

    trainer = Trainer(config["trainer"], model, loss, train_loader, val_loader, optimizer, device)
    trainer.train()

def print_params(model): 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: %d' %(total_params))
    print('Trainable params: %d' %(trainable_params))

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)