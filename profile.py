from torch.profiler.profiler import profile
from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import torch
import json 
import argparse
from metric.iou import IoU
def main(config):     
    print('Analyzing ENet for 2D Semantic Segmentation task on ScanNet using Pytorch Profiler...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    trainset = ScanNet2D(config["train_loader"])
    train_loader = DataLoader(trainset, batch_size = config["train_loader"]["batch_size"],shuffle = config["train_loader"]["shuffle"], num_workers = config["train_loader"]["num_workers"], pin_memory= True)
    
    model = ENet(config["models"])
    print_params(model)
    model.to(device)

    loss = nn.CrossEntropyLoss(ignore_index = config["ignore_index"])

    optimizer = optim.Adam(model.parameters(), lr = config["optimizer"]["learning_rate"], weight_decay= config["optimizer"]["weight_decay"])

    model.train()

    with torch.profiler.profile(
        schedule= torch.profiler.schedule(skip_first=3, wait = 3, warmup = 1, active= 3, repeat = 2 ), 
        on_trace_ready= torch.profiler.tensorboard_trace_handler('./saved/profiler'), 
        record_shapes= True, 
        with_stack= True, 
        profile_memory= True
    ) as prof: 
        for step, (imgs, targets) in enumerate(train_loader, 0):
            if step >= 3 + (3+1+3)*2: 
                break 
            _train_step(imgs, targets, device, model, loss, optimizer)
            prof.step()
    print('Analyzing model done.')

def _train_step(imgs, targets, device, model, loss_func, optimizer): 
        imgs = imgs.to(device) # (N, 3, img_size, img_size)
        targets = targets.to(device) # (N, img_size, img_size)
        optimizer.zero_grad()
        outputs = model(imgs) # (N, C, img_size, img_size)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
def print_params(model): 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: %d' %(total_params))
    print('Trainable params: %d' %(trainable_params))

if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Run Pytorch Profiler')
    parser.add_argument('-c', '--config', default='train_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)