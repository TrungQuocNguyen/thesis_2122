import json 
import argparse
import os
import torch
from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet, create_enet
import numpy as np 
from metric.iou import IoU
from utils.helpers import CLASS_LABELS
from utils.helpers import plot_preds
from torch.utils.tensorboard import SummaryWriter
def main(config): 
    if config["add_figure_tensorboard"]: 
        dir_name = os.path.basename(os.path.dirname(config["models"]["load_path"]))
        writer = SummaryWriter(os.path.join("saved/test_results/enet", dir_name))
    print('Testing trained ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    testset = ScanNet2D(config["test_loader"])
    test_loader = DataLoader(testset, batch_size = config["test_loader"]["batch_size"],shuffle = config["test_loader"]["shuffle"], num_workers = config["test_loader"]["num_workers"], pin_memory= True)
    #model = ENet(config["models"])
    
    model = create_enet(config["models"]["num_classes"])
    checkpoint = torch.load(config["models"]["load_path"])
    print("epoch %d: "%(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = config["ignore_index"])
    metric = IoU(num_classes=config["models"]["num_classes"], ignore_index=config["IoU_ignore_index"])
    metric.reset()
    val_losses = []
    mean = torch.tensor(config["test_loader"]["mean"]).reshape(1,3,1,1)
    std = torch.tensor(config["test_loader"]["std"]).reshape(1,3,1,1)
    with torch.no_grad(): 
        for i, (imgs, targets) in enumerate(test_loader, 0): 
            print(i)
            loss, preds= _eval_step(imgs, targets, device, model, loss_fn, metric)
            val_losses.append(loss)
            if config["add_figure_tensorboard"]: 
                    writer.add_figure('test predictions vs targets', plot_preds(imgs*std+mean, targets, preds), global_step =i)
    iou, miou = metric.value()

    val_loss = np.mean(val_losses)
    print('TEST loss: %.3f' %(val_loss))
    for label, class_iou in zip(CLASS_LABELS, iou):
        print("{0}: {1:.4f}".format(label, class_iou))
    print('TEST mIoU: %.3f' %(miou))

def _eval_step(imgs, targets, device, model, loss_fn, metric): 
    imgs = imgs.to(device)
    targets = targets.to(device)

    outputs = model(imgs)
    loss = loss_fn(outputs, targets)
    loss = loss.cpu().detach().numpy()
    _, preds = torch.max(outputs, 1)
    metric.add(preds.detach(), targets.detach())
    return loss, preds.cpu().detach()
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='experiments/cfgs/test_enet.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)