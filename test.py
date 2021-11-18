import json 
import argparse
import os
import torch
from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import numpy as np 
from metric.iou import IoU
from utils.helpers import CLASS_LABELS
from utils.helpers import plot_preds
from torch.utils.tensorboard import SummaryWriter
def main(config): 
    dir_name = os.path.basename(os.path.dirname(config["models"]["load_path"]))
    writer = SummaryWriter(os.path.join("saved/test_results", dir_name))
    print('Testing trained ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    testset = ScanNet2D(config["test_loader"])
    test_loader = DataLoader(testset, batch_size = config["test_loader"]["batch_size"],shuffle = config["test_loader"]["shuffle"], num_workers = config["test_loader"]["num_workers"], pin_memory= True)
    model = ENet(config["models"])

    checkpoint = torch.load(config["models"]["load_path"])
    print("epoch %d: "%(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = config["ignore_index"])
    metric = IoU(num_classes=config["models"]["num_classes"], ignore_index=config["IoU_ignore_index"])
    metric.reset()
    val_losses = []
    val_accs = []
    mean = torch.tensor(config["test_loader"]["mean"]).reshape(1,3,1,1)
    std = torch.tensor(config["test_loader"]["std"]).reshape(1,3,1,1)
    with torch.no_grad(): 
        for i, (imgs, targets) in enumerate(test_loader, 0): 
            loss, acc, preds= _eval_step(imgs, targets, device, model, loss_fn, metric)
            val_losses.append(loss)
            val_accs.append(acc)
            if config["add_figure_tensorboard"]: 
                    writer.add_figure('test predictions vs targets', plot_preds(imgs*std+mean, targets, preds), global_step =i)
    iou, miou = metric.value()

    val_loss, val_acc = np.mean(val_losses), np.mean(val_accs)
    print('TEST loss/acc: %.3f/%.3f' %(val_loss, val_acc))
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
    target_mask = targets >0
    acc = np.mean((preds == targets)[target_mask].cpu().detach().numpy())
    metric.add(preds.detach(), targets.detach())
    return loss, acc, preds.cpu().detach()
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='test_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)