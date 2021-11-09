import json 
import argparse
import os
import torch
from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import numpy as np 
def main(config): 
    print('Testing trained ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    testset = ScanNet2D(config["test_loader"])
    test_loader = DataLoader(testset, batch_size = config["test_loader"]["batch_size"],shuffle = config["test_loader"]["shuffle"], num_workers = config["test_loader"]["num_workers"])
    model = ENet(config["models"])

    checkpoint = torch.load(os.path.join(config["models"]["load_path"],"model_best.pth.tar"))
    print("epoch %d: "%(checkpoint["epoch"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index = config["ignore_index"])

    val_losses = []
    val_accs = []
    with torch.no_grad(): 
        for (imgs, targets) in test_loader: 
            loss, acc, preds= _eval_step(imgs, targets, device, model, loss_fn)
            val_losses.append(loss)
            val_accs.append(acc)
            

    val_loss, val_acc = np.mean(val_losses), np.mean(val_accs)
    print('TEST loss/acc: %.3f/%.3f' %(val_loss, val_acc))


def _eval_step(imgs, targets, device, model, loss_fn): 
    imgs = imgs.to(device)
    targets = targets.to(device)

    outputs = model(imgs)
    loss = loss_fn(outputs, targets)
    loss = loss.cpu().detach().numpy()
    _, preds = torch.max(outputs, 1)
    target_mask = targets >0
    acc = np.mean((preds == targets)[target_mask].cpu().detach().numpy())
    preds = preds.cpu().detach().numpy()
        
    return loss, acc, preds
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='test_config.json',type=str,
                        help='Path to the config file (default: config.json)')
    args = parser.parse_args()
    config = json.load(open(args.config))
    main(config)