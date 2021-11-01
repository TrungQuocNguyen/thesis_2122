from datasets import ScanNet2D
from torch.utils.data import DataLoader
from models import ENet
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer
import torch
def main(): 
    print('Training ENet for 2D Semantic Segmentation task on ScanNet...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 4
    epochs = 2000
    log_nth = 2
    single_sample = False # set to True will disable the validation set
    learning_rate =4e-3
    img_size =(64,64)
    add_figure_tensorboard = False

    trainset = ScanNet2D('/mnt/raid/tnguyen/scannet_10_images/trainset', '/mnt/raid/tnguyen/labels_10_images/trainset', img_size = img_size, is_transform= True, augmentation= False)
    train_loader = DataLoader(trainset, batch_size = batch_size,shuffle = True, num_workers = 4)

    valset = ScanNet2D('/mnt/raid/tnguyen/scannet_10_images/valset', '/mnt/raid/tnguyen/labels_10_images/valset', img_size = img_size, is_transform= True, augmentation= False)
    val_loader = DataLoader(valset, batch_size = batch_size,shuffle = True, num_workers = 4)
    
    model = ENet(num_classes= 41, in_channels=3, freeze_bn= True, freeze_dropout= True)
    print_params(model)
    model.to(device)

    loss = nn.CrossEntropyLoss(ignore_index = 0)

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    trainer = Trainer(model, loss, train_loader, val_loader, optimizer, epochs, log_nth, device, single_sample, add_figure_tensorboard)
    trainer.train()

def print_params(model): 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: %d' %(total_params))
    print('Trainable params: %d' %(trainable_params))

if __name__ =='__main__': 
    main()