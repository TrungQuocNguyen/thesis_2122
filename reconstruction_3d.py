import argparse
import json 
import torch
import torch.nn as nn
import torch.optim as optim
from models import Dense3DNetwork, SurfaceNet
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction
from utils.helpers import print_params, make_intrinsic, adjust_intrinsic, init_weights
def train(cfg): 
    print('Training network for 3D reconstruction...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    print(device)

    dataset_train = ScanNet2D3D(cfg, cfg["TRAIN_FILELIST"], mode='chunk')
    dataset_val = ScanNet2D3D(cfg, cfg["VAL_FILELIST"], mode='chunk')
    num_images = len(dataset_train[0]["nearest_images"]["depths"]) # number of surrounding images of a chunk 
    if cfg["NUM_IMAGES"] < num_images: 
        num_images = cfg["NUM_IMAGES"]

    dataloader_train = get_dataloader(cfg, dataset_train, batch_size= cfg["BATCH_SIZE"], shuffle = True, num_workers=4)
    dataloader_val = get_dataloader(cfg, dataset_val, batch_size= cfg["BATCH_SIZE"], shuffle= True, num_workers=4)

    intrinsic = make_intrinsic(cfg["fx"], cfg["fy"], cfg["mx"], cfg["my"])
    intrinsic = adjust_intrinsic(intrinsic, [cfg["INTRINSIC_IMAGE_WIDTH"], cfg["INTRINSIC_IMAGE_HEIGHT"]], cfg["DEPTH_SHAPE"])


    model  = Dense3DNetwork(cfg, num_images)
    #model.apply(init_weights)
    print_params(model)
    model.to(device)

    loss = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([41.58], device = 'cuda'))
    #loss = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    
    

    trainer = Trainer3DReconstruction(cfg, model, loss, dataloader_train, dataloader_val, intrinsic, optimizer, device)
    trainer.train()
    
def test(cfg): 
    pass
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Training network for 3D reconstruction task')
    parser.add_argument('-c', '--config', default='experiments/cfgs/overfit_3d_reconstruction.json',type=str,
                        help='Path to the config file (default: train_3d_reconstruction.json)')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.mode == 'train': 
        cfg["MODE"] = 'train'
        train(cfg)
    
    