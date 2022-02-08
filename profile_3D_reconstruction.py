import argparse
import json 
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Dense3DNetwork, SurfaceNet
from datasets import ScanNet2D3D, get_dataloader
from trainer import Trainer3DReconstruction
from utils.helpers import print_params, make_intrinsic, adjust_intrinsic, init_weights
from projection import ProjectionHelper
seed_value= 1234
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set `pytorch` pseudo-random generator at a fixed value
torch.manual_seed(seed_value)


def train(cfg): 
    print('Analyzing network for 3D reconstruction using Torch Profiler')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    print(device)

    dataset_train = ScanNet2D3D(cfg, cfg["TRAIN_FILELIST"], mode='chunk')
    dataset_val = ScanNet2D3D(cfg, cfg["VAL_FILELIST"], mode='chunk')
    num_images = len(dataset_train[0]["nearest_images"]["depths"]) # number of surrounding images of a chunk 
    if cfg["NUM_IMAGES"] < num_images: 
        num_images = cfg["NUM_IMAGES"]

    dataloader_train = get_dataloader(cfg, dataset_train, batch_size= cfg["BATCH_SIZE"], shuffle = cfg["SHUFFLE_TRAIN"], num_workers=cfg["NUM_WORKERS"], pin_memory= cfg["PIN_MEMORY"])
    dataloader_val = get_dataloader(cfg, dataset_val, batch_size= cfg["BATCH_SIZE"], shuffle= cfg["SHUFFLE_VAL"], num_workers=cfg["NUM_WORKERS"], pin_memory= cfg["PIN_MEMORY"])

    intrinsic = make_intrinsic(cfg["fx"], cfg["fy"], cfg["mx"], cfg["my"])
    intrinsic = adjust_intrinsic(intrinsic, [cfg["INTRINSIC_IMAGE_WIDTH"], cfg["INTRINSIC_IMAGE_HEIGHT"]], cfg["DEPTH_SHAPE"])


    model  = Dense3DNetwork(cfg, num_images)
    #model.apply(init_weights)
    print_params(model)
    model.to(device)

    loss_func = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([44.5], device = 'cuda'))
    #loss = nn.L1Loss()

    optimizer = optim.Adam(model.parameters(), lr = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"])
    #optimizer = optim.SGD(model.parameters(), lr  = cfg["optimizer"]["learning_rate"], weight_decay= cfg["optimizer"]["weight_decay"], momentum = 0, nesterov= False)


    model.train()

    with torch.profiler.profile(
        schedule= torch.profiler.schedule(skip_first=4, wait = 4, warmup = 2, active= 6, repeat = 2 ), 
        on_trace_ready= torch.profiler.tensorboard_trace_handler('./saved/profiler/3d_reconstruction'), 
        record_shapes= True, 
        with_stack= True, 
        profile_memory= True
    ) as prof: 
        val_iterator = iter(dataloader_val)
        for step, batch in enumerate(dataloader_train, 0):
            blobs = batch.data
            model.train()
            if step >= 2 + (2+1+3)*2: 
                break 
            jump_flag = _voxel_pixel_association(cfg, blobs, device, intrinsic)
            if jump_flag:
                print('error in train batch, skipping the current batch...') 
                continue
            loss = _train_step(blobs, device, model, loss_func, optimizer)
            print('TRAIN loss: %.3f' %(loss))
            prof.step()

            model.eval()
            with torch.no_grad(): 
                try: 
                    blobs_val = next(val_iterator).data
                except StopIteration: 
                    val_iterator = iter(dataloader_val)
                    blobs_val = next(val_iterator).data
                jump_flag = _voxel_pixel_association(cfg, blobs_val, device, intrinsic)
                if jump_flag: 
                    print('error in single validation batch, skipping the current batch...')
                    continue
                val_loss = _eval_step(blobs_val, device, model, loss_func)
            print('VAL loss: %.3f' %(val_loss))
            prof.step()

    print('Analyzing model done.')

def _eval_step(blobs, device, model, loss_func): 
        targets = blobs['data'].to(device) # [N, 1, 96, 48, 96] 
        preds = model(blobs,device) #[N, 1, 96, 48, 96]
        loss = loss_func(preds, targets)
        return loss.item()

def _train_step(blobs, device, model, loss_func, optimizer): 
        targets = blobs['data'].to(device) # [N, 1, 96, 48, 96] 
        preds = model(blobs, device) #[N, 1, 96, 48, 96]
        loss = loss_func(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

def _voxel_pixel_association(cfg, blobs, device, intrinsic): 
        batch_size = blobs['data'].shape[0]
        grid_shape = blobs['data'].shape[-3:] # [96,48,96]
        projection_helper = ProjectionHelper(intrinsic, cfg["PROJ_DEPTH_MIN"], cfg["PROJ_DEPTH_MAX"], cfg["DEPTH_SHAPE"], grid_shape, cfg["VOXEL_SIZE"], device)
        proj_mapping = [[projection_helper.compute_projection(d.to(device), c.to(device), t.to(device)) for d, c, t in zip(blobs['nearest_images']['depths'][i], blobs['nearest_images']['poses'][i], blobs['nearest_images']['world2grid'][i])] for i in range(batch_size)]
        blobs['proj_ind_3d'] = []
        blobs['proj_ind_2d'] = []
        jump_flag = False
        for i in range(batch_size):
            if None in proj_mapping[i]: #invalid sample
                jump_flag = True
                break
        if  not jump_flag: 
            for i in range(batch_size):
                proj_mapping0, proj_mapping1 = zip(*proj_mapping[i])
                blobs['proj_ind_3d'].append(torch.stack(proj_mapping0)) # list of [max_num_images,96*48*96 + 1], total batch_size elements in the list 
                blobs['proj_ind_2d'].append(torch.stack(proj_mapping1)) # list of [max_num_images,96*48*96 + 1], total batch_size elements in the list      
        return jump_flag

    
if __name__ =='__main__': 
    parser = argparse.ArgumentParser(description='Profiling network for 3D reconstruction task')
    parser.add_argument('-c', '--config', default='experiments/cfgs/overfit_3d_reconstruction.json',type=str,
                        help='Path to the config file (default: train_3d_reconstruction.json)')
    parser.add_argument('--mode', type=str, default='train')
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    if args.mode == 'train': 
        cfg["MODE"] = 'train'
        train(cfg)