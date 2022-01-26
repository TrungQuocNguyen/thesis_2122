
import json 

from datasets import ScanNet2D3D

cfg = json.load(open('experiments/cfgs/overfit_3d_reconstruction.json'))
cfg["MODE"] = 'train'
dataset_train = ScanNet2D3D(cfg, cfg["TRAIN_FILELIST"], mode='chunk')
#dataset_val = ScanNet2D3D(cfg, cfg["VAL_FILELIST"], mode='chunk')
with open(cfg["TRAIN_FILELIST"], "r") as f:
#with open(cfg["VAL_FILELIST"], "r") as f:
    lines = f.readlines()
    f.close()
with open("experiments/filelists/ScanNet/v2/train_new2.txt", "w") as f:
    #for i in range(len(dataset_val)):
    #    sample = dataset_val[i]
    for i in range(len(dataset_train)):
        sample = dataset_train[i] 
        if len(sample['image_files']) != 5:
            #del lines[i]
            print(len(sample['image_files']))
            print(sample['id'])
#with open("experiments/filelists/ScanNet/v2/val_new.txt", "w") as f:
        else: 
            f.write(lines[i])
