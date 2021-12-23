import os
import datetime
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
class BaseTrainer: 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.loss_func = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = cfg["trainer"]["epochs"]
        self.log_nth = cfg["trainer"]["log_nth"]
        self.single_sample = cfg["trainer"]["single_sample"]
        self.start_epoch = 0

        if cfg["trainer"]["load_path"]:
            checkpoint_path = cfg["trainer"]["load_path"]
            if os.path.isfile(checkpoint_path):
                print("Loading checkpoint '{}'".format(checkpoint_path))
                self.checkpoint = torch.load(checkpoint_path)
                self.start_epoch = self.checkpoint["epoch"]
                self.model.load_state_dict(self.checkpoint["state_dict"])
                self.optimizer.load_state_dict(self.checkpoint["optimizer"])

                dir_name = os.path.basename(os.path.dirname(checkpoint_path))
            else:
                print("No checkpoint found at {}".format(checkpoint_path))
        else: 
            dir_name = datetime.datetime.now().strftime('%m-%d_%H-%M')

        self.writer = SummaryWriter(os.path.join("saved/runs/3d_reconstruction", dir_name))
        model_path = os.path.join("saved/models/3d_reconstruction", dir_name)
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        self.checkpoint_path =  os.path.join(model_path, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(model_path, 'model_best.pth.tar')
        #what would happen if I load from checkpoint.pth.tar / model_best.pth.tar ?
        #load from checkpoint.pth.tar: continue training from where we left off
        #load from model_best.pth.tar: training from the best model 


        #self.epochs = 1500
    def train(self): 
        raise NotImplementedError
    def _train_epoch(self, epoch): 
        raise NotImplementedError
    def _val_epoch(self, epoch): 
        raise NotImplementedError
    def save_checkpoint(self, state, is_best): 
        torch.save(state, self.checkpoint_path)
        if is_best:
            print("Saving best model path")
            shutil.copyfile(self.checkpoint_path, self.best_model_path)
        