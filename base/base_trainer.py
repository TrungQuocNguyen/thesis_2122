import os
import datetime
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
class BaseTrainer: 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, metric, device):
        self.cfg = cfg
        self.model = model
        self.loss_func = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.epochs = cfg["epochs"]
        self.log_nth = cfg["log_nth"]
        self.single_sample = cfg["single_sample"]
        self.add_figure_tensorboard = cfg["add_figure_tensorboard"]
        self.start_epoch = 0
        self.best_miou = 0

        if cfg["load_path"]:
            checkpoint_path = cfg["load_path"]
            if os.path.isfile(checkpoint_path):
                print("Loading checkpoint '{}'".format(checkpoint_path))
                checkpoint = torch.load(checkpoint_path)
                self.start_epoch = checkpoint["epoch"]
                self.best_miou = checkpoint["best_miou"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                dir_name = os.path.basename(os.path.dirname(checkpoint_path))
            else:
                print("No checkpoint found at {}".format(checkpoint_path))
        else: 
            dir_name = datetime.datetime.now().strftime('%m-%d_%H-%M')

        self.writer = SummaryWriter(os.path.join("saved/runs", dir_name))
        model_path = os.path.join("saved/models", dir_name)
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        self.checkpoint_path =  os.path.join(model_path, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(model_path, 'model_best.pth.tar')
        #what would happen if I load from checkpoint.pth.tar / model_best.pth.tar ?
        #load from checkpoint.pth.tar: continue training from where we left off
        #load from model_best.pth.tar: training from the best model 


        #self.epochs = 1500
    def train(self): 
        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch)
            if not self.single_sample: 
                miou = self._val_epoch(epoch)
                is_best = miou > self.best_miou
                self.best_miou = max(miou, self.best_miou)
                self.save_checkpoint({
                    'epoch': epoch+1, 
                    'state_dict': self.model.state_dict(), 
                    'best_miou': self.best_miou, 
                    'optimizer': self.optimizer.state_dict(),
                }, is_best)

    def _train_epoch(self, epoch): 
        raise NotImplementedError
    def _val_epoch(self, epoch): 
        raise NotImplementedError
    def save_checkpoint(self, state, is_best): 
        torch.save(state, self.checkpoint_path)
        if is_best:
            print("Saving best model path")
            shutil.copyfile(self.checkpoint_path, self.best_model_path)
        