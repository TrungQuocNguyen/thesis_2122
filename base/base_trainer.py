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
        self.epochs = cfg["epochs"]
        self.log_nth = cfg["log_nth"]
        self.single_sample = cfg["single_sample"]
        self.add_figure_tensorboard = cfg["add_figure_tensorboard"]
        
        self.train_loss_history = []
        self.train_acc_history = []

        #TENSORBOARDS
        start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
        write_dir = os.path.join(cfg["tensorboard_path"], start_time)
        self.writer = SummaryWriter(write_dir)
        
        model_path = os.path.join(self.cfg["models_path"], start_time)
        if not os.path.exists(model_path): 
            os.makedirs(model_path)
        self.checkpoint_path =  os.path.join(model_path, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(model_path, 'model_best.pth.tar')
        
    def train(self): 
        best_acc = 0
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            if not self.single_sample: 
                val_acc = self._val_epoch(epoch)
                val_acc = val_acc.item()
                is_best = val_acc > best_acc
                best_acc = max(val_acc, best_acc)
                self.save_checkpoint({
                    'epoch': epoch+1, 
                    'state_dict': self.model.state_dict(), 
                    'best_acc': best_acc, 
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
        