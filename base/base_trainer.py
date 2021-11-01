import os
import datetime
from torch.utils.tensorboard import SummaryWriter
class BaseTrainer: 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, device):

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

        write_dir = os.path.join('saved/runs', start_time)
        self.writer = SummaryWriter(write_dir)
    def train(self): 
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            if not self.single_sample: 
                self._val_epoch(epoch) 

    def _train_epoch(self, epoch): 
        raise NotImplementedError
    def _val_epoch(self, epoch): 
        raise NotImplementedError
        