import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import shutil
from torch.utils.tensorboard import SummaryWriter
class BaseTrainer: 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, device):
        self.cfg = cfg
        self.model = model
        self.criterion = loss
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = cfg["trainer"]["epochs"]
        self.log_nth = cfg["trainer"]["log_nth"]
        self.single_sample = cfg["trainer"]["single_sample"]
        self.start_epoch = 0
        self.resume_training = cfg["trainer"]["resume_training"]

        if self.resume_training:
            checkpoint_path = cfg["trainer"]["load_path"]
            assert os.path.isfile(checkpoint_path), "No checkpoint found at {}".format(checkpoint_path)
            print("Loading checkpoint of 3D network'{}'".format(checkpoint_path))
            self.checkpoint = torch.load(checkpoint_path)
            self.start_epoch = self.checkpoint["epoch"]
            self.model.load_state_dict(self.checkpoint["state_dict"])
            self.optimizer.load_state_dict(self.checkpoint["optimizer"])
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg["optimizer"]["learning_rate"] 
            self.dir_name = os.path.basename(os.path.dirname(checkpoint_path))
        else: 
            self.dir_name = datetime.datetime.now().strftime('%m-%d_%H-%M')

        if cfg["trainer"]["training_type"] == "3D": 
            if cfg["use_2d_feat_input"]: 
                base_path = "3d_recon_2dfeat_input"
            else: 
                base_path = "3d_recon_RGB_input"
        elif cfg["trainer"]["training_type"] == "2D":
            base_path = "enet"
        else: 
            raise ValueError("training_type is unknown")

        self.writer = SummaryWriter(os.path.join("saved/runs", base_path, self.dir_name))
        self.model_folder = os.path.join("saved/models", base_path, self.dir_name)
        if not os.path.exists(self.model_folder): 
            os.makedirs(self.model_folder)
        self.checkpoint_path =  os.path.join(self.model_folder, 'checkpoint.pth.tar')
        self.best_model_path = os.path.join(self.model_folder, 'model_best.pth.tar')
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
    def save_checkpoint(self, state, is_best, checkpoint_path, best_model_path): 
        torch.save(state, checkpoint_path)
        if is_best:
            print("Saving best model path")
            shutil.copyfile(checkpoint_path, best_model_path)
    def plot_grad_flow(self,named_parameters, epoch):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
    
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n) and ('bn' not in n):
                if p.grad is not None:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())
                else:
                    print(n)
        figure = plt.figure(figsize = (26.5,14.5))
        ax = figure.add_subplot(111)
        ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")

        ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        ax.set_xticks(range(0,len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation="vertical")

        ax.set_xlim(left=0, right=len(ave_grads))
        ax.set_ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        ax.set_xlabel("Layers")
        ax.set_ylabel("average gradient")
        ax.set_title("Gradient flow")
        ax.grid(True)
        ax.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        figure.tight_layout()
        plot_dir = os.path.join('saved/plot_gradient', self.dir_name)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, 'gradient_'+'epoch_' + str(epoch+1) + '.png')
        figure.savefig(plot_path)
        plt.close(figure)