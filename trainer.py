from base import BaseTrainer
import numpy as np 
import torch
from utils.helpers import plot_preds
class Trainer(BaseTrainer): 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, metric, device):
        super(Trainer, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, metric, device)

    def _train_epoch(self, epoch):
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        self.model.train()
        for i, (imgs, targets) in enumerate(self.train_loader, 0): 
            loss, acc, preds = self._train_step(imgs, targets)
            train_loss.update(loss, imgs.size(0))
            train_acc.update(acc, imgs.size(0))
            if self.log_nth and i % self.log_nth == self.log_nth-1: 
                mean = torch.tensor(self.train_loader.dataset.mean).reshape(1,3,1,1)
                std = torch.tensor(self.train_loader.dataset.std).reshape(1,3,1,1)
                self.writer.add_scalar('train_loss', train_loss.val, global_step= len(self.train_loader)*epoch + i )
                self.writer.add_scalar('train_accuracy', train_acc.val, global_step= len(self.train_loader)*epoch + i )
                print('[Iteration %d/%d] TRAIN loss: %.3f(%.3f)   TRAIN accuracy: %.3f(%.3f)' %(len(self.train_loader)*epoch + i+1, len(self.train_loader)*self.epochs, train_loss.val, train_loss.avg, train_acc.val, train_acc.avg))
                if self.add_figure_tensorboard: 
                    self.writer.add_figure('train predictions vs targets', plot_preds(imgs*std+mean, targets, preds), global_step = len(self.train_loader)*epoch + i)
                if not self.single_sample: 
                    self.model.eval()
                    with torch.no_grad(): 
                        imgs, targets = next(iter(self.val_loader))
                        val_loss, val_acc, val_preds = self._eval_step(imgs, targets)
                    self.writer.add_scalar('val_loss', val_loss, global_step= len(self.train_loader)*epoch + i)
                    self.writer.add_scalar('val_accuracy', val_acc, global_step= len(self.train_loader)*epoch + i)
                    print('[Iteration %d/%d] VAL loss: %.3f   VAL accuracy: %.3f' %(len(self.train_loader)*epoch + i+1, len(self.train_loader)*self.epochs, val_loss, val_acc))
                    if self.add_figure_tensorboard: 
                        self.writer.add_figure('val predictions vs targets', plot_preds(imgs*std+mean, targets, val_preds), global_step = len(self.train_loader)*epoch + i)
        if self.log_nth and not self.single_sample: 
            print('[Epoch %d/%d] TRAIN loss/acc: %.3f/%.3f' %(epoch+1, self.epochs, train_loss.avg, train_acc.avg))

    def _train_step(self, imgs, targets): 
        imgs = imgs.to(self.device) # (N, 3, img_size, img_size)
        targets = targets.to(self.device) # (N, img_size, img_size)
        self.optimizer.zero_grad()
        outputs = self.model(imgs) # (N, C, img_size, img_size)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        self.optimizer.step()

        _, preds = torch.max(outputs, 1) # [N, img_size, img_size]
        target_mask = targets >0
        acc = torch.mean((preds == targets)[target_mask].float())     
        preds = preds.cpu().detach()    
        return loss.item(), acc.item(), preds

    def _val_epoch(self, epoch): 
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        self.model.eval()
        self.metric.reset()
        with torch.no_grad(): 
            for (imgs, targets) in self.val_loader: 
               loss, acc, _= self._eval_step(imgs, targets)
               val_loss.update(loss, imgs.size(0))
               val_acc.update(acc, imgs.size(0))
        iou, miou = self.metric.value()
        if self.log_nth: 
            self.writer.add_scalar('val_epoch_loss', val_loss.avg, global_step= epoch)
            self.writer.add_scalar('val_epoch_accuracy', val_acc.avg, global_step= epoch)
            self.writer.add_scalar('val_epoch_mIoU', miou, global_step= epoch)
            print('[Epoch %d/%d] VAL loss/acc: %.3f/%.3f           mIoU: %.3f' %(epoch+1, self.epochs, val_loss.avg, val_acc.avg, miou))
        return miou
    def _eval_step(self, imgs, targets): 
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)
        loss = self.loss_func(outputs, targets)


        _, preds = torch.max(outputs, 1)
        target_mask = targets >0
        acc = torch.mean((preds == targets)[target_mask].float())
        self.metric.add(preds.detach(), targets.detach())
        
        return loss.item(), acc.item(), preds.cpu().detach()
class AverageMeter(object): 
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()
    def reset(self): 
        self.val = 0
        self.avg = 0
        self.sum = 0 
        self.count = 0
    def update(self, val, n=1): 
        self.val = val
        self.sum += val*n
        self.count +=n
        self.avg = self.sum/self.count