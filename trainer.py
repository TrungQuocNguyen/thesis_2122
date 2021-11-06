from base import BaseTrainer
import numpy as np 
import torch
import matplotlib.pyplot as plt
class Trainer(BaseTrainer): 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, device):
        super(Trainer, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, device)

    def _train_epoch(self, epoch):
        train_loss_epoch = []
        train_acc_epoch = []
        self.model.train()
        for i, (imgs, targets) in enumerate(self.train_loader, 0): 
            loss, acc, preds = self._train_step(imgs, targets)
            if self.log_nth and i % self.log_nth == self.log_nth-1: 
                train_loss = np.mean(self.train_loss_history[-self.log_nth:])
                train_acc = np.mean(self.train_acc_history[-self.log_nth:])
                self.writer.add_scalar('train_loss', train_loss, global_step= len(self.train_loader)*epoch + i )
                self.writer.add_scalar('train_accuracy', train_acc, global_step= len(self.train_loader)*epoch + i )
                print('[Iteration %d/%d] TRAIN loss: %.3f   TRAIN accuracy: %.3f' %(len(self.train_loader)*epoch + i, len(self.train_loader)*self.epochs-1, train_loss, train_acc))
                if self.add_figure_tensorboard: 
                    self.writer.add_figure('train predictions vs targets', plot_preds(imgs, targets, preds), global_step = len(self.train_loader)*epoch + i)
                if not self.single_sample: 
                    self.model.eval()
                    with torch.no_grad(): 
                        imgs, targets = next(iter(self.val_loader))
                        val_loss, val_acc, val_preds = self._eval_step(imgs, targets)
                    self.writer.add_scalar('val_loss', val_loss, global_step= len(self.train_loader)*epoch + i)
                    self.writer.add_scalar('val_accuracy', val_acc, global_step= len(self.train_loader)*epoch + i)
                    print('[Iteration %d/%d] VAL loss: %.3f   VAL accuracy: %.3f' %(len(self.train_loader)*epoch + i, len(self.train_loader)*self.epochs-1, val_loss, val_acc))
                    if self.add_figure_tensorboard: 
                        self.writer.add_figure('val predictions vs targets', plot_preds(imgs, targets, val_preds), global_step = len(self.train_loader)*epoch + i)
            train_loss_epoch.append(loss)
            train_acc_epoch.append(acc)
        if self.log_nth and not self.single_sample: 
            print('[Epoch %d/%d] TRAIN loss/acc: %.3f/%.3f' %(epoch, self.epochs-1, np.mean(train_loss_epoch), np.mean(train_acc_epoch)))

    def _train_step(self, imgs, targets): 
        imgs = imgs.to(self.device) # (N, 3, img_size, img_size)
        targets = targets.to(self.device) # (N, img_size, img_size)
        self.optimizer.zero_grad()
        outputs = self.model(imgs) # (N, C, img_size, img_size)
        loss = self.loss_func(outputs, targets)
        loss.backward()
        self.optimizer.step()

        loss = loss.cpu().detach().numpy()
        self.train_loss_history.append(loss)

        _, preds = torch.max(outputs, 1) # [N, img_size, img_size]
        target_mask = targets >0
        acc = np.mean((preds == targets)[target_mask].cpu().detach().numpy())
        self.train_acc_history.append(acc)      
        preds = preds.cpu().detach()    
        return loss, acc, preds

    def _val_epoch(self, epoch): 
        val_losses = []
        val_accs = []
        self.model.eval()
        with torch.no_grad(): 
            for (imgs, targets) in self.val_loader: 
               loss, acc, _= self._eval_step(imgs, targets)
               val_losses.append(loss)
               val_accs.append(acc)

        val_loss, val_acc = np.mean(val_losses), np.mean(val_accs)
        if self.log_nth: 
            self.writer.add_scalar('val_epoch_loss', val_loss, global_step= epoch)
            self.writer.add_scalar('val_epoch_accuracy', val_acc, global_step= epoch)
            print('[Epoch %d/%d] VAL loss/acc: %.3f/%.3f' %(epoch, self.epochs-1, val_loss, val_acc))

    def _eval_step(self, imgs, targets): 
        imgs = imgs.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(imgs)
        loss = self.loss_func(outputs, targets)

        loss = loss.cpu().detach().numpy()

        _, preds = torch.max(outputs, 1)
        target_mask = targets >0
        acc = np.mean((preds == targets)[target_mask].cpu().detach().numpy())
        preds = preds.cpu().detach()
        
        return loss, acc, preds
def plot_preds(imgs, targets, preds): 
    #imgs: [N, 3, img_size, img_size]
    #targets: [N, img_size, img_size]
    #preds: [N, img_size, img_size]
    N = imgs.size(0)
    num_img_show = N if N <= 4 else 4
    imgs = imgs.permute(0,2,3,1).numpy()
    targets = targets.numpy()
    preds = preds.numpy()
    fig = plt.figure(figsize=(14,8))
    for idx in range(num_img_show): 
        fig.add_subplot(3,num_img_show,idx +1)
        plt.imshow(imgs[idx])
        fig.add_subplot(3,num_img_show,idx +1 + num_img_show)
        plt.imshow(preds[idx])
        fig.add_subplot(3,num_img_show,idx +1 + num_img_show*2)
        plt.imshow(targets[idx])
    return fig