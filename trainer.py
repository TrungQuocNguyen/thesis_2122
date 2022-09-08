from base import BaseTrainer
import os
import math
import numpy as np 
import torch
import torch.nn as nn
from utils.helpers import plot_preds
class Trainer3DReconstruction(BaseTrainer): 
    def __init__(self, cfg, model, loss, train_loader, val_loader, projector, optimizer, device, metric_3d,  **kwargs): 
        super(Trainer3DReconstruction, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, device)
        self.metric_3d = metric_3d
        self.proxy_loss = cfg["model_2d"]["proxy_loss"]
        self.plot_gradient = cfg["trainer"]["plot_gradient"]
        if cfg["use_2d_feat_input"]:
            self.checkpoint_2d_model = os.path.join(self.model_folder, 'checkpoint_2d_model.pth.tar')
            self.best_2d_model_path = os.path.join(self.model_folder, 'model_2d_best.pth.tar')
            self.checkpoint_2d_optimizer = os.path.join(self.model_folder, 'checkpoint_2d_optimizer.pth.tar')
            self.best_2d_optimizer = os.path.join(self.model_folder, 'optimizer_2d_best.pth.tar')
            print("Using 2D features from ENet as input")
            self.model_2d_fixed = kwargs["model_2d_fixed"]
            self.model_2d_trainable = kwargs["model_2d_trainable"]
            self.model_2d_classification = kwargs["model_2d_classification"]
            self.optimizer2d = kwargs["optimizer2d"]
            if self.proxy_loss: 
                print("Using proxy loss for 2D features")
                self.eta = kwargs["eta"]
                self.criterion2d = kwargs["criterion2d"]
                self.add_figure_tensorboard = cfg["trainer"]["add_figure_tensorboard"]
                if self.add_figure_tensorboard: 
                    self.mean = torch.tensor(self.cfg["color_mean"]).reshape(1,3,1,1)
                    self.std = torch.tensor(self.cfg["color_std"]).reshape(1,3,1,1)
            if self.resume_training: 
                load_path = 'optimizer_2d_best.pth.tar' if 'best' in cfg["trainer"]["load_path"] else 'checkpoint_2d_optimizer.pth.tar'
                self.optimizer2d.load_state_dict(torch.load(os.path.join(os.path.dirname(cfg["trainer"]["load_path"]), load_path))['optimizer'])
        self.num_images = self.cfg["num_images"]
        self.accum_step = self.cfg["trainer"]["accumulation_step"]
        self.val_check_interval = cfg["trainer"]["val_check_interval"]
        self.projector = projector
        self.best_loss = np.inf
        if self.resume_training:
            self.best_loss = self.checkpoint["best_loss"]
    def train(self): 
        if self.resume_training: 
            self.count = self.checkpoint["count"]
            self.count_val = self.checkpoint["count_val"]
        else: 
            self.count = 0
            self.count_val = 0
        for epoch in range(self.start_epoch, self.epochs):
            self._train_epoch(epoch)

    def _train_epoch(self, epoch): 
        train_loss = AverageMeter()
        loss_sum = 0.0
        if self.proxy_loss: 
            train_loss2d = AverageMeter()
            loss2d_sum = 0.0

        val_iterator = iter(self.val_loader)

        self.optimizer.zero_grad()
        if self.cfg["use_2d_feat_input"]:
            self.optimizer2d.zero_grad()
        count_jump_flag_train = 0
        for batch_idx, batch in enumerate(self.train_loader, 0):
            blobs = batch.data
            self.model.train()
            if self.cfg['use_2d_feat_input']: 
                self.model_2d_trainable.train()
                if self.proxy_loss: 
                    self.model_2d_classification.train()
            #blobs['data']: [N, 32, 32, 64] N: batch_size
            batch_size = blobs['data'].shape[0]
            jump_flag = self._voxel_pixel_association(blobs)
            if jump_flag:
                print('error in train batch, skipping the current batch...') 
                count_jump_flag_train +=1
                continue
            loss, loss2d, tensorboard_preds= self._train_step(blobs, batch_idx - count_jump_flag_train)
            loss_sum += loss
            if self.proxy_loss: 
                loss2d_sum += loss2d
            if (batch_idx - count_jump_flag_train +1) % self.accum_step == 0: 
                train_loss.update(loss_sum, batch_size*self.accum_step)
                loss_sum = 0.0
                if self.proxy_loss: 
                    train_loss2d.update(loss2d_sum, self.num_images*batch_size*self.accum_step)
                    loss2d_sum =0.0
                self.count +=1
                
            if self.log_nth and (batch_idx  -count_jump_flag_train +1) % (self.log_nth * self.accum_step)== 0 : 
                if self.single_sample: 
                    self.writer.add_scalar('train_loss', train_loss.val, global_step= self.count)
                    if self.proxy_loss: 
                        self.writer.add_scalar('train_loss2d', train_loss2d.val, global_step= self.count)
                print('[Iteration %d] TRAIN loss: %.3f(%.3f)' %(self.count, train_loss.val, train_loss.avg))
                if self.proxy_loss: 
                    print('[Iteration %d] TRAIN loss2d: %.3f(%.3f)' %(self.count, train_loss2d.val, train_loss2d.avg))
                    eta0 = self.eta[0].item()
                    eta1 = self.eta[1].item()
                    self.writer.add_scalar('eta_0', eta0, global_step= self.count)
                    self.writer.add_scalar('eta_1', eta1, global_step= self.count)
                    if self.add_figure_tensorboard: 
                        self.writer.add_figure('train predictions vs targets', plot_preds(blobs['nearest_images']['images'][0]*self.std+self.mean, blobs['nearest_images']['label_images'][0], tensorboard_preds), global_step = self.count)
                if not self.single_sample: 
                    self.model.eval()
                    if self.cfg['use_2d_feat_input']: 
                        self.model_2d_trainable.eval()
                        if self.proxy_loss: 
                            self.model_2d_classification.eval()
                    val_loss = 0.0
                    if self.proxy_loss: 
                        val_loss2d = 0.0
                    j = 0
                    self.metric_3d.reset()
                    with torch.no_grad(): 
                        while j < self.accum_step:  
                            try: 
                                blobs = next(val_iterator).data
                            except StopIteration: 
                                val_iterator = iter(self.val_loader)
                                blobs = next(val_iterator).data
                            jump_flag = self._voxel_pixel_association(blobs)
                            if jump_flag: 
                                print('error in single validation batch, skipping the current batch...')
                                continue
                            temp1, temp2, tensorboard_preds= self._eval_step(blobs, True)
                            val_loss += temp1
                            if self.proxy_loss: 
                                val_loss2d += temp2
                            j = j+1
                        iou3d, _ = self.metric_3d.value()
                    #self.writer.add_scalar('val_loss', val_loss, global_step= len(self.train_loader)*epoch + i)
                    self.writer.add_scalar('step_IoU', iou3d[1], global_step= self.count)
                    self.writer.add_scalars('step_loss', {'train_loss': train_loss.val, 'val_loss': val_loss}, global_step = self.count)
                    print('[Iteration %d] VAL loss: %.3f' %(self.count, val_loss))
                    if self.proxy_loss: 
                        self.writer.add_scalars('step_loss2d', {'train_loss': train_loss2d.val, 'val_loss': val_loss2d}, global_step = self.count)
                        self.writer.add_scalars('step_final_loss', {'train_loss': math.exp(-eta0)*train_loss.val + math.exp(-eta1)*train_loss2d.val + (eta0 + eta1)/2, 'val_loss': math.exp(-eta0)*val_loss + math.exp(-eta1)*val_loss2d + (eta0 + eta1)/2}, global_step = self.count)
                        print('[Iteration %d] VAL loss2d: %.3f' %(self.count, val_loss2d))
                        if self.add_figure_tensorboard: 
                            self.writer.add_figure('val predictions vs targets', plot_preds(blobs['nearest_images']['images'][0]*self.std+self.mean, blobs['nearest_images']['label_images'][0], tensorboard_preds), global_step = self.count)

            if self.val_check_interval and (batch_idx - count_jump_flag_train+1) % (self.val_check_interval*self.accum_step) == 0: 
                if not self.single_sample: 
                    self.count_val +=1
                    if self.proxy_loss: 
                        loss, loss2d = self._val_epoch(self.count_val)  #comment this when overfitting with 10 train and 4 val
                    else: 
                        loss = self._val_epoch(self.count_val)
                is_best = loss < self.best_loss
                self.best_loss = min(loss, self.best_loss)
                
                self.save_checkpoint({
                    'epoch': epoch, 
                    'count': self.count, 
                    'count_val': self.count_val,
                    'state_dict': self.model.state_dict(), 
                    'best_loss': self.best_loss, 
                    'optimizer': self.optimizer.state_dict()
                }, is_best, self.checkpoint_path, self.best_model_path)
                if self.cfg['use_2d_feat_input']: 
                    self.save_checkpoint({
                        'state_dict': nn.Sequential(*(list(self.model_2d_fixed) + list(self.model_2d_trainable) + list(self.model_2d_classification))).state_dict()
                    }, is_best, self.checkpoint_2d_model, self.best_2d_model_path)

                    self.save_checkpoint({
                    'optimizer': self.optimizer2d.state_dict()
                }, is_best, self.checkpoint_2d_optimizer, self.best_2d_optimizer)


    def _train_step(self, blobs, batch_idx): 
        targets = blobs['data'].long().to(self.device) # [N, 32, 32, 64] 
        batch_size = targets.shape[0]
        rgb_images = []
        target_images = []
        loss2d = torch.zeros(1)
        tensorboard_preds = 0.0
        if self.cfg['use_2d_feat_input']: 
            for i in range(batch_size):
                rgb_images.append(blobs['nearest_images']['images'][i])
                if self.proxy_loss: 
                    target_images.append(blobs['nearest_images']['label_images'][i])
                    
            rgb_images = torch.cat(rgb_images).to(self.device) # [max_num_images*batch_size,3, 256, 328]
            imageft = self.model_2d_fixed(rgb_images) 
            imageft = self.model_2d_trainable(imageft) # [max_num_images*batch_size,128, 32, 41]
            blobs['feat_2d'] = imageft
            if self.proxy_loss: 
                predicted_images = self.model_2d_classification(imageft) # [max_num_images*batch_size,41, 32, 41]
                if self.add_figure_tensorboard: 
                    mask = predicted_images[0:5]
                    _, tensorboard_preds = torch.max(mask, 1)
                    tensorboard_preds = tensorboard_preds.cpu().detach()
        preds = self.model(blobs, self.device) #[N, 2, 32, 32, 64]
        loss = self.criterion(preds, targets)/self.accum_step

        if not self.proxy_loss: 
            loss.backward()
        else: 
            target_images = torch.cat(target_images).to(self.device)# [max_num_images*batch_size, 32, 41]
            loss2d = self.criterion2d(predicted_images, target_images)/self.accum_step
            final_loss = torch.exp(-self.eta)[0]*loss + torch.exp(-self.eta)[1]*loss2d + self.eta.sum()/(2*self.accum_step)
            final_loss.backward()

        if (batch_idx+1) % self.accum_step == 0: 
            if self.plot_gradient: 
                self.plot_grad_flow('model_3d', self.model.named_parameters(), self.count+1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.cfg['use_2d_feat_input']: 
                if self.plot_gradient: 
                    self.plot_grad_flow('model_2d_trainable', self.model_2d_trainable.named_parameters(), self.count+1)
                    if self.proxy_loss: 
                        self.plot_grad_flow('model_2d_classification', self.model_2d_classification.named_parameters(), self.count+1)
                self.optimizer2d.step()
                self.optimizer2d.zero_grad() # optimizer call with and without proxy loss is the same 
        return loss.item(), loss2d.item(), tensorboard_preds

    def _val_epoch(self, epoch): 
        loss = 0.0
        val_loss = AverageMeter()
        if self.proxy_loss: 
            loss2d = 0.0
            val_loss2d = AverageMeter()
        self.model.eval()
        if self.cfg['use_2d_feat_input']: 
            self.model_2d_trainable.eval()
            if self.proxy_loss: 
                self.model_2d_classification.eval()
        self.metric_3d.reset()
        count_jump_flag_val = 0
        with torch.no_grad(): 
            for i,sample in enumerate(self.val_loader):
                blobs = sample.data
                batch_size = blobs['data'].shape[0] 
                jump_flag = self._voxel_pixel_association(blobs)
                if jump_flag:
                    print('error in validation batch, skipping the current batch...')
                    count_jump_flag_val +=1
                    continue
                temp1, temp2, _= self._eval_step(blobs, True)
                loss+= temp1
                if self.proxy_loss: 
                    loss2d += temp2
                if (i - count_jump_flag_val +1) % self.accum_step == 0: 
                    val_loss.update(loss, batch_size*self.accum_step)
                    loss = 0.0
                    if self.proxy_loss: 
                        val_loss2d.update(loss2d, self.num_images*batch_size*self.accum_step)
                        loss2d = 0.0
        iou, _ = self.metric_3d.value()
        if self.log_nth:  
            self.writer.add_scalar('val_epoch_loss', val_loss.avg, global_step= epoch)
            print('[VAL epoch %d] VAL loss: %.3f' %(epoch, val_loss.avg))
            if self.proxy_loss: 
                self.writer.add_scalar('val_epoch_loss2d', val_loss2d.avg, global_step= epoch)
                print('[VAL epoch %d] VAL loss2d: %.3f' %(epoch, val_loss2d.avg))
            self.writer.add_scalar('val_epoch_IoU', iou[1], global_step= epoch)
        if self.proxy_loss: 
            return val_loss.avg, val_loss2d.avg
        else: 
            return val_loss.avg

    def _eval_step(self, blobs, is_validating): 
        targets = blobs['data'].long().to(self.device) # [N, 32, 32, 64]
        batch_size = targets.shape[0]
        rgb_images = []
        target_images = []
        loss2d = torch.zeros(1)
        tensorboard_preds = 0.0
        if self.cfg['use_2d_feat_input']: 
            for i in range(batch_size):
                rgb_images.append(blobs['nearest_images']['images'][i])
                if self.proxy_loss: 
                    target_images.append(blobs['nearest_images']['label_images'][i])
            rgb_images = torch.cat(rgb_images).to(self.device) # [max_num_images*batch_size,3, 256, 328]
            imageft = self.model_2d_fixed(rgb_images) 
            imageft = self.model_2d_trainable(imageft) # [max_num_images*batch_size,128, 32, 41]
            blobs['feat_2d'] = imageft
            if self.proxy_loss: 
                predicted_images = self.model_2d_classification(imageft) # [max_num_images*batch_size,41, 32, 41]
                if self.add_figure_tensorboard: 
                    mask = predicted_images[0:5]
                    _, tensorboard_preds = torch.max(mask, 1)
                    tensorboard_preds = tensorboard_preds.cpu().detach()

        preds = self.model(blobs, self.device) #[N, 2, 32, 32, 64]
        loss = self.criterion(preds, targets)/self.accum_step
        if self.proxy_loss: 
            target_images = torch.cat(target_images).to(self.device)# [max_num_images*batch_size, 32, 41]
            loss2d = self.criterion2d(predicted_images, target_images)/self.accum_step
        if is_validating: 
            _, preds = torch.max(preds, 1) # preds: [N, 32, 32, 64], cuda
            targets[targets == -100] = 2 #target: [N, 32, 32, 64], cuda
            self.metric_3d.add(preds.detach(), targets.detach())
            if self.proxy_loss: 
                _, predicted_images = torch.max(predicted_images, 1)
                predicted_images = predicted_images.detach()
                target_images = target_images.detach()
        return loss.item(), loss2d.item(), tensorboard_preds
    def _voxel_pixel_association(self, blobs): 
        batch_size = blobs['data'].shape[0]
        proj_mapping = [[self.projector.compute_projection(d.to(self.device), c.to(self.device), t.to(self.device)) for d, c, t in zip(blobs['nearest_images']['depths'][i], blobs['nearest_images']['poses'][i], blobs['nearest_images']['world2grid'][i])] for i in range(batch_size)]
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
                blobs['proj_ind_3d'].append(torch.stack(proj_mapping0)) # list of [max_num_images, 32*32*64+ 1], total batch_size elements in the list 
                blobs['proj_ind_2d'].append(torch.stack(proj_mapping1)) # list of [max_num_images, 32*32*64 + 1], total batch_size elements in the list      
        return jump_flag


class TrainerENet(BaseTrainer): 
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, metric, metric_all_classes, device):
        super(TrainerENet, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, device)
        self.add_figure_tensorboard = cfg["trainer"]["add_figure_tensorboard"]
        self.metric = metric
        self.metric_all_classes = metric_all_classes
        self.best_miou = 0
        if self.resume_training:
            self.best_miou = self.checkpoint["best_miou"]
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
                }, is_best, self.checkpoint_path, self.best_model_path)
    def _train_epoch(self, epoch):
        train_loss = AverageMeter()
        val_iterator = iter(self.val_loader)
        for i, (imgs, targets) in enumerate(self.train_loader, 0): # imgs: [N, C, H, W] (N, 3, 256, 328), targets: [N, H, W]
            self.model.train()
            loss, preds = self._train_step(imgs, targets)
            train_loss.update(loss, imgs.size(0))
            if self.log_nth and i % self.log_nth == self.log_nth-1: 
                mean = torch.tensor(self.train_loader.dataset.mean).reshape(1,3,1,1)
                std = torch.tensor(self.train_loader.dataset.std).reshape(1,3,1,1)
                self.writer.add_scalar('train_loss', train_loss.val, global_step= len(self.train_loader)*epoch + i )
                print('[Iteration %d/%d] TRAIN loss: %.3f(%.3f) ' %(len(self.train_loader)*epoch + i+1, len(self.train_loader)*self.epochs, train_loss.val, train_loss.avg))
                if self.add_figure_tensorboard: 
                    self.writer.add_figure('train predictions vs targets', plot_preds(imgs*std+mean, targets, preds), global_step = len(self.train_loader)*epoch + i)
                if not self.single_sample: 
                    self.model.eval()
                    self.metric.reset()
                    self.metric_all_classes.reset()
                    with torch.no_grad():
                        try: 
                            imgs, targets = next(val_iterator)
                        except StopIteration: 
                            val_iterator = iter(self.val_loader)
                            imgs, targets = next(val_iterator)
                        
                        val_loss, val_preds = self._eval_step(imgs, targets, True)
                    _, miou = self.metric.value()
                    _, miou_all_classes = self.metric_all_classes.value()
                    self.writer.add_scalar('val_mIoU', miou, global_step= len(self.train_loader)*epoch + i )
                    self.writer.add_scalar('val_mIoU_all_classes', miou_all_classes, global_step= len(self.train_loader)*epoch + i )
                    self.writer.add_scalar('val_loss', val_loss, global_step= len(self.train_loader)*epoch + i)
                    print('[Iteration %d/%d] VAL loss: %.3f   ' %(len(self.train_loader)*epoch + i+1, len(self.train_loader)*self.epochs, val_loss))
                    if self.add_figure_tensorboard: 
                        self.writer.add_figure('val predictions vs targets', plot_preds(imgs*std+mean, targets, val_preds), global_step = len(self.train_loader)*epoch + i)
        if self.log_nth and not self.single_sample: 
            print('[Epoch %d/%d] TRAIN loss: %.3f' %(epoch+1, self.epochs, train_loss.avg))

    def _train_step(self, imgs, targets): 
        imgs = imgs.to(self.device) # (N, 3, H,W)
        targets = targets.to(self.device) # (N, H, W)
        self.optimizer.zero_grad()
        outputs = self.model(imgs) # (N, C, H, W)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        _, preds = torch.max(outputs, 1) # [N, H, W]   
        preds = preds.cpu().detach()    
        return loss.item(), preds

    def _val_epoch(self, epoch): 
        val_loss = AverageMeter()
        self.model.eval()
        self.metric.reset()
        self.metric_all_classes.reset()
        with torch.no_grad(): 
            for (imgs, targets) in self.val_loader: 
               loss, _= self._eval_step(imgs, targets, True)
               val_loss.update(loss, imgs.size(0))
        _, miou = self.metric.value()
        _, miou_all_classes = self.metric_all_classes.value()
        if self.log_nth: 
            self.writer.add_scalar('val_epoch_loss', val_loss.avg, global_step= epoch)
            self.writer.add_scalar('val_epoch_mIoU', miou, global_step= epoch)
            self.writer.add_scalar('val_epoch_mIoU_all_classes', miou_all_classes, global_step= epoch)
            print('[Epoch %d/%d] VAL loss: %.3f           mIoU: %.3f           mIoU_all_classes: %.3f' %(epoch+1, self.epochs, val_loss.avg, miou, miou_all_classes))
        return miou
    def _eval_step(self, imgs, targets, is_validating): 
        imgs = imgs.to(self.device)
        targets = targets.to(self.device) # [N, H, W]

        outputs = self.model(imgs) # (N, C, H, W)
        loss = self.criterion(outputs, targets)


        _, preds = torch.max(outputs, 1) # [N, H, W]
        if is_validating: 
            pred_ = preds.detach()
            target_ = targets.detach()
            self.metric.add(pred_, target_)
            self.metric_all_classes.add(pred_, target_)
        
        return loss.item(), preds.cpu().detach()
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