from base import BaseTrainer
import os
import numpy as np 
import torch
import torch.nn as nn
from utils.helpers import plot_preds
class Trainer3DReconstruction(BaseTrainer): 
    def __init__(self, cfg, model, loss, train_loader, val_loader, projector, optimizer, device, metric_3d,  **kwargs): 
        super(Trainer3DReconstruction, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, device, metric_3d)
        if cfg["use_2d_feat_input"]:

            self.checkpoint_2d_model = os.path.join(self.model_folder, 'checkpoint_2d_model.pth.tar')
            self.best_2d_model_path = os.path.join(self.model_folder, 'model_2d_best.pth.tar')
            self.checkpoint_2d_optimizer = os.path.join(self.model_folder, 'checkpoint_2d_optimizer.pth.tar')
            self.best_2d_optimizer = os.path.join(self.model_folder, 'optimizer_2d_best.pth.tar')

            print("Using 2D features from ENet as input")
            self.model_2d = kwargs["model_2d"]
            self.optimizer2d = kwargs["optimizer2d"]
            self.proxy_loss = cfg["model_2d"]["proxy_loss"]
            if self.proxy_loss: 
                print("Using proxy loss for 2D features")
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
        for epoch in range(self.start_epoch, self.epochs):
            loss = self._train_epoch(epoch)
            #if epoch % 10 == 9: 
            #    if self.cfg["trainer"]["plot_gradient"]: 
            #        self.plot_grad_flow(self.model.named_parameters(), epoch)

    def _train_epoch(self, epoch): 
        train_loss = AverageMeter()
        loss = 0.0
        if self.proxy_loss: 
            train_loss2d = AverageMeter()
            loss2d = 0.0

        val_iterator = iter(self.val_loader)

        self.optimizer.zero_grad()
        if self.cfg["use_2d_feat_input"]:
            self.optimizer2d.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader, 0):
            blobs = batch.data
            self.model.train()
            if self.cfg['use_2d_feat_input']: 
                for layer_idx, layer in enumerate(self.model_2d.children()): 
                    if layer_idx >=15: 
                        layer.train()
            #blobs['data']: [N, 32, 32, 64] N: batch_size
            batch_size = blobs['data'].shape[0]
            jump_flag = self._voxel_pixel_association(blobs)
            if jump_flag:
                print('error in train batch, skipping the current batch...') 
                continue
            temp1, temp2, tensorboard_preds= self._train_step(blobs, batch_idx)
            loss += temp1
            if self.proxy_loss: 
                loss2d += temp2
            if (batch_idx+1) % self.accum_step == 0: 
                train_loss.update(loss, batch_size*self.accum_step)
                loss = 0.0
                if self.proxy_loss: 
                    train_loss2d.update(loss2d, self.num_images*batch_size*self.accum_step)
                    loss2d =0.0
                
            if self.log_nth and (batch_idx+1) % (self.log_nth * self.accum_step)== 0 : 
                if self.single_sample: 
                    self.writer.add_scalar('train_loss', train_loss.val, global_step= len(self.train_loader)*epoch + batch_idx )
                print('[Iteration %d/%d] TRAIN loss: %.3f(%.3f)' %(len(self.train_loader)*epoch + batch_idx+1, len(self.train_loader)*self.epochs, train_loss.val, train_loss.avg))
                if self.proxy_loss: 
                    print('[Iteration %d/%d] TRAIN loss2d: %.3f(%.3f)' %(len(self.train_loader)*epoch + batch_idx+1, len(self.train_loader)*self.epochs, train_loss2d.val, train_loss2d.avg))
                    if self.add_figure_tensorboard: 
                        self.writer.add_figure('train predictions vs targets', plot_preds(blobs['nearest_images']['images'][0]*self.std+self.mean, blobs['nearest_images']['label_images'][0], tensorboard_preds), global_step = len(self.train_loader)*epoch + batch_idx)
                if not self.single_sample: 
                    self.model.eval()
                    if self.cfg['use_2d_feat_input']: 
                        self.model_2d.eval()
                    with torch.no_grad(): 
                        val_loss = 0.0
                        if self.proxy_loss: 
                            val_loss2d = 0.0
                        for j in range(self.accum_step): 
                            try: 
                                blobs = next(val_iterator).data
                            except StopIteration: 
                                val_iterator = iter(self.val_loader)
                                blobs = next(val_iterator).data
                            jump_flag = self._voxel_pixel_association(blobs)
                            if jump_flag: 
                                print('error in single validation batch, skipping the current batch...')
                                continue
                            temp1, temp2, tensorboard_preds= self._eval_step(blobs, False)
                            val_loss += temp1
                            if self.proxy_loss: 
                                val_loss2d += temp2
                    #self.writer.add_scalar('val_loss', val_loss, global_step= len(self.train_loader)*epoch + i)
                    self.writer.add_scalars('step_loss', {'train_loss': train_loss.val, 'val_loss': val_loss}, global_step = len(self.train_loader)*epoch + batch_idx)
                    print('[Iteration %d/%d] VAL loss: %.3f' %(len(self.train_loader)*epoch + batch_idx+1, len(self.train_loader)*self.epochs, val_loss))
                    if self.proxy_loss: 
                        self.writer.add_scalars('step_loss2d', {'train_loss': train_loss2d.val, 'val_loss': val_loss2d}, global_step = len(self.train_loader)*epoch + batch_idx)
                        print('[Iteration %d/%d] VAL loss2d: %.3f' %(len(self.train_loader)*epoch + batch_idx+1, len(self.train_loader)*self.epochs, val_loss2d))
                        if self.add_figure_tensorboard: 
                            self.writer.add_figure('val predictions vs targets', plot_preds(blobs['nearest_images']['images'][0]*self.std+self.mean, blobs['nearest_images']['label_images'][0], tensorboard_preds), global_step = len(self.train_loader)*epoch + batch_idx)

            if self.val_check_interval and (batch_idx+1) % (self.val_check_interval*self.accum_step) == 0: 
                if not self.single_sample: 
                    num_val_epoch = epoch*(len(self.train_loader)// self.val_check_interval) + batch_idx//self.val_check_interval +1
                    if self.proxy_loss: 
                        loss, loss2d = self._val_epoch(num_val_epoch)  #comment this when overfitting with 10 train and 4 val
                    else: 
                        loss = self._val_epoch(num_val_epoch)
                is_best = loss < self.best_loss
                self.best_loss = min(loss, self.best_loss)
                
                self.save_checkpoint({
                    'epoch': epoch, 
                    'state_dict': self.model.state_dict(), 
                    'best_loss': self.best_loss, 
                    'optimizer': self.optimizer.state_dict()
                }, is_best, self.checkpoint_path, self.best_model_path)
                if self.cfg['use_2d_feat_input']: 
                    self.save_checkpoint({
                        'state_dict': self.model_2d.state_dict()
                    }, is_best, self.checkpoint_2d_model, self.best_2d_model_path)

                    self.save_checkpoint({
                    'optimizer': self.optimizer2d.state_dict()
                }, is_best, self.checkpoint_2d_optimizer, self.best_2d_optimizer)
                
                val_epoch_loss = loss 
                loss = 0.0
                if self.proxy_loss: 
                    val_epoch_loss2d = loss2d 
                    loss2d = 0.0
        # comment this block when overfitting with 10 train and 4 val    
        if self.log_nth and not self.single_sample: 
            print('[Epoch %d/%d] TRAIN loss: %.3f' %(epoch+1, self.epochs, train_loss.avg))
            self.writer.add_scalars('epoch_loss', {'train_loss': train_loss.avg, 'val_loss': val_epoch_loss }, global_step = epoch)
            if self.proxy_loss: 
                print('[Epoch %d/%d] TRAIN loss2d: %.3f' %(epoch+1, self.epochs, train_loss2d.avg))
                self.writer.add_scalars('epoch_loss2d', {'train_loss': train_loss2d.avg, 'val_loss': val_epoch_loss2d }, global_step = epoch)
        return train_loss.avg


    def _train_step(self, blobs, batch_idx): 
        targets = blobs['data'].long().to(self.device) # [N, 32, 32, 64] 
        batch_size = targets.shape[0]
        predicted_images = []
        target_images = []
        loss2d = torch.zeros(1)
        tensorboard_preds = 0.0
        if self.cfg['use_2d_feat_input']: 
            blobs['feat_2d'] =  []
            for i in range(batch_size):
                imageft, mask = self.model_2d(blobs['nearest_images']['images'][i].to(self.device)) # feat: [max_num_images, 128, 32, 41], mask: [max_num_images, 41, 256, 328]
                blobs['feat_2d'].append(imageft) # list of tensor, each tensor size [max_num_images, 128, depth_shape[1], depth_shape[0]]
                if self.proxy_loss: 
                    predicted_images.append(mask)
                    target_images.append(blobs['nearest_images']['label_images'][i])
                    if i == 0 and self.add_figure_tensorboard: # take only the first sample in the batch for visualization
                        _, tensorboard_preds = torch.max(mask, 1)
                        tensorboard_preds = tensorboard_preds.cpu().detach()
        preds = self.model(blobs, self.device) #[N, 2, 32, 32, 64]
        loss = self.criterion(preds, targets)/self.accum_step

        if not self.proxy_loss: 
            loss.backward()
        else: 
            predicted_images = torch.cat(predicted_images) # [max_num_images*batch_size, 41, 256, 328]
            target_images = torch.cat(target_images).to(self.device) # [max_num_images*batch_size, 256, 328]
            loss2d = self.criterion2d(predicted_images, target_images)/self.accum_step
            (loss + loss2d).backward()

        if (batch_idx+1) % self.accum_step == 0: 
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.cfg['use_2d_feat_input']: 
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
            self.model_2d.eval()
        self.metric_3d.reset()
        with torch.no_grad(): 
            for i,sample in enumerate(self.val_loader):
                blobs = sample.data
                batch_size = blobs['data'].shape[0] 
                jump_flag = self._voxel_pixel_association(blobs)
                if jump_flag:
                    print('error in validation batch, skipping the current batch...')
                    continue
                temp1, temp2, _= self._eval_step(blobs, True)
                loss+= temp1
                if self.proxy_loss: 
                    loss2d += temp2
                if (i+1) % self.accum_step == 0: 
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
        predicted_images = []
        target_images = []
        loss2d = torch.zeros(1)
        tensorboard_preds = 0.0
        if self.cfg['use_2d_feat_input']: 
            blobs['feat_2d'] =  []
            for i in range(batch_size):
                imageft, mask = self.model_2d(blobs['nearest_images']['images'][i].to(self.device)) # feat: [max_num_images, 128, 32, 41], mask: [max_num_images, 41, 256, 328]
                blobs['feat_2d'].append(imageft) # list of tensor, each tensor size [max_num_images, 128, depth_shape[1], depth_shape[0]]
                if self.proxy_loss: 
                    predicted_images.append(mask)
                    target_images.append(blobs['nearest_images']['label_images'][i])
                    if i == 0 and self.add_figure_tensorboard: #take only the first sample in the batch for visualization
                        _, tensorboard_preds = torch.max(mask, 1)
                        tensorboard_preds = tensorboard_preds.cpu().detach()
        preds = self.model(blobs, self.device) #[N, 2, 32, 32, 64]
        loss = self.criterion(preds, targets)/self.accum_step
        if self.proxy_loss: 
            predicted_images = torch.cat(predicted_images) # [max_num_images*batch_size, 41, 256, 328]
            target_images = torch.cat(target_images).to(self.device) # [max_num_images*batch_size, 256, 328]
            loss2d = self.criterion2d(predicted_images, target_images)/self.accum_step
        if is_validating: 
            preds = torch.max(preds, 1) # preds: [N, 32, 32, 64], cuda
            targets[targets == -100] = 2 #target: [N, 32, 32, 64], cuda
            self.metric_3d.add(preds.detach(), targets.detach())
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
    def __init__(self, cfg, model, loss, train_loader, val_loader, optimizer, metric, device):
        super(TrainerENet, self).__init__(cfg, model, loss, train_loader, val_loader, optimizer, device)
        self.add_figure_tensorboard = cfg["trainer"]["add_figure_tensorboard"]
        self.metric = metric
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
                    with torch.no_grad():
                        try: 
                            imgs, targets = next(val_iterator)
                        except StopIteration: 
                            val_iterator = iter(self.val_loader)
                            imgs, targets = next(val_iterator)
                        
                        val_loss, val_preds = self._eval_step(imgs, targets, False)
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
        _, outputs = self.model(imgs) # (N, C, H, W)
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
        with torch.no_grad(): 
            for (imgs, targets) in self.val_loader: 
               loss, _= self._eval_step(imgs, targets, True)
               val_loss.update(loss, imgs.size(0))
        iou, miou = self.metric.value()
        if self.log_nth: 
            self.writer.add_scalar('val_epoch_loss', val_loss.avg, global_step= epoch)
            self.writer.add_scalar('val_epoch_mIoU', miou, global_step= epoch)
            print('[Epoch %d/%d] VAL loss: %.3f           mIoU: %.3f' %(epoch+1, self.epochs, val_loss.avg, miou))
        return miou
    def _eval_step(self, imgs, targets, is_validating): 
        imgs = imgs.to(self.device)
        targets = targets.to(self.device) # [N, H, W]

        _, outputs = self.model(imgs) # (N, C, H, W)
        loss = self.criterion(outputs, targets)


        _, preds = torch.max(outputs, 1) # [N, H, W]
        if is_validating: 
            self.metric.add(preds.detach(), targets.detach())
        
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