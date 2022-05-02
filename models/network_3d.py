from tokenize import group
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from projection import Projection
from utils.helpers import initialize_weights
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, groups = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, groups = groups)
        self.conv3 = nn.Conv3d(planes, inplanes, kernel_size=1)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.bn3 = nn.BatchNorm3d(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out = self.bn3(out)
        out += residual
        out = self.relu(out)
        return out

class Dense3DNetwork(nn.Module):
        '''Dense 3D CNN network consisting series of ResNet blocks'''
        def __init__(self, cfg, num_images): 
                super(Dense3DNetwork, self).__init__()
                self.cfg = cfg
                self.encoder  = nn.Sequential(
                        nn.Conv3d(num_images*3, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # [N, 64, 16, 16, 32]
                        nn.BatchNorm3d(64),
                        nn.ReLU(True),
                        Bottleneck(64, 32, stride=1), # [N, 64, 16, 16, 32]
                        nn.MaxPool3d(3, 1, 1), # [N, 64, 16, 16, 32]
                        nn.Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # [N, 128, 8, 8, 16]
                        nn.BatchNorm3d(128),
                        nn.ReLU(True),
                        Bottleneck(128, 32, stride=1), # [N, 128, 8, 8, 16]
                        nn.MaxPool3d(3, 1, 1), # [N, 128, 8, 8, 16],
                        nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), # [N, 128, 8, 8, 16]
                        nn.BatchNorm3d(128),
                        nn.ReLU(True),
                        Bottleneck(128, 64, stride=1), # [N, 128, 8, 8, 16]
                        Bottleneck(128, 64, stride=1), # [N, 128, 8, 8, 16]
                        nn.MaxPool3d(3, 1, 1))  # [N, 128, 8, 8, 16]
                self.decoder = nn.Sequential(
                        nn.ConvTranspose3d(128, 128, kernel_size= (3,3,3), stride= (1,1,1), padding= (1,1,1)),  # [N, 128, 8, 8, 16]
                        nn.BatchNorm3d(128),
                        nn.ReLU(True), 
                        Bottleneck(128, 64, stride = 1),
                        nn.ConvTranspose3d(128, 64, kernel_size= (2,2,2), stride= (2,2,2), padding = (0,0,0)),  # [N, 64, 16, 16, 32]
                        nn.BatchNorm3d(64),
                        nn.ReLU(True),  
                        Bottleneck(64, 32, stride = 1),
                        nn.ConvTranspose3d(64, 32, kernel_size= (2,2,2), stride= (2,2,2), padding = (0,0,0)),  #  [N, 32, 32, 32, 64]
                        nn.BatchNorm3d(32),
                        nn.ReLU(True), 
                        Bottleneck(32, 16, stride = 1), # [N, 32, 32, 32, 64]
                        nn.Conv3d(32, 2, kernel_size= (3,3,3), stride= (1,1,1), padding= (1,1,1))) # [N, 2, 32, 32, 64]
                initialize_weights(self)
                nn.init.xavier_uniform_(self.decoder[-1].weight)

        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 32, 32, 64]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [32, 32, 64]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images,32*32*64 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images,32*32*64 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 64, 32, 32] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 64, 32, 32]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 64, 32, 32]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 32, 32, 64][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 32, 32, 64] [in order x,y,z]
                out = self.encoder(_imageft)  # [N, 128, 8, 8, 16]
                out = self.decoder(out) # [N, 2, 32, 32, 64]

                return out

class ResUNet(nn.Module): 
        def __init__(self, cfg, num_images): 
                super(ResUNet, self).__init__()
                self.cfg = cfg
                self.conv1 = nn.Sequential(
                        nn.Conv3d(num_images*3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU())
                self.bottleneck1 = nn.Sequential(
                        Bottleneck(32, 8), 
                        Bottleneck(32, 8))
                self.conv2 = nn.Sequential(
                        nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU())
                self.bottleneck2 = nn.Sequential(
                        Bottleneck(64, 16), 
                        Bottleneck(64, 16))
                self.conv3 = nn.Sequential(
                        nn.Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(128), 
                        nn.ReLU())
                self.bottleneck3 = nn.Sequential(
                        Bottleneck(128, 32), 
                        Bottleneck(128, 32))
                self.conv4 = nn.Sequential(
                        nn.Conv3d(128, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(256), 
                        nn.ReLU())
                self.bottleneck4 = nn.Sequential(
                        Bottleneck(256, 64), 
                        Bottleneck(256, 64))
                self.conv5 = nn.Sequential(
                        nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), 
                        nn.BatchNorm3d(512), 
                        nn.ReLU())
                self.bottleneck5 = nn.Sequential(
                        Bottleneck(512, 128), 
                        Bottleneck(512, 128))
                #################################
                self.upconv1 = nn.Sequential(
                        nn.ConvTranspose3d(512, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(128), 
                        nn.ReLU())
                self.up_bottleneck1 = nn.Sequential(
                        Bottleneck(256, 64), 
                        Bottleneck(256, 64))
                self.upconv2 = nn.Sequential(
                        nn.ConvTranspose3d(256, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU())
                self.up_bottleneck2 = nn.Sequential(
                        Bottleneck(128, 32), 
                        Bottleneck(128, 32))
                self.upconv3 = nn.Sequential(
                        nn.ConvTranspose3d(128, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU())

                self.classifier = nn.Sequential(
                        nn.Conv3d(64,64, kernel_size= (3,3,3), stride = (1,1,1), padding = (1,1,1)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU(),
                        nn.Conv3d(64, 2, kernel_size= (1,1,1), padding= 0))

                initialize_weights(self)
                nn.init.xavier_uniform_(self.classifier[-1].weight)
        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 32, 32, 64]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [32, 32, 64]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images, 32*32*64 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images, 32*32*64 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 64, 32, 32] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 64, 32, 32]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 64, 32, 32]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 32, 32, 64][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 32, 32, 64] [in order x,y,z]
                ######################################################################
                feat1 = self.bottleneck1(self.conv1(_imageft)) # [N, 32, 32, 32, 64]
                feat2 = self.bottleneck2(self.conv2(feat1))  #[N, 64, 16, 16, 32]
                feat3 = self.bottleneck3(self.conv3(feat2)) # [N, 128, 8,8,16]
                feat4 = self.bottleneck4(self.conv4(feat3)) # [N, 256, 4,4,8]
                feat5 = self.bottleneck5(self.conv5(feat4)) # [N, 512, 4,4,8]

                out1 = self.upconv1(feat5) # [N, 128, 8,8,16]
                out2 = self.upconv2(self.up_bottleneck1(torch.concat((out1, feat3), dim = 1))) # [N, 64, 16, 16, 32]
                out3 =self.upconv3(self.up_bottleneck2(torch.concat((out2, feat2), dim = 1))) # [N, 32, 32, 32, 64]
                out = self.classifier(torch.concat((out3, feat1), dim = 1)) # [N, 2, 32, 32, 64]

                return out 

class ResNeXtUNet(nn.Module): 
        def __init__(self, cfg, num_images): 
                super(ResNeXtUNet, self).__init__()
                self.cfg = cfg
                self.conv1 = nn.Sequential(
                        nn.Conv3d(num_images*3, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU())
                self.bottleneck1 = nn.Sequential(
                        Bottleneck(32, 16, groups = 4), 
                        Bottleneck(32, 16, groups = 4),
                        Bottleneck(32, 16, groups = 4))   # (ResNeXt 32, 16, C = 4)
                self.conv2 = nn.Sequential(
                        nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU())
                self.bottleneck2 = nn.Sequential(
                        Bottleneck(64, 32, groups = 8), 
                        Bottleneck(64, 32, groups = 8), 
                        Bottleneck(64, 32, groups = 8), 
                        Bottleneck(64, 32, groups = 8)) # (ResNeXt 64, 32, C = 8)
                self.conv3 = nn.Sequential(
                        nn.Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(128), 
                        nn.ReLU())
                self.bottleneck3 = nn.Sequential(
                        Bottleneck(128, 64, groups = 16), 
                        Bottleneck(128, 64, groups = 16), 
                        Bottleneck(128, 64, groups = 16), 
                        Bottleneck(128, 64, groups = 16),
                        Bottleneck(128, 64, groups = 16),
                        Bottleneck(128, 64, groups = 16))  # (ResNeXt 128, 64, C = 16)
                self.conv4 = nn.Sequential(
                        nn.Conv3d(128, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(256), 
                        nn.ReLU())
                self.bottleneck4 = nn.Sequential(
                        Bottleneck(256, 128, groups = 32), 
                        Bottleneck(256, 128, groups = 32), 
                        Bottleneck(256, 128, groups = 32)) # (ResNeXt 256, 128, C = 32)
                self.conv5 = nn.Sequential(
                        nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), 
                        nn.BatchNorm3d(512), 
                        nn.ReLU())
                self.bottleneck5 = nn.Sequential(
                        Bottleneck(512, 256, groups = 32), 
                        Bottleneck(512, 256, groups = 32), 
                        Bottleneck(512, 256, groups = 32))  # (ResNeXt 512, 256, C = 32)
                #################################
                self.upconv1 = nn.Sequential(
                        nn.ConvTranspose3d(512, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(128), 
                        nn.ReLU())
                self.up_bottleneck1 = nn.Sequential(
                        Bottleneck(256, 128, groups = 32), 
                        Bottleneck(256, 128, groups = 32), 
                        Bottleneck(256, 128, groups = 32), 
                        Bottleneck(256, 128, groups = 32))  # (ResNeXt 256, 128, C = 32)
                self.upconv2 = nn.Sequential(
                        nn.ConvTranspose3d(256, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU())
                self.up_bottleneck2 = nn.Sequential(
                        Bottleneck(128, 64, groups = 16), 
                        Bottleneck(128, 64, groups = 16), 
                        Bottleneck(128, 64, groups = 16))  # (ResNeXt 128, 64, C = 16)
                self.upconv3 = nn.Sequential(
                        nn.ConvTranspose3d(128, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU())

                self.classifier = nn.Sequential(
                        nn.Conv3d(64,64, kernel_size= (3,3,3), stride = (1,1,1), padding = (1,1,1)), 
                        nn.BatchNorm3d(64), 
                        nn.ReLU(),
                        nn.Conv3d(64, 2, kernel_size= (1,1,1), padding= 0))

                initialize_weights(self)
                nn.init.xavier_uniform_(self.classifier[-1].weight)
        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 32, 32, 64]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [32, 32, 64]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images, 32*32*64 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images, 32*32*64 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 64, 32, 32] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 64, 32, 32]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 64, 32, 32]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 32, 32, 64][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 32, 32, 64] [in order x,y,z]
                ######################################################################
                feat1 = self.bottleneck1(self.conv1(_imageft)) # [N, 32, 32, 32, 64]
                feat2 = self.bottleneck2(self.conv2(feat1))  #[N, 64, 16, 16, 32]
                feat3 = self.bottleneck3(self.conv3(feat2)) # [N, 128, 8,8,16]
                feat4 = self.bottleneck4(self.conv4(feat3)) # [N, 256, 4,4,8]
                feat5 = self.bottleneck5(self.conv5(feat4)) # [N, 512, 4,4,8]

                out1 = self.upconv1(feat5) # [N, 128, 8,8,16]
                out2 = self.upconv2(self.up_bottleneck1(torch.concat((out1, feat3), dim = 1))) # [N, 64, 16, 16, 32]
                out3 =self.upconv3(self.up_bottleneck2(torch.concat((out2, feat2), dim = 1))) # [N, 32, 32, 32, 64]
                out = self.classifier(torch.concat((out3, feat1), dim = 1)) # [N, 2, 32, 32, 64]

                return out 

class SurfaceNet(nn.Module): 
        '''Network following SurfaceNet architecture, used for 3D reconstruction task'''
        def __init__(self, cfg, num_images): 
                super(SurfaceNet, self).__init__()
                self.cfg = cfg
                self.conv1 = nn.Sequential(
                        nn.Conv3d(num_images*3, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1), 
                        nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1)
                        ) # [N, 32, 32, 32, 64]

                self.upconv1 = nn.Sequential(
                        nn.ConvTranspose3d(32, 16, kernel_size=(1, 1, 1), padding = 0 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 32, 32, 64]

                self.pooling1 = nn.MaxPool3d(kernel_size=2, stride=2) # [N, 32, 16, 16, 32]

                self.conv2 = nn.Sequential(
                        nn.Conv3d(32, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1)
                        ) # [N, 80, 16, 16, 32]

                self.upconv2 = nn.Sequential(
                        nn.ConvTranspose3d(80, 16, kernel_size=(1, 1, 1), padding = 0, stride=2, output_padding=1 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 32, 32, 64]
                
                self.pooling2 = nn.MaxPool3d(kernel_size=2, stride=2) # [N, 80, 8, 8, 16]

                self.conv3 = nn.Sequential(
                        nn.Conv3d(80, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(160, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(160, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1)
                        ) # [N, 160, 8, 8, 16]

                self.upconv3 = nn.Sequential(
                        nn.ConvTranspose3d(160, 16, kernel_size=(1, 1, 1), padding = 0, stride=4, output_padding=3 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 32, 32, 64]
                self.conv4 = nn.Sequential(
                        nn.Conv3d(160, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(300, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(300, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1)
                        ) # [N, 300, 8, 8, 16]

                self.upconv4 = nn.Sequential(
                        nn.ConvTranspose3d(300, 16, kernel_size=(1, 1, 1), padding = 0, stride=4, output_padding=3 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid()) # [N, 16, 32, 32, 64]
        
                self.conv5 = nn.Sequential(
                        nn.Conv3d(64, 100, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(100), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1),
                        nn.Conv3d(100, 100, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(100), 
                        nn.ReLU(True), 
                        #nn.Dropout3d(p = 0.1)
                        ) # [N, 100, 32, 32, 64]
                self.classifier = nn.Sequential(
                        nn.Conv3d(100, 2, kernel_size= (1,1,1), padding= 0))

        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 32, 32, 64]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [32, 32, 64]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images, 32*32*64 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images, 32*32*64 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 64, 32, 32] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 64, 32, 32]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 64, 32, 32]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 32, 32, 64][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 32, 32, 64] [in order x,y,z]

                out = self.conv1(_imageft)
                s1 = self.upconv1(out) #[N, 16, 32, 32, 64]
                out = self.pooling1(out)
                
                out  = self.conv2(out)
                s2 = self.upconv2(out) #[N, 16, 32, 32, 64]
                out = self.pooling2(out)

                out = self.conv3(out)
                s3 = self.upconv3(out) #[N, 16, 32, 32, 64]

                out  = self.conv4(out)
                s4 = self.upconv4(out) #[N, 16, 32, 32, 64]
                
                out = self.conv5(torch.cat((s1,s2,s3,s4), dim = 1)) # [N, 100, 32, 32, 64]
                out = self.classifier(out) # [N, 2, 32, 32, 64]

                return out 

