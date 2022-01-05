import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from projection import Projection
from utils.helpers import initialize_weights
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(planes, inplanes, kernel_size=1)

        #self.bn1 = nn.BatchNorm3d(planes)
        #self.bn2 = nn.BatchNorm3d(planes)
        #self.bn3 = nn.BatchNorm3d(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        #out = self.bn3(out)
        out = self.relu(out)
        return out

class Dense3DNetwork(nn.Module):
        '''Dense 3D CNN network consisting series of ResNet blocks'''
        def __init__(self, cfg, num_images): 
                super(Dense3DNetwork, self).__init__()
                self.cfg = cfg
                self.encoder  = nn.Sequential(
                        nn.Conv3d(num_images*3, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # [N, 64, 48, 24, 48]
                        #nn.BatchNorm3d(64),
                        nn.ReLU(True),
                        Bottleneck(64, 32, stride=1), # [N, 64, 48, 24, 48]
                        nn.MaxPool3d(3, 1, 1), # [N, 64, 48, 24, 48]
                        nn.Conv3d(64, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # [N, 128, 24, 12, 24]
                        #nn.BatchNorm3d(128),
                        nn.ReLU(True),
                        Bottleneck(128, 32, stride=1), # [N, 128, 24, 12, 24 ]
                        nn.MaxPool3d(3, 1, 1), # [N, 128, 24, 12, 24],
                        nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), # [N, 128, 24, 12, 24 ]
                        #nn.BatchNorm3d(128),
                        nn.ReLU(True),
                        Bottleneck(128, 64, stride=1), # [N, 128, 24, 12, 24 ]
                        Bottleneck(128, 64, stride=1), # [N, 128, 24, 12, 24 ]
                        nn.MaxPool3d(3, 1, 1))  # [N, 128, 24, 12, 24 ]
                self.decoder = nn.Sequential(
                        nn.ConvTranspose3d(128, 128, kernel_size= (3,3,3), stride= (1,1,1), padding= (1,1,1)),  # [N, 128, 24, 12, 24]
                        #nn.BatchNorm3d(128),
                        nn.ReLU(True), 
                        Bottleneck(128, 64, stride = 1),
                        nn.ConvTranspose3d(128, 64, kernel_size= (2,2,2), stride= (2,2,2), padding = (0,0,0)),  # [N, 64, 48, 24, 48]
                        #nn.BatchNorm3d(64),
                        nn.ReLU(True),  
                        Bottleneck(64, 32, stride = 1),
                        nn.ConvTranspose3d(64, 32, kernel_size= (2,2,2), stride= (2,2,2), padding = (0,0,0)),  #  [N, 32, 96, 48, 96]
                        #nn.BatchNorm3d(32),
                        nn.ReLU(True), 
                        Bottleneck(32, 16, stride = 1), # [N, 32, 96, 48, 96]
                        nn.Conv3d(32, 1, kernel_size= (3,3,3), stride= (1,1,1), padding= (1,1,1))) # [N, 1, 96, 48, 96]
                initialize_weights(self)
                nn.init.xavier_uniform_(self.decoder[9].weight)

        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 2, 96, 48, 96]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [96,48,96]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images,96*48*96 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images,96*48*96 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 96, 48, 96] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 96, 48, 96]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 96, 48, 96]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 96, 48, 96][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 96, 48, 96] [in order x,y,z]
                out = self.encoder(_imageft)  # [N, 128, 24, 12, 24 ]
                out = self.decoder(out) # [N, 1, 96, 48, 96]

                return out
                #return _imageft

        
class SurfaceNet(nn.Module): 
        '''Network following SurfaceNet architecture, used for 3D reconstruction task'''
        def __init__(self, cfg, num_images): 
                super(SurfaceNet, self).__init__()
                self.cfg = cfg
                self.conv1 = nn.Sequential(
                        nn.Conv3d(num_images*3, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(32), 
                        nn.ReLU(True), 
                        ) # [N, 32, 96, 48, 96]

                self.upconv1 = nn.Sequential(
                        nn.ConvTranspose3d(32, 16, kernel_size=(1, 1, 1), padding = 0 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 96, 48, 96]

                self.pooling1 = nn.MaxPool3d(kernel_size=2, stride=2) # [N, 32, 48, 24, 48]

                self.conv2 = nn.Sequential(
                        nn.Conv3d(32, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        nn.Conv3d(80, 80, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(80), 
                        nn.ReLU(True), 
                        ) # [N, 80, 48, 24, 48]

                self.upconv2 = nn.Sequential(
                        nn.ConvTranspose3d(80, 16, kernel_size=(1, 1, 1), padding = 0, stride=2, output_padding=1 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 96, 48, 96]
                
                self.pooling2 = nn.MaxPool3d(kernel_size=2, stride=2) # [N, 80, 24, 12, 24]

                self.conv3 = nn.Sequential(
                        nn.Conv3d(80, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        nn.Conv3d(160, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        nn.Conv3d(160, 160, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(160), 
                        nn.ReLU(True), 
                        ) # [N, 160, 24, 12, 24]

                self.upconv3 = nn.Sequential(
                        nn.ConvTranspose3d(160, 16, kernel_size=(1, 1, 1), padding = 0, stride=4, output_padding=3 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid())  #[N, 16, 96, 48, 96]
                self.conv4 = nn.Sequential(
                        nn.Conv3d(160, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        nn.Conv3d(300, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        nn.Conv3d(300, 300, kernel_size=(3, 3, 3), padding =2, stride=1, dilation=2), 
                        nn.BatchNorm3d(300), 
                        nn.ReLU(True), 
                        ) # [N, 300, 24, 12, 24]

                self.upconv4 = nn.Sequential(
                        nn.ConvTranspose3d(300, 16, kernel_size=(1, 1, 1), padding = 0, stride=4, output_padding=3 ),
                        nn.BatchNorm3d(16), 
                        nn.Sigmoid()) # [N, 16, 96 , 48, 96]
        
                self.conv5 = nn.Sequential(
                        nn.Conv3d(64, 100, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(100), 
                        nn.ReLU(True), 
                        nn.Conv3d(100, 100, kernel_size=(3, 3, 3), padding =1), 
                        nn.BatchNorm3d(100), 
                        nn.ReLU(True), 
                        ) # [N, 100, 96, 48, 96]
                self.classifier = nn.Sequential(
                        nn.Conv3d(100, 1, kernel_size= (1,1,1), padding= 0))

        def forward(self, blobs, device): 
                #blobs['data']: [batch_size, 2, 96, 48, 96]
                self.batch_size = blobs['data'].shape[0]
                grid_shape = blobs['data'].shape[-3:] # [96,48,96]
                _imageft = []
                for i in range(self.batch_size):
                        num_images = blobs['nearest_images']['images'][i].shape[0] # max_num_images
                        imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
                        proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images,96*48*96 + 1]
                        proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images,96*48*96 + 1]

                        imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
                        imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 96, 48, 96] [max_num_images, C, z, y, x]
                        sz = imageft.shape # [max_num_images, 3, 96, 48, 96]
                        imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 96, 48, 96]
                        _imageft.append(imageft.permute(0,3,2,1).contiguous()) # list of [max_num_images*3, 96, 48, 96][max_num_images*3, x,y,z]
                _imageft = torch.stack(_imageft, dim = 0)  # [batch_size, max_num_images*3, 96, 48, 96] [in order x,y,z]

                out = self.conv1(_imageft)
                s1 = self.upconv1(out) #[N, 16, 96, 48, 96]
                out = self.pooling1(out)
                
                out  = self.conv2(out)
                s2 = self.upconv2(out) #[N, 16, 96, 48, 96]
                out = self.pooling2(out)

                out = self.conv3(out)
                s3 = self.upconv3(out) #[N, 16, 96, 48, 96]

                out  = self.conv4(out)
                s4 = self.upconv4(out) #[N, 16, 96, 48, 96]
                
                out = self.conv5(torch.cat((s1,s2,s3,s4), dim = 1)) # [N, 100, 96, 48, 96]
                out = self.classifier(out) # [N, 1, 96, 48, 96]

                return out 

