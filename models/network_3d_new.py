import math
import torch
import torch.nn as nn
from projection import Projection
class BottleneckDown(nn.Module):
    """
    RexNeXt bottleneck type C, for downsampling
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, downsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(BottleneckDown, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv3d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(D*C)
        self.conv2 = nn.Conv3d(D*C, D*C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm3d(D*C)
        self.conv3 = nn.Conv3d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckUp(nn.Module):
    """
    RexNeXt bottleneck type C, for upsampling
    """
    expansion = 4

    def __init__(self, inplanes, planes, baseWidth, cardinality, stride=1, upsample=None):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: base width.
            cardinality: num of convolution groups.
            stride: conv stride. Replaces pooling layer.
        """
        super(BottleneckUp, self).__init__()

        D = int(math.floor(planes * (baseWidth / 64)))
        C = cardinality

        self.conv1 = nn.Conv3d(inplanes, D*C, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(D*C)
        self.conv2 = nn.ConvTranspose3d(D*C, D*C, kernel_size=3, stride=stride, padding=1,output_padding=stride-1, groups=C, bias=False)
        self.bn2 = nn.BatchNorm3d(D*C)
        self.conv3 = nn.Conv3d(D*C, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.upsample = upsample

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

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out



class Model3DResNeXt(nn.Module):
    """
    3D reconstruction network with ResNeXt blocks
    """
    def __init__(self, cfg, num_images):
        """ Constructor
        Args:
            baseWidth: baseWidth for ResNeXt.
            cardinality: number of convolution groups.
            layers: config of layers, e.g., [3, 4, 6, 3]
            num_classes: number of classes
        """
        super(Model3DResNeXt, self).__init__()
        self.cfg = cfg
        self.inplanes = 128 if cfg["use_2d_feat_input"] else num_images*3  # 15 or 128
        self.cardinality = 32
        self.baseWidth = 4
        layers = [3,3, 27,3, 5,3,3]

        #################### Encoder ###############################
        self.layer1 = self._make_layer(BottleneckDown, 16, layers[0])
        self.layer2 = self._make_layer(BottleneckDown, 32, layers[1], 2)
        self.layer3 = self._make_layer(BottleneckDown, 64, layers[2], 2)
        self.layer4 = self._make_layer(BottleneckDown, 128, layers[3])

        #################### Decoder ###############################
        self.layer5 = self._make_layer(BottleneckUp, 64, layers[4], 2)
        self.layer6 = self._make_layer(BottleneckUp, 32, layers[5], 2)
        self.layer7 = self._make_layer(BottleneckUp, 16, layers[6])
        self.classifier = nn.Conv3d(64, 2, kernel_size=1, stride=1, bias=False)

        #################### Initialization ########################
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            block: block type used to construct ResNext
            planes: number of output channels (need to multiply by block.expansion)
            blocks: number of blocks to be built
            stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.inplanes < planes* block.expansion: 
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else: 
                downsample = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, output_padding = stride-1, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.baseWidth, self.cardinality))

        return nn.Sequential(*layers)

    #def forward(self, blobs, device):
    def forward(self, x):
        #blobs['data']: [batch_size, 32, 32, 64]
        '''self.batch_size = blobs['data'].shape[0]
        grid_shape = blobs['data'].shape[-3:] # [32, 32, 64]
        x = []
        for i in range(self.batch_size):
            if self.cfg['use_2d_feat_input']: 
                imageft = blobs['feat_2d'][i] # [max_num_images, 128, 32, 41]
            else: 
                imageft = blobs['nearest_images']['images'][i].to(device)  #[max_num_images, 3, 256, 328]
            proj3d = blobs['proj_ind_3d'][i].to(device) # [max_num_images, 32*32*64 + 1]
            proj2d = blobs['proj_ind_2d'][i].to(device) #[max_num_images, 32*32*64 + 1]

            imageft = [Projection.apply(ft, ind3d, ind2d, grid_shape) for ft, ind3d, ind2d in zip(imageft, proj3d, proj2d)]
            if self.cfg['use_2d_feat_input']: 
                imageft = torch.stack(imageft, dim=4) # [128, 64, 32, 32, max_num_images] [C, z, y, x, max_num_images]
                sz = imageft.shape # [128, 64, 32, 32, max_num_images]
                imageft = imageft.view(sz[0], -1, sz[4]) # [128, 64*32*32, max_num_images]
                imageft = self.initial_pooling(imageft)  # [128, 64*32*32,1]
                imageft = imageft.view(sz[0], sz[1], sz[2], sz[3]) # [128, 64, 32, 32]
                x.append(imageft.permute(0,3,2,1)) # list of [128, 32, 32, 64] [in order x, y, z]

            else: 
                imageft = torch.stack(imageft, dim=0) #[max_num_images, 3, 64, 32, 32] [max_num_images, C, z, y, x]
                sz = imageft.shape # [max_num_images, 3, 64, 32, 32]
                imageft = imageft.view(-1, sz[2], sz[3], sz[4]) # [max_num_images*3, 64, 32, 32]
                x.append(imageft.permute(0,3,2,1)) # list of [max_num_images*3, 32, 32, 64][max_num_images*3, x,y,z]
        x = torch.stack(x, dim = 0)  # [batch_size, (max_num_images*3) | 128, 32, 32, 64] [in order x,y,z]'''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.classifier(x)

        return x
