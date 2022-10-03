from typing import List

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import ASPPConv, ASPPPooling
import torch.nn as nn
import torch
import torch.nn.functional as F


class ASPPCustom(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], hidden_dim, out_channels: int = 128) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False), 
                nn.BatchNorm2d(hidden_dim), 
                nn.ReLU()
            )
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, hidden_dim, rate))

        modules.append(ASPPPooling(in_channels, hidden_dim))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            # relu and dropout in the next layer below
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)

class DeepLabHeadCustom(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.aspp = ASPPCustom(in_channels, [12, 24, 36], 128, 128)
        self.conv = nn.Sequential(
            # relu and dropout that were in ASPP
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, 1)
        )

    def forward(self, x, return_features, return_preds):
        x = self.aspp(x)
        feats = x
        # only features
        if return_features and not return_preds:
            return feats, None
        # get predictions, return both
        preds = self.conv(x)
        return feats, preds

class DeepLabv3(nn.Module):
    ''' 
    pretrained deeplabv3 + finetune on our dataset
    '''
    def __init__(self, num_classes, cfg=None):
        super(DeepLabv3, self).__init__()
        self.num_classes = num_classes

        self.dlv3 = deeplabv3_resnet50(pretrained=True, progress=True)
        self.dlv3.aux_classifier = None

        self.dlv3.classifier = DeepLabHeadCustom(2048, self.num_classes)
    
    def forward(self, x, return_features=False, return_preds=True):
        input_shape = x.shape[-2:]
        features = self.dlv3.backbone(x)
        x = features["out"]
        
        # now get intermediate features or final output
        feats, preds = self.dlv3.classifier(x, return_features=return_features, 
                                            return_preds=return_preds)
        # handle all 3 cases - feats/preds only, both
        if return_features and not return_preds:
            return feats
        if return_preds:
            # get preds at original size
            preds = F.interpolate(preds, size=input_shape, mode="bilinear", align_corners=False)
            if return_features:
                return feats, preds
            else:
                return preds