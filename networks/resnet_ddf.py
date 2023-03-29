# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from ddf import DDFBlock

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResnetDDF(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetDDF, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.layers = [2, 2, 2, 2]
        self.num_layers = num_layers


        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if pretrained:
            self._init_pretrained_weights(model_urls['resnet18'])
        
        self.layer1 = self._make_layer(DDFBlock, self.num_ch_enc[1], self.layers[0])
        self.layer2 = self._make_layer(DDFBlock, self.num_ch_enc[2], self.layers[1], stride=2)
        self.layer3 = self._make_layer(DDFBlock, self.num_ch_enc[3], self.layers[2], stride=2)
        self.layer4 = self._make_layer(DDFBlock, self.num_ch_enc[4], self.layers[3], stride=2)
            
    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        self.features.append(x)
        self.features.append(self.layer1(self.maxpool(self.features[-1])))
        self.features.append(self.layer2(self.features[-1]))
        self.features.append(self.layer3(self.features[-1]))
        self.features.append(self.layer4(self.features[-1]))
        return self.features

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_pretrained_weights(self, model_url):
        pretrain_dict = model_zoo.load_url(model_url)
        self.load_state_dict(pretrain_dict, strict=False)