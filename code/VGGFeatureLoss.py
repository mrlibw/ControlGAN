from __future__ import division
from torchvision import models
from torchvision import transforms
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.utils.model_zoo as model_zoo

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['8'] ## relu2_2 

        model = models.vgg16()
        url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        model.load_state_dict(model_zoo.load_url(url))

        for param in model.parameters():
        	param.resquires_grad = False

        print('Load pretrained model from ', url)
        self.vgg = model.features
        
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

