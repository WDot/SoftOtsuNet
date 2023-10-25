#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ConvNext(nn.Module):
    def __init__(self, output_classes,device):
        super(ConvNext, self).__init__()
        self.device = device
        self.conv = timm.create_model('convnext_base.fb_in22k_ft_in1k',pretrained=True,num_classes=0,drop_rate=0.4,drop_path_rate=0.2)
        self.do = nn.Dropout(0.5)
        self.classifier = nn.Sequential(#nn.Dropout(0.5),
                                        nn.Linear(self.conv.num_features,output_classes))
        self.means = nn.Parameter(torch.reshape(torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32),[1,3,1,1]),requires_grad=False)
        self.stds = nn.Parameter(torch.reshape(torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32),[1,3,1,1]),requires_grad=False)

    def forward(self, x,noise=None):
        #xnorm = (x- self.means) / (self.stds + 1e-8)
        #xinput = xnorm.detach()
        fv = self.do(self.conv(x))
        #print
        return self.classifier(fv)
