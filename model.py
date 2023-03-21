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

class EfficientNetB4(nn.Module):
    def __init__(self, output_classes,device):
        super(EfficientNetB4, self).__init__()
        self.device = device
        self.conv = timm.create_model('tf_efficientnet_b4_ns',pretrained=True,num_classes=0,drop_rate=0.4,drop_path_rate=0.2)
        self.classifier = nn.Sequential(#nn.Dropout(0.5),
                                        nn.Linear(self.conv.num_features,output_classes))

    def forward(self, x,noise=None):
        x = self.conv(x)
        return self.classifier(x)
