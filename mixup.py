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
from torch.cuda.amp import autocast
from util import cal_loss

class MixUp(nn.Module):
    def __init__(self, alpha=0.0,device=None):
        super(MixUp, self,).__init__()
        self.alpha = alpha
        self.device = device
        self.beta = torch.distributions.beta.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha]))
    def forward(self,x,labels=None):
        if self.training:
            batch_size = x.shape[0]
            lambdaVal = self.beta.sample([batch_size]).to(self.device)
            lambdaVal2d = torch.reshape(lambdaVal,[batch_size,1])
            lambdaVal4d = torch.reshape(lambdaVal,[batch_size,1,1,1])
            mix_candidate_idxs = torch.randperm(batch_size)
            mix_candidates_data = x[mix_candidate_idxs,:,:] #YOU NEED TO DO THIS WITH THE TEDIOUS INDEXING REMEMBER?
            #print(mix_candidates_data.shape)
            #mixed_labels = []
            mix_candidates_labels = labels[mix_candidate_idxs,:]
            mixed_data = (1-lambdaVal4d)*x + lambdaVal4d*mix_candidates_data
            mixed_labels = (1-lambdaVal2d)*labels + lambdaVal2d*mix_candidates_labels
            #mixed_labels.append(mixed_label)
            return mixed_data,mixed_labels
        else:
            return x, labels