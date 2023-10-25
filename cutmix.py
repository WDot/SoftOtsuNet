#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.cuda.amp import autocast
from util import cal_loss

class CutMix(nn.Module):
    def __init__(self, alpha=1.0,device=None):
        super(CutMix, self,).__init__()
        self.alpha = alpha
        self.device = device
        self.beta = torch.distributions.beta.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha]))
    def forward(self,x,labels=None,lambdaVal=None,start_idx_vals=None):
        if self.training:
            x2 = x.detach().clone()
            batch_size = x.shape[0]
            if lambdaVal is None:
                lambdaVal = self.beta.sample([batch_size]).to(self.device)
            lambdaVal2d = torch.reshape(lambdaVal,[batch_size,1])
            #lambdaVal4d = torch.reshape(lambdaVal,[batch_size,1,1,1])
            cutSizes = torch.squeeze(x.shape[2]*(torch.sqrt(1-lambdaVal))).type(torch.int64)
            if start_idx_vals is None:
                start_idx_vals = torch.rand([batch_size],device=self.device)
            start_idxs = torch.squeeze((start_idx_vals*(x.shape[2] - cutSizes)).type(torch.int64))
            #cut_idxs = torch.arange(start_idx.item(),start_idx.item() + cutSize.item()).type(torch.int64)
            mix_candidate_idxs = torch.randperm(batch_size)
            xold = x
            for i in range(batch_size):
                #print(start_idxs[i].shape)
                #print(cutSizes[i])
                startIdx = torch.squeeze(start_idxs[i]).item()
                endIdx = startIdx + torch.squeeze(cutSizes[i]).item()
                x2[i,:,startIdx:endIdx,startIdx:endIdx] = x[mix_candidate_idxs[i].item(),:,startIdx:endIdx,startIdx:endIdx] #YOU NEED TO DO THIS WITH THE TEDIOUS INDEXING REMEMBER?
                #print(torch.mean(torch.pow(xold - x,2)))
            #print(mix_candidates_data.shape)
            #mixed_labels = []
            mix_candidates_labels = labels[mix_candidate_idxs,:]
            #mixed_data = (1-self.alpha)*x + self.alpha*mix_candidates_data
            mixed_labels = lambdaVal2d*labels + (1-lambdaVal2d)*mix_candidates_labels
            #mixed_labels.append(mixed_label)
            return x2,mixed_labels
        else:
            return x, labels