import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CutOut(nn.Module):
    def __init__(self, alpha=1.0,device=None):
        super(CutOut, self,).__init__()
        self.alpha = alpha
        self.device = device
        self.eps = 1e-3
        self.beta = torch.distributions.beta.Beta(torch.tensor([self.alpha]), torch.tensor([self.alpha]))
    def forward(self,x):
        batch_size = x.shape[0]
        mask = (1-self.eps)*torch.ones(x.shape,dtype=torch.float32,device=self.device)
        lambdaVal = self.beta.sample([batch_size]).to(self.device)
        cutSizes = torch.squeeze(x.shape[2]*(torch.sqrt(1-lambdaVal))).type(torch.int64)
        start_idx_vals = torch.rand([batch_size],device=self.device)
        start_idxs = torch.squeeze((start_idx_vals*(x.shape[2] - cutSizes)).type(torch.int64))
        for i in range(batch_size):
            startIdx = torch.squeeze(start_idxs[i]).item()
            endIdx = startIdx + torch.squeeze(cutSizes[i]).item()
            mask[i,:,startIdx:endIdx,startIdx:endIdx] = 1e-3 #Maskout
        return mask