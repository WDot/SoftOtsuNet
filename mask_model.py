from unet_model import UNet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from cutout import CutOut

class MaskModel(nn.Module):
    def __init__(self,device):
        super(MaskModel, self).__init__()
        self.device = device

        self.in_channels = 5
        self.cutout = CutOut(device=device)
        self.unet = UNet(self.in_channels,self.in_channels+1,True)

    def forward(self, x,noise=None):
        B = x.shape[0]
        xVals = torch.tile(torch.reshape(torch.linspace(-1,1,x.shape[2],dtype=torch.float32,device=self.device),[1,1,x.shape[2],1]),[x.shape[0],1,1,x.shape[3]])
        yVals = torch.tile(torch.reshape(torch.linspace(-1,1,x.shape[3],dtype=torch.float32,device=self.device),[1,1,1,x.shape[3]]),[x.shape[0],1,x.shape[2],1])
        xFull = torch.cat((x,xVals,yVals),dim=1)
        yboth,fv = self.unet(xFull)

        yraw = yboth[:,self.in_channels,:,:].unsqueeze(1)
        rgb2grey = torch.softmax(yboth[:,0:self.in_channels,:,:],1)

        xgrey = torch.sum(xFull*rgb2grey,dim=1,keepdims=True)


        y = torch.sigmoid(yraw)

        if self.training:
            mask = self.cutout(y)
            y = y*mask

        #otsu soft loss
        w1 = torch.mean(y,dim=(1,2,3))
        w0 = torch.mean((1 - y),dim=(1,2,3)) 

        masked_x_grey = xgrey*y
        inverse_masked_x_grey = xgrey - masked_x_grey

        var1 = torch.var(masked_x_grey,dim=(1,2,3))
        var0 = torch.var(inverse_masked_x_grey,dim=(1,2,3))

        #mean distance separation
        means_distance = torch.mean(masked_x_grey,dim=(1,2,3)) - torch.mean(inverse_masked_x_grey,dim=(1,2,3))
        sqdistance = torch.square(means_distance)
        means_distance_loss = torch.exp(-sqdistance)

        ysmooth = F.interpolate(F.avg_pool2d(y,2),scale_factor=2,mode='bilinear',antialias=True,align_corners=True).detach()
        #ysmooth = F.interpolate(F.max_pool2d(y,10),scale_factor=10,mode='nearest').detach()
        smoothness_loss = F.binary_cross_entropy_with_logits(yraw,ysmooth)


        loss = w1*var1 + w0*var0 + means_distance_loss + smoothness_loss

        maskChooser = (torch.mean(y,dim=(1,2,3),keepdims=True) > torch.mean(1-y,dim=(1,2,3),keepdims=True)).type(torch.float32)

        outmask = y*maskChooser + (1-y)*(1-maskChooser)
    
        return outmask,loss