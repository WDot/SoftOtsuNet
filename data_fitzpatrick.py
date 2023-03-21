#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms.functional as TF
import torch
import PIL
import json
from PIL import Image
from timm.data.auto_augment import rand_augment_transform
import pandas as pd

class Fitzpatrick17k(Dataset):
    def __init__(self,
        path,
        csv_list,
        indices,
        label_dict,
        initial_side_length=int(380*1.15),
        crop_size=380,
        fitz_one_hot=False,
        augment=True,
        is_eval=False,
        randaug=False):
        self.path = path
        self.indices = indices
        self.images = csv_list
        self.label_dict = label_dict
        self.num_classes = len(list(self.label_dict.keys()))
        self.initial_side_length = initial_side_length
        self.crop_size = crop_size
        self.fitz_one_hot = fitz_one_hot
        self.augment = augment
        self.is_eval = is_eval
        self.randaug = randaug

        #self.queue = list(self.images.keys())
        if self.augment:
            self.hflip = tv.transforms.RandomHorizontalFlip(p=0.5)
            self.vflip = tv.transforms.RandomVerticalFlip(p=0.5)
            self.crop = tv.transforms.RandomCrop(self.crop_size)
            self.randaug = rand_augment_transform('rand-m15-n1',None)
        else:
            self.crop = tv.transforms.CenterCrop(self.crop_size)
        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def smooth_labels(self,labels, factor):
        shape_index = 1 if len(labels.shape) > 1 else 0
        smoothed_labels = labels * (1 - factor)
        smoothed_labels += (factor / labels.shape[shape_index])
        return smoothed_labels

    def get_ground_truth(self,idx):
        index = self.label_dict[self.images['label'].iloc[self.indices[idx]]]
        one_hot = np.zeros(self.num_classes)
        one_hot[index] = 1
        one_hot = self.smooth_labels(one_hot,0.2)
        one_hot /= np.sum(one_hot)
        return one_hot

    def get_fitzpatrick(self,idx):
        index = self.images['fitzpatrick'].iloc[self.indices[idx]]
        if self.fitz_one_hot:
            one_hot = np.zeros(6)
            one_hot[index-1] = 1
            #one_hot = self.smooth_labels(one_hot,0.2)
            return one_hot.astype(np.float32)
        else:
            return index


    def __getitem__(self, item):
        filename = self.path + self.images['md5hash'].iloc[self.indices[item]] + '.jpg'#self.queue[item]

        X = Image.open(filename).convert('RGB')
        X = np.array(X)

        X = torch.tensor(X,dtype=torch.float32)
        X = X.permute(2,0,1)
        Xshape = torch.tensor([X.shape[1],X.shape[2]])
        biggerIndex = torch.argmax(Xshape)
        smallerIndex = torch.argmin(Xshape)
        aspectratio = float(Xshape[biggerIndex]) / Xshape[smallerIndex]
        if biggerIndex == 1:
            H = self.initial_side_length
            W = int(self.initial_side_length*aspectratio)
        else:
            H = int(self.initial_side_length*aspectratio)
            W = self.initial_side_length
        X = TF.resize(X,(H,W),interpolation=TF.InterpolationMode.BICUBIC)
        X = self.crop(X).type(torch.FloatTensor)

        if self.augment:

            X = self.hflip(X)
            X = self.vflip(X)
            X = X.permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
            X = Image.fromarray(X)
            if self.randaug:
                X = np.array(self.randaug(X))
            else:
                X = np.array(X)
            X = torch.tensor(X,dtype=torch.uint8).permute(2,0,1).float()
        X = X / 255.0
        X = self.normalize(X)

        y = self.get_ground_truth(item)
        fitz = self.get_fitzpatrick(item)
        
        if self.augment:
            noise = torch.rand(X.shape)
            noise = self.normalize(noise)
            return X,y,fitz,noise
        else:
            noise = X
            return X,y,fitz,noise

    def __len__(self):
        return self.indices.shape[0]

if __name__ == '__main__':
    train = Fitzpatrick17k()
    test = Fitzpatrick17k()
