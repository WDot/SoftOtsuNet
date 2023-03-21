#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data_fitzpatrick import Fitzpatrick17k
from model import EfficientNetB4
from mask_model import MaskModel
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss,gini,IOStream
import sklearn.metrics as metrics
import torch.distributed as dist
import sys
from PIL import Image
from cutout import CutOut
from collections import deque
import time
from timm.utils import dispatch_clip_grad
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
from sklearn.model_selection import train_test_split


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    if not os.path.exists('checkpoints/'+args.exp_name + '/visuals/'):
        os.makedirs('checkpoints/'+args.exp_name + '/visuals/')
    if not os.path.exists('checkpoints/'+args.exp_name + '/masks/'):
        os.makedirs('checkpoints/'+args.exp_name + '/masks/')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')
    

def train(args, io):

    stds = np.reshape([0.229, 0.224, 0.225], [1,1,-1])
    means = np.reshape([0.485, 0.456, 0.406], [1,1,-1])

    dataset= pd.read_csv(args.label_path)
    unique_labels = pd.unique(dataset['label'])
    label_dict = {}
    all_labels = np.zeros(dataset['label'].shape[0])
    for i in range(unique_labels.shape[0]):
        label_dict[unique_labels[i]] = i
    for i in range(dataset['label'].shape[0]):
        all_labels[i] = label_dict[dataset['label'].iloc[i]]


    X_train,X_test,_,_ = train_test_split(np.arange(dataset.shape[0]),all_labels,test_size=0.2,random_state=args.seed,stratify=all_labels)

    print(X_train)
    
    train_dataset = Fitzpatrick17k(args.image_path,dataset,X_train,label_dict,augment=True,is_eval=False,randaug=args.randaug)
    train_loader = DataLoader(train_dataset, num_workers=12,
                              batch_size=args.batch_size,shuffle=True,drop_last=True)
    test1_dataset = Fitzpatrick17k(args.image_path,dataset,X_test,label_dict,augment=False,is_eval=True,randaug=False)
    test1_loader = DataLoader(test1_dataset, num_workers=12,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = 'cuda'

    model = EfficientNetB4(train_dataset.num_classes,device=device).to(device)
    if not args.baseline:
        mask_model = MaskModel(device=device).to(device)

    model.cuda()
    if not args.baseline:
        mask_model.cuda()

    if args.baseline and args.cutout:
        cutout = CutOut(device=device)
        cutout.cuda()

    sys.stdout.flush()

    model = nn.DataParallel(model)
    if not args.baseline:
        mask_model = nn.DataParallel(mask_model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    print("Use SGD")
    if not args.baseline:
        opt = optim.SGD([
                    {'params': model.parameters()},
                    {'params': mask_model.parameters()}
                ], lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=True)
    else:
        opt = optim.SGD([
                    {'params': model.parameters()},
                ], lr=args.lr, momentum=args.momentum, weight_decay=1e-4, nesterov=True)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    
    scaler = torch.cuda.amp.GradScaler()

    #test_accs = deque(maxlen=5)
    for epoch in range(args.epochs):
        t = time.time()
        opt.zero_grad()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        if not args.baseline:
            mask_model.train()
        if args.baseline and args.cutout:
            cutout.train()
        train_preds = []
        train_trues = []
        for data, labels, fitz, noise in train_loader:
            noise = noise.to(device)
            data= data.to(device)
            labels = labels.to(device)
            batch_size = data.size()[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                if not args.baseline:
                    y,otsuloss = mask_model(data)
                    
                    xBimodal = (data * y) + ((1 - y) * noise)
                    metamask = torch.reshape((torch.rand([data.shape[0]],dtype=torch.float32,device=device) > 0.5),[-1,1,1,1]).type(torch.float32)
                    x = xBimodal*metamask + (1-metamask)*data
                    outputs = model(x.detach())
                    loss = cal_loss(outputs,labels) + otsuloss
                else:
                    if args.cutout:
                        mask = cutout(data)
                        metamask = torch.reshape((torch.rand([data.shape[0]],dtype=torch.float32,device=device) > 0.5),[-1,1,1,1]).type(torch.float32)
                        xCutout = data * mask + (1- mask) * noise
                        x = xCutout * metamask + (1-metamask)*data
                    else:
                        x = data
                    outputs = model(x)
                    loss = cal_loss(outputs,labels)
                loss = loss.mean()
            scaler.scale(loss).backward()

            scaler.step(opt)
            
            scaler.update()
            preds = outputs.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_trues.append(labels.detach().cpu().numpy())
            train_preds.append(preds.detach().cpu().numpy())
            iteration = (count / batch_size)
            #if count > 100:
            #    break
            #break
        train_trues = np.concatenate(train_trues)
        train_preds = np.concatenate(train_preds)
        scheduler.step()
        outstr = 'Train {0}, loss: {1}, train acc: {2}'.format(epoch,
                                                                train_loss*1.0/count,
                                                                metrics.accuracy_score(np.argmax(train_trues,-1), train_preds))
        io.cprint(outstr)
        sys.stdout.flush()

        ####################
        # Test
        ####################
        
        test_loss = 0.0
        count = 0.0
        model.eval()
        if not args.baseline:
            mask_model.eval()
        if args.baseline and args.cutout:
            cutout.eval()
        test_preds = []
        test_trues = []
        test_items = []
        test_outputs = []
        #test_masks = []
        with torch.no_grad():
            for data, labels, fitz,noise in test1_loader:
                noise = noise.to(device)
                data= data.to(device)
                labels = labels.to(device)
                batch_size = data.size()[0]
                with torch.cuda.amp.autocast():

                    outputs = model(data)
                    loss = cal_loss(outputs,labels)
                loss = loss.mean()
                preds = outputs.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_trues.append(labels.detach().cpu().numpy())
                test_preds.append(preds.detach().cpu().numpy())
                test_items.append(fitz.detach().cpu().numpy())
                test_outputs.append(data.detach().cpu().numpy())
                #if count > 100:
                #    break

        test_trues = np.concatenate(test_trues)
        test_preds = np.concatenate(test_preds)
        test_items = np.concatenate(test_items)
        test_outputs = np.concatenate(test_outputs)
        fitz_accs = []
        for findex in range(6):
            test_trues_index = np.argmax(test_trues,-1)
            per_fitz_test_trues_index = test_trues_index[test_items == (findex + 1)]
            per_fitz_test_preds = test_preds[test_items == (findex + 1)]
            fitz_accs.append(metrics.accuracy_score(per_fitz_test_trues_index,per_fitz_test_preds))
        mean_test_accs = metrics.accuracy_score(np.argmax(test_trues,-1), test_preds)
        outstr = 'Test Masked {0}, loss: {1}, test acc: {2}, fitz accs: {3} gini: {4}'.format(epoch,
                                                                test_loss*1.0/count,
                                                                mean_test_accs,
                                                                fitz_accs,
                                                                gini(fitz_accs))
        io.cprint(outstr)
        sys.stdout.flush()
        if np.mean(mean_test_accs) >= best_test_acc:
            best_test_acc = np.mean(mean_test_accs)
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            if not args.baseline:
                torch.save(mask_model.state_dict(), 'checkpoints/%s/models/mask_model.t7' % args.exp_name)
            io.cprint('Model saved!')
        print('Total time: {0}'.format(time.time() - t))


def test(args, io):
    dataset= pd.read_csv(args.label_path)
    unique_labels = pd.unique(dataset['label'])
    label_dict = {}
    all_labels = np.zeros(dataset['label'].shape[0])
    for i in range(unique_labels.shape[0]):
        label_dict[unique_labels[i]] = i
    for i in range(dataset['label'].shape[0]):
        all_labels[i] = label_dict[dataset['label'].iloc[i]]


    X_train,X_test,_,_ = train_test_split(np.arange(dataset.shape[0]),all_labels,test_size=0.2,random_state=100,stratify=all_labels)
    test_dataset = Fitzpatrick17k(args.image_path,dataset,X_test,label_dict,augment=False,is_eval=True)
    test_loader = DataLoader(test_dataset, num_workers=12,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = EfficientNetB4(test_dataset.num_classes,device=device).to(device)
    model.cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_preds = []
    test_trues = []
    test_items = []
    t = time.time()
    for data, labels, items,noise in test_loader:
        data= data.to(device)
        labels = labels.to(device)

        batch_size = data.size()[0]
        outputs,otsuloss = model(data,noise)
        loss = cal_loss(outputs, labels, smoothing=True) + 0.1*otsuloss
        loss = loss.mean()
        preds = outputs.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_trues.append(labels.detach().cpu().numpy())
        test_preds.append(preds.detach().cpu().numpy())
        test_items.append(items.detach().cpu().numpy())
        if count > 100:
            break
    test_trues = np.concatenate(test_trues)
    test_preds = np.concatenate(test_preds)
    test_items = np.concatenate(test_items)
    fitz_accs = []
    mean_test_accs = metrics.accuracy_score(np.argmax(test_trues,-1), test_preds)
    outstr = 'Test Results, loss: {0}, test acc: {1}'.format(
                                                                test_loss*1.0/count,
                                                                mean_test_accs,
                                                                )
    io.cprint(outstr)
    sys.stdout.flush()
    print('Total time: {0}'.format(time.time() - t))
    


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Derm Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgn', 'dgndeep','dgndeep2'],
                        help='Model to use, [dgcnn, graphcnn]')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_path_old', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--randaug', action='store_true')
    parser.add_argument('--cutout',action='store_true')
    parser.add_argument('--image_path',type=str)
    parser.add_argument('--label_path',type=str)

    args = parser.parse_args()

    _init_()
    #if dist.get_rank() == 0:
    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    #else:
    #    io = None

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        #if dist.get_rank() == 0:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        #if dist.get_rank() == 0:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
