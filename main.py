import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL

import torch
from torch.utils import data
from torchvision import transforms as tsf

from models import *
import loadData

# ****** LOSS FUNCTION ******
def soft_dice_loss(inputs, targets):
        num = targets.size(0)
        m1  = inputs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1 - score.sum()/num
        return score


# ***** LOAD DATA ********

splits = loadData.createKSplits(670, 5, random_state=0)
train_data, val_data = loadData.readFromDisk(splits[0])

s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256,256)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256,256),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)

dataset = loadData.Dataset(train_data,s_trans,t_trans)
dataloader = torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.iterPrint = 5

#model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
model = UNet2(3,1).cuda(0) # Kaggle notebook implementation
optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)

for epoch in range(200):
    running_loss = 0
    for i, data in enumerate(dataloader, 0):
    #for x_train, y_train in tqdm(dataloader):
        inputs, masks = data
        x_train = torch.autograd.Variable(inputs).cuda(0)
        y_train = torch.autograd.Variable(masks).cuda(0)
        optimizer.zero_grad()
        o = model(x_train)
        loss = soft_dice_loss(o, y_train)
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.data
       	if i % args.iterPrint == args.iterPrint-1:    # print every iterPrint mini-batches
            print('[%d, %5d] loss: %.3f' %
	      (epoch + 1, i + 1, running_loss / args.iterPrint))
            running_loss = 0.0


