import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL

# PYTORCH
import torch
from torch.utils import data
from torchvision import transforms as tsf
import ipdb

# OUR FUNCTIONS
from models import *
import loadData
import util

# ***** LOAD DATA ********
TRAIN_PATH = './data/train.pth'
TEST_PATH = './data/test.tph'

splits = loadData.createKSplits(670, 5, random_state=0)
train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)

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

validset = loadData.Dataset(val_data,s_trans,t_trans)
validdataloader = torch.utils.data.DataLoader(validset,num_workers=2,batch_size=4)


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.iterPrint = 5
args.iterPlot = 20
args.numEpochs = 20

# ***** SET MODEL *****
# model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
model = UNet2(3,1) # Kaggle notebook implementation

#os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,5'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)
lossFunc = util.soft_dice_loss


# ***** TRAIN *****

def evaluate_model(model, lossFunc):
   running_accuracy = 0
   for i, data in enumerate(validdataloader, 0):
      inputs, masks, masks_multilabel = data
      x_valid = torch.autograd.Variable(inputs).cuda()
      y_valid = torch.autograd.Variable(masks).cuda()

      # forward
      output = model(x_valid)
      loss = lossFunc(output, y_valid)

      # statistics
      running_accuracy += loss.data

   return 1.0-running_accuracy/(i+1.0)


def train_model(model, lossFunc, num_epochs=100):
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, masks, masks_multiLabel = data
            x_train = torch.autograd.Variable(inputs).cuda()
            y_train = torch.autograd.Variable(masks).cuda()
            optimizer.zero_grad()
         
            # forward
            output = model(x_train)
            loss = lossFunc(output, y_train)

            # train
            loss.backward()
            optimizer.step()
           
            # statistics
            running_loss += loss.data
           
            if i % args.iterPrint == args.iterPrint-1:    # print every iterPrint mini-batch
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / args.iterPrint))
                running_loss = 0.0

            # plot some segmented training examples
            if 0 and i % args.iterPlot == args.iterPlot-1:
                idx = 0
                #ipdb.set_trace()
                util.plotExample(inputs[idx,:], masks[idx,0,:,:], output[idx,0,:,:].data, epoch, i, lossFunc(output[idx,:].data.cpu(), masks[idx,:]), False)

        acc = evaluate_model(model, lossFunc)
        print('acc: %.3f' % (acc))

    return model

  
model = train_model(model, lossFunc, args.numEpochs)


# ***** EVALUATION ********
testset = loadData.TestDataset(TEST_PATH, s_trans)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=4)

# TODO
# model = model.eval()
# for data in testdataloader:
#     data = t.autograd.Variable(data, volatile=True).cuda())
#     o = model(data)
#     break







