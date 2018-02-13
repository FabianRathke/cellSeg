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
from torchvision import datasets, transforms

from logger_tensorboard.logger import Logger 


import ipdb

# ***** LOGGING *****

logger = Logger('./logs')

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_np(x):
    return x.data.cpu().numpy()


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

dataset = datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))

# Replace the numerical value with mask
dataset = [( data[0],  (data[0] > 0).type(torch.FloatTensor)) for data in dataset]

dataloader = torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=100)

# Validataion data
validset = datasets.MNIST('../data', train=False, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))

validset = [( data[0],  (data[0] > 0).type(torch.FloatTensor)) for data in validset]

validdataloader = torch.utils.data.DataLoader(validset,num_workers=2,batch_size=100)


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.iterPrint = 1
args.numEpochs = 20

# ***** SET MODEL *****
# model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
model = UNet2(1,1) # Kaggle notebook implementation

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3'
model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3)


# ***** TRAIN *****

def evaluate_model(model):
   running_accuracy = 0
   for i, data in enumerate(validdataloader, 0):
      inputs, masks = data
      x_valid = torch.autograd.Variable(inputs).cuda()
      y_valid = torch.autograd.Variable(masks).cuda()

      # forward
      output = model(x_valid)
      loss = soft_dice_loss(output, y_valid)

      # statistics
      running_accuracy += loss.data

   return 1.0-running_accuracy/(i+1.0)


def train_model(model, num_epochs=200):
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):

            # Choose only very small subset of data, comment for full dataset.
            if i < 50:
                inputs, masks = data
                # _  , m = data
                 
                # ipdb.set_trace()


                x_train = torch.autograd.Variable(inputs).cuda()
                y_train = torch.autograd.Variable(masks).cuda()


                optimizer.zero_grad()
              
                # forward
                output = model(x_train)
                loss = soft_dice_loss(output, y_train)

                # train
                loss.backward()
                optimizer.step()
                
                # statistics
                running_loss += loss.data

                if i % args.iterPrint == args.iterPrint-1:    # print every iterPrint mini-batch
                    acc = evaluate_model(model)
                     
                    print('[%d, %5d] loss: %.3f, acc %.3f' %
                        (epoch + 1, i + 1, running_loss / args.iterPrint, acc))
                    running_loss = 0.0


                    # ================= TensorBoard logging ==================#
                    # (1) Log the scalar values
                    info = {
                        'loss': loss.data[0],
                        'accuracy': acc
                    }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, i+1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in model.named_parameters():
                        tag = tag.replace('.', '/')
                        logger.histo_summary(tag, to_np(value), i+1)
                        logger.histo_summary(tag+'/grad', to_np(value.grad), i+1)

                    # (3) Log the images
                    info = {
                        'images': inputs.view(-1, 32, 32)[:10].cpu().numpy()
                    }

                    for tag, images in info.items():
                    	logger.image_summary(tag, images, i+1)
          
    return model

  
model = train_model(model,args.numEpochs)

# ipdb.set_trace()


# ***** EVALUATION ********
# testset = loadData.TestDataset(TEST_PATH, s_trans)
# testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=4)

# TODO
# model = model.eval()
# for data in testdataloader:
#     data = t.autograd.Variable(data, volatile=True).cuda())
#     o = model(data)
#     break







