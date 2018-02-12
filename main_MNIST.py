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

import ipdb

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
TRAIN_PATH = './data/train.pth'
TEST_PATH = './data/test.tph'

#splits = loadData.createKSplits(670, 5, random_state=0)
#train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)

s_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((32,32)),
    tsf.ToTensor(),
    tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
]
)
t_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((32,32),interpolation=PIL.Image.NEAREST),
    tsf.ToTensor(),]
)

#dataset = loadData.Dataset(train_data,s_trans,t_trans)
#dataloader = torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)

#validset = loadData.Dataset(val_data,s_trans,t_trans)
#validdataloader = torch.utils.data.DataLoader(validset,num_workers=2,batch_size=4)


dataset = datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))

# Replace the numerical value with mask
dataset = [( data[0],  (data[0] > 0).type(torch.FloatTensor)) for data in dataset]

dataloader = torch.utils.data.DataLoader(dataset,num_workers=2,batch_size=4)



# Validataion data
validset= datasets.MNIST('../data', train=False, download=True,
                            transform=transforms.Compose([
                            transforms.Pad(2),
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                            ]))

validset = [( data[0],  (data[0] > 0).type(torch.FloatTensor)) for data in validset]

validdataloader = torch.utils.data.DataLoader(validset,num_workers=2,batch_size=4)

def visualize_image(data,idx):
    d = data[idx][0].cpu().permute(1,2,0).numpy()
    m = data[idx][1].cpu().permute(1,2,0).numpy()
    if d.shape[-1] == 1:
        d = np.reshape(d,d.shape[:-1])
        m = np.reshape(m,m.shape[:-1])

    plt.subplot(1,2,1)
    plt.imshow(d)
    plt.title('idx ' + np.str(idx)) 
    plt.subplot(1,2,2)
    plt.imshow(m)
    plt.title('mask')



# plt.figure(1)
# visualize_image(dataset,1)

# plt.figure(2)
# visualize_image(validset,1)
# plt.show()


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.iterPrint = 5
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
                print('[%d, %5d] loss: %.3f' %
               (epoch + 1, i + 1, running_loss / args.iterPrint))
                running_loss = 0.0
   
      acc = evaluate_model(model)
      
      print('acc: %.3f' % (acc))

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







