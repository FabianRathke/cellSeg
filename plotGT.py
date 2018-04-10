import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from skimage.transform import resize

# PYTORCH
import torch
from torch.utils import data
from torchvision import transforms as tsf
import ipdb

# OUR FUNCTIONS
from models import *
import loadData
import util

# ***** SET PARAMETERS *****

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.iterPrint = 5
args.iterPlot = 20
args.numEpochs = 100
args.learnWeights = True
args.dataAugm = True
args.imgWidth = 256
args.normalize = True

os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3' # 0,1,2,3,4
print("Using gpus {}.".format(os.environ['CUDA_VISIBLE_DEVICES']))

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
if args.dataAugm:
    st_trans = tsf.Compose([
        tsf.ToPILImage(),
        # tsf.Resize((256,256)) # 382
    ])

    s_trans = tsf.Compose([
        # tsf.CenterCrop(256)
        tsf.ToPILImage(),
        tsf.Resize((args.imgWidth,args.imgWidth), interpolation=PIL.Image.BILINEAR), # 382
        tsf.ToTensor(),
        normalize,
    ])

    t_trans = tsf.Compose([
        # tsf.CenterCrop(256),
        tsf.ToPILImage(),
        tsf.Resize((args.imgWidth,args.imgWidth), interpolation=PIL.Image.NEAREST), # 382
        tsf.ToTensor()
    ])
else:
    st_trans = None

    s_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((256,256)),
        tsf.ToTensor(),
        normalize,
    ])
    t_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((256,256),interpolation=PIL.Image.NEAREST),
        tsf.ToTensor()
    ])


runClass = 0
trainSamples = 541 if (runClass == 0) else 124


TRAIN_PATH = './data/train_class' + str(runClass) + '.pth'

splits = loadData.createKSplits(trainSamples, 5, random_state=0)
# use no validation set
splits[0] = np.zeros((0))
train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)
cielab = True if runClass == 1 else False; cielab = False
histEq = True if runClass == 0 else False
dataset = loadData.Dataset(train_data, s_trans, t_trans, st_trans, args.dataAugm, histEq, args.imgWidth, maskConf = [1,1,0], cielab=cielab)

for i,data in enumerate(dataset):
    print(i)
    plt.figure(figsize=(15,8))
    plt.subplot(121)
    plt.imshow(data[0].permute(1,2,0)*0.5 + 0.5)
    plt.subplot(122)
    plt.imshow(data[2][0,:])
    plt.savefig("./plots/gallery_0/img_{}.png".format(i))
    plt.close()
