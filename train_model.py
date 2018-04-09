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


classSelect = [1]

# bodies unet
for runClass in classSelect:
    args.modelName = './models/model-cl' + str(runClass) + '-0'
    args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-0'

    # ***** LOAD DATA ********
    TRAIN_PATH = './data/train_class' + str(runClass) + '.pth'
    TEST_PATH = './data/test_class' + str(runClass) + '.pth'

    # Class 0: 541
    # Class 1: 124
    trainSamples = 541 if (runClass == 0) else 124

    splits = loadData.createKSplits(trainSamples, 5, random_state=0)
    # use no validation set
    splits[0] = np.zeros((0))
    train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)

    normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
    lossFunc = util.soft_dice_loss

    print("Train Unet with dice loss")
    cielab = True if runClass == 1 else False; cielab = False
    histEq = True if runClass == 0 else False
    dataset = loadData.Dataset(train_data, s_trans, t_trans, st_trans, args.dataAugm, histEq, args.imgWidth, maskConf = [1,1,0], cielab=cielab)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers = 2, batch_size = 4)

    validset = loadData.Dataset(val_data, s_trans, t_trans, st_trans, args.dataAugm, histEq, args.imgWidth, maskConf = [1,1,0], cielab=cielab)
    validdataloader = torch.utils.data.DataLoader(validset, num_workers = 2, batch_size = 4)

    #model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
    model = UNet2(3,2,learn_weights=args.learnWeights, softmax=False) # Kaggle notebook implementation
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.2*1e-3)

    model = util.train_model(model, optimizer, lossFunc, dataloader, validdataloader, args)
    util.save_model(model,args.modelName) 

# boundary detection model
if 1:
    for runClass in classSelect:
        args.modelName = './models/model-cl' + str(runClass) + '-0'
        args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-0'

        # ***** LOAD DATA ********
        TRAIN_PATH = './data/train_class' + str(runClass) + '.pth'
        TEST_PATH = './data/test_class' + str(runClass) + '.pth'

        # Class 0: 541
        # Class 1: 124
        trainSamples = 541 if (runClass == 0) else 124

        print("Load data")
        splits = loadData.createKSplits(trainSamples, 5, random_state=0)
        # use no validation set
        splits[0] = np.zeros((0))
        train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)

        normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
        lossFunc = util.cross_entropy

        print("Train Unet with softmax")
        cielab = True if runClass == 1 else False; cielab = False
        histEq = True if runClass == 0 else False
        dataset = loadData.Dataset(train_data, s_trans, t_trans, st_trans, args.dataAugm, histEq, args.imgWidth, maskConf = [0,0,0], cielab=cielab)
        dataloader = torch.utils.data.DataLoader(dataset, num_workers = 2, batch_size = 4)


        validset = loadData.Dataset(val_data, s_trans, t_trans, st_trans, args.dataAugm, histEq, args.imgWidth, maskConf = [0,0,0], cielab=cielab)
        validdataloader = torch.utils.data.DataLoader(validset, num_workers = 2, batch_size = 4)

        #model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
        model = UNet2(3,3,learn_weights=args.learnWeights, softmax=True) # Kaggle notebook implementation
        model = nn.DataParallel(model).cuda()

        optimizer = torch.optim.Adam(model.parameters(),lr = 0.2*1e-3)

        model = util.train_model(model, optimizer, lossFunc, dataloader, validdataloader, args)
        util.save_model(model,args.modelName) 
