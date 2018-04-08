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
args.numEpochs = 125
args.learnWeights = True
args.dataAugm = True
args.imgWidth = 256
args.normalize = True

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4' # 0,1,2,3,4
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



# class 0 = grayscale
for runClass in range(1):
    args.modelName = 'model-cl' + str(runClass) + '-0'
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
    #lossFunc = util.soft_dice_loss
    lossFunc = util.cross_entropy
    # normalize = tsf.Normalize(mean = [0.17071716,  0.15513969,  0.18911588], std = [0.03701544,  0.05455154,  0.03268249])

    dataset = loadData.Dataset(train_data, s_trans, t_trans, st_trans, args.dataAugm, True, args.imgWidth, maskConf = [0,0,0])
    dataloader = torch.utils.data.DataLoader(dataset, num_workers = 2, batch_size = 4)

    validset = loadData.Dataset(val_data, s_trans, t_trans, st_trans, args.dataAugm, True, args.imgWidth, maskConf = [0,0,0])
    validdataloader = torch.utils.data.DataLoader(validset, num_workers = 2, batch_size = 4)

    #model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
    model = UNet2(3,3,learn_weights=args.learnWeights, softmax=True) # Kaggle notebook implementation
    model = nn.DataParallel(model).cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.2*1e-3)

    model = util.train_model(model, optimizer, lossFunc, dataloader, validdataloader, args)
    util.save_model(model,args.modelName) 
    #model.eval()

    #numModel = [6]
    #model = util.load_model('./model-cl' + str(runClass) + '-' + str(numModel[0]) + '.pt')
    #model = model.module.eval()

    #util.plot_all_predictions(model,validdataloader,'plots/gallery_'+str(runClass))
