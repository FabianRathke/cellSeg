import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from skimage.transform import resize
import sys
import ipdb
import imp

# PYTORCH
import torch
from torch.utils import data
from torchvision import transforms as tsf

# OUR FUNCTIONS
from models import *
import loadData
import util
import post_processing


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.dataAugm = True
args.imgWidth = 256
args.normalize = False

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
test_trans = tsf.Compose([
	tsf.ToPILImage(),
    #tsf.Resize((256,256)),
    tsf.ToTensor(),
    normalize
])

runClass = 0
TEST_PATH = './data/test_final_class' + str(runClass) + '.pth'
cielab = False # True if runClass == 1 else False; #cielab = False
testset0 = loadData.TestDataset(TEST_PATH, test_trans, args.normalize,cielab)

runClass = 1
TEST_PATH = './data/test_final_class' + str(runClass) + '.pth'
cielab = False # True if runClass == 1 else False; #cielab = False
testset1 = loadData.TestDataset(TEST_PATH, test_trans, args.normalize,cielab)


def correct_csv(testset, df):
    nameprev = ''
    for i, rleString in enumerate(df['EncodedPixels']):
        name = df['ImageId'][i]
        
        # ipdb.set_trace()

        if name != nameprev: 
            for k in range(len(testset)):
                if testset[k][2] == name:
                    (rows,cols) = testset[k][1]
                    nameprev = name        
                    break

        try:
            rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
            if (rleNumbers[-1] + rleNumbers[-2] - 1) > rows*cols:
                #print("out of bounds")
                rleNumbers[-1] -= 1
                # ipdb.set_trace()
                df['EncodedPixels'][i] = ' '.join(str(e) for e in rleNumbers)
                # df['EncodedPixels'][i] = pd.Series(rleNumbers).apply(lambda x: ' '.join(str(y) for y in x))

        except:
            #print("idx %i: %s" % (i, rleString))
            df['EncodedPixels'][i] = " "
    return df


print("Correcting color")
df1 = pd.read_csv('./sub-dsbowl2018_cl1-7.csv')
df1_correctd = correct_csv(testset1,df1)
    
print("Correcting grayscale")    
df0 = pd.read_csv('./sub-dsbowl2018_cl0-32.csv')
df0_corrected = correct_csv(testset0,df0)

# ipdb.set_trace()

frames = [df0_corrected, df1_corrected]
result = pd.concat(frames)
result.to_csv('result.csv', index=False)




