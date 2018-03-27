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
numModel = 2

os.environ['CUDA_VISIBLE_DEVICES'] = '4' # 0,1,2,3,4
print("Using gpus {}.".format(os.environ['CUDA_VISIBLE_DEVICES']))

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])

test_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256,256)),
    tsf.ToTensor(),
    normalize
])    


#def equalHist(inputs):
#    for i in range(inputs.shape[0]):
#        img = inputs[i,:].permute(1,2,0).numpy()*0.5 + 0.5
#        img_adapteq = exposure.equalize_adapthist(img)
#        img_denoise = denoise_tv_chambolle(img_adapteq, weight=0.02, multichannel=True)

# class 0 = grayscale
for runClass in range(2):

    args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-' + str(numModel)

    # ***** LOAD DATA ********
    TEST_PATH = './data/test_class' + str(runClass) + '.pth'

    model = util.load_model('./model-cl' + str(runClass) + '-' + str(numModel) + '.pt')
    model = model.module#.cuda(int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])) # unwrap the data parallelism
    
    # ***** EVALUATION ********
    testset = loadData.TestDataset(TEST_PATH, test_trans)
    testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=1)

    # make predictions for all test samples
    model = model.eval()
    inputs_ = []
    results = []
    test_ids = []
    for i, data in enumerate(testdataloader):
        print(i)
        inputs, shape, name = data
        x_test = t.autograd.Variable(inputs, volatile=True).cuda()
        output = model(x_test)
        results.append((output.cpu().squeeze(),shape))
        inputs_.append(inputs)
        test_ids.append(name[0])
        
    # upsample and encode
    new_test_ids = []
    rles = []
    for i,item in enumerate(results):
        print(i)
        output_t = (item[0] > 0.5).data.numpy().astype(np.int8)
        # upsample
        preds_test_upsampled = resize(output_t[0], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)
        preds_test_upsampled = np.stack((preds_test_upsampled,resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)))

        labels = util.competition_loss_func(preds_test_upsampled)
        util.plotExampleTest(inputs_[i][0,:], item[0].data.cpu(), labels,  i, item[1][0], 'plots/testset-' + str(runClass))

        rle = list(util.prob_to_rles(labels))
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    # save to submission file
    util.save_submission_file(sub,args.submissionName)
