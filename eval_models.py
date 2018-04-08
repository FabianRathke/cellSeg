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

imp.reload(util)
imp.reload(loadData)

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
numModel = [2,3]

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3' # 0,1,2,3,4
print("Using gpus {}.".format(os.environ['CUDA_VISIBLE_DEVICES']))

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])

test_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256,256)),
    tsf.ToTensor(),
    normalize
])    

def write_csv(results, results_splits, images, test_ids, folder):
    ''' Loops through results, upsamples and encodes the results in the competition file layout. '''
    new_test_ids = []
    rles = []
    print("Encoding and plotting")
    for i,item in enumerate(results):
        sys.stdout.write("\r" + str(i))
        output_t = item[0].data.numpy()#.astype(np.int8)
        #output_t[0,output_t[0,:]<0.9] = 0 # filter bodies differently from cell masks
        #output_t[1,output_t[1,:]<0.5] = 0 # filter bodies differently from cell masks
        output_t[output_t < 0.5] = 0
        output_t[output_t > 0] = 1;
        output_t = output_t.astype(np.int8)

        # upsample
        preds_test_upsampled = resize(output_t[0], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)
        preds_test_upsampled = np.stack((preds_test_upsampled,resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)))
        if len(results_splits) == len(results):
            data = results_splits[i][0].cpu().data.numpy()
            #output_t = (results_splits[i][0] > 0.5).data.numpy().astype(np.int8) # old approach with one model having smaller cell bodies (model 5)
            output_t = data.argmax(axis=0)
            output_t[output_t == 1] = 0
            output_t[output_t == 2] = 1
            #preds_test_upsampled[1,:] = resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)
            preds_test_upsampled = np.vstack((preds_test_upsampled,np.expand_dims(resize(output_t, (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True),0)))
       
        # predict labels
        labels = util.competition_loss_func(preds_test_upsampled)
        util.plotExampleTest(images[i][0,:], item[0].data.cpu(), labels,  i, item[1][0], folder)

        rle = list(util.prob_to_rles(labels))
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))

    print("")
    return new_test_ids, rles

def make_predictions(model, testdataloader):
    inputs_ = []; results = []
    test_ids = []
    print("Make predictions")
    for i, data in enumerate(testdataloader):
        sys.stdout.write("\r" + str(i))
        inputs, shape, name = data
        output = model(t.autograd.Variable(inputs, volatile=True).cuda())
        results.append((output.cpu().squeeze(),shape))
        inputs_.append(inputs)
        test_ids.append(name[0])
    
    print("")
    return inputs_, results, test_ids


# class 0 = grayscale
for runClass in range(1):
    # ***** LOAD DATA ********
    TEST_PATH = './data/test_class' + str(runClass) + '.pth'
    testset = loadData.TestDataset(TEST_PATH, test_trans, args.normalize)
    testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=1)

    print("Load model {}".format('./models/model-cl' + str(runClass) + '-' + str(numModel[0]) + '.pt'))
    model = util.load_model('./models/model-cl' + str(runClass) + '-' + str(numModel[0]) + '.pt')
    model = model.module.eval()
    model.softmax = False
    
    inputs_, results, test_ids = make_predictions(model, testdataloader)
    
    # make predictions for all test samples
    if len(numModel) > 1:
        print("Load model {}".format('./models/model-cl' + str(runClass) + '-' + str(numModel[1]) + '.pt'))
        model = util.load_model('./models/model-cl' + str(runClass) + '-' + str(numModel[1]) + '.pt')
        model = model.module.eval()
        _, results_splits, _ = make_predictions(model, testdataloader)
        results_splits
    else:
        results_splits = []
   
    # ipdb.set_trace() 
    # upsample and encode
    new_test_ids, rles = write_csv(results, results_splits, inputs_, test_ids, 'plots/testset-hist-' + str(runClass))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    # save to submission file
    args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-' + str(numModel[0])
    util.save_submission_file(sub,args.submissionName)
