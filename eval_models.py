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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,3' # 0,1,2,3,4
print("Using gpus {}.".format(os.environ['CUDA_VISIBLE_DEVICES']))

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])

def pred_labels(pred, pred_splits, shape, tiled):
    output_t = pred.data.numpy().copy()#.astype(np.int8)
    #output_t[0,output_t[0,:]<0.9] = 0 # filter bodies differently from cell masks
    #output_t[1,output_t[1,:]<0.5] = 0 # filter bodies differently from cell masks
    output_t[output_t < 0.5] = 0
    output_t[output_t > 0] = 1;
    output_t = output_t.astype(np.int8)

    # upsample
    if not tiled:
        preds_test_upsampled = resize(output_t[0], (shape[0], shape[1]),  mode='constant', preserve_range=True)
        preds_test_upsampled = np.stack((preds_test_upsampled,resize(output_t[1], (shape[0], shape[1]),  mode='constant', preserve_range=True)))
    else:
        preds_test_upsampled = output_t

    output_t = pred_splits.cpu().data.numpy().argmax(axis=0)
    output_t[output_t == 1] = 0
    output_t[output_t == 2] = 1
    if not tiled:
        preds_test_upsampled = np.vstack((preds_test_upsampled,np.expand_dims(resize(output_t, (shape[0], shape[1]),  mode='constant', preserve_range=True),0)))
    else:
        preds_test_upsampled = np.vstack((preds_test_upsampled,np.expand_dims(output_t,0)))
   
    # predict labels
    return util.competition_loss_func(preds_test_upsampled, printMessage = True)


def write_csv(results, results_splits, images, tiled, test_ids, folder):
    ''' Loops through results, upsamples and encodes the results in the competition file layout. '''
    new_test_ids = []
    rles = []
    print("Encoding and plotting")
    for i in range(len(results)):
        sys.stdout.write("\r" + str(i))
        item = results[i]
        item_splits = results_splits[i][0]
        labels = pred_labels(item[0], item_splits, item[1][0], False)
        
        cell_size = post_processing.estimate_cell_size(labels)
        
        # ############# POST PROCESSING ################
        labels_filled = post_processing.fill_holes(labels)
        if np.sum(labels != labels_filled) > 0:
            print("Filled holes for scan {}".format(i))
            #plt.figure()
            #plt.subplot(121); plt.imshow(labels); plt.subplot(122); plt.imshow(labels_filled); plt.show(block=False)
            labels = labels_filled


        # ############ PLOTTING ################
        util.plotExampleTest(images[i][0,:], item[0].data.cpu(), labels,  i, item[1][0], folder)

        # ############### WRITE TO CSV ############
        rle = list(util.prob_to_rles(labels))
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))

    print("")
    return new_test_ids, rles

def get_cell_sizes(results, results_splits):
    cell_sizes = []
    for i in range(len(results)):
        sys.stdout.write("\r" + str(i))
        item = results[i]
        item_splits = results_splits[i][0]
        labels = pred_labels(item[0], item_splits, item[1][0], False)

        cell_sizes.append(post_processing.estimate_cell_size(labels))

    return cell_sizes

def make_predictions(model, testdataloader, testAugm, tiled, outChannels):
    inputs_ = []; results = []
    test_ids = []
    print("Make predictions")
    for i, data in enumerate(testdataloader):
        sys.stdout.write("\r" + str(i))
        inputs, shape, name = data
        if sum(testAugm) > 0:
            if tiled:
                output = t.autograd.Variable(t.from_numpy(util.evaluate_model_tiled(model, t.autograd.Variable(inputs, volatile=True).cuda(), outChannels, 256, testAugm)))
            else:
                output = t.autograd.Variable(util.eval_augmentation(model, inputs, testAugm))
        else:
            output = model(t.autograd.Variable(inputs, volatile=True).cuda())

        results.append((output.cpu().squeeze(),shape))
        inputs_.append(inputs)
        test_ids.append(name[0])
    
    print("")
    return inputs_, results, test_ids


classSelect = [0]
writeCSV = 0
tiled = False

if tiled:
    test_trans = tsf.Compose([
        tsf.ToTensor(),
        normalize
    ])
else:
    test_trans = tsf.Compose([
        tsf.ToPILImage(),
        tsf.Resize((256,256)),
        tsf.ToTensor(),
        normalize
    ])    


# class 0 = grayscale
for runClass in classSelect:
    if runClass == 0:
        numModel = [4,7]
    else:
        #numModel = [8,9]
        numModel = [1,3] # retrained with external dataset
    # ***** LOAD DATA ********
    TEST_PATH = './data/test_class' + str(runClass) + '.pth'
    cielab = True if runClass == 1 else False; #cielab = False
    testset = loadData.TestDataset(TEST_PATH, test_trans, args.normalize,cielab)
    testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=1)

    print("Load model {}".format('./models/model-cl' + str(runClass) + '-' + str(numModel[0]) + '.pt'))
    model = util.load_model('./models/model-cl' + str(runClass) + '-' + str(numModel[0]) + '.pt')
    model = model.module.eval()
    model.softmax = False
    
    inputs_, results, test_ids = make_predictions(model, testdataloader, [1,1,1,1], tiled, 2)
    
    # make predictions for all test samples
    if len(numModel) > 1:
        print("Load model {}".format('./models/model-cl' + str(runClass) + '-' + str(numModel[1]) + '.pt'))
        model_splits = util.load_model('./models/model-cl' + str(runClass) + '-' + str(numModel[1]) + '.pt')
        model_splits = model_splits.module.eval()
        _, results_splits, _ = make_predictions(model_splits, testdataloader, [1,1,1,1], tiled, 3)
        results_splits
    else:
        results_splits = []
   
    cell_sizes = get_cell_sizes(results, results_splits)

    if writeCSV:
        # upsample and encode
        new_test_ids, rles = write_csv(results, results_splits, inputs_, tiled, test_ids, 'plots/testset-hist-' + str(runClass))

        sub = pd.DataFrame()
        sub['ImageId'] = new_test_ids
        sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

        # save to submission file
        args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-' + str(numModel[0])
        util.save_submission_file(sub,args.submissionName)
