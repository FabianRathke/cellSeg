import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import PIL
from skimage.transform import resize
import pandas as pd

# PYTORCH
import torch
from torch.utils import data
from torchvision import transforms as tsf
import ipdb

# OUR FUNCTIONS
from models import *
import loadData
import util
from skimage import exposure

# model = util.load_model('model-1.pt')

useCentroid = 0
submissionName = 'sub-dsbowl2018-cl0-0'

model_cl0 = util.load_model('model-cl0-6.pt')
model_cl1 = util.load_model('model-cl0-6.pt')


TEST_PATH = './data/test.pth'
# TEST_PATH = './data/train_class1.pth'

normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])

test_trans = tsf.Compose([
    tsf.ToPILImage(),
    # tsf.Resize((256,256)),
    tsf.ToTensor(),
    normalize
])    


# s.environ['CUDA_VISIBLE_DEVICES'] = 0 # 0,1,2,3,4

df = pd.read_csv('class_means.csv', sep=',',header=None, index_col=False)
classmeans = np.genfromtxt('class_means.csv', delimiter=',')[1:,1:]
classmeans /= 255.0

# ipdb.set_trace()
 
# util.plot_results_for_images(model_cl0, dataloader)

# ***** EVALUATION ********
testset = loadData.TestDataset(TEST_PATH, test_trans, normalize=True)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=1)

# make predictions for all test samples
model_cl0 = model_cl0.eval()
model_cl1 = model_cl1.eval()
results = []
test_ids = []
for i, data in enumerate(testdataloader):
    print(i)
    inputs, shape, name = data
    # ipdb.set_trace()
    x_test = t.autograd.Variable(inputs, volatile=True).cuda()
    img = x_test[0,:].cpu().permute(1,2,0).data.numpy()*0.5 + 0.5
    rows, cols, dims = img.shape
    imgmean = np.mean(np.reshape(img.copy(),(rows*cols,dims)), axis=0)
    imgmean.shape = (imgmean.shape[0], 1)
    cdist = np.zeros((classmeans.shape[0],1))
    for k in range(classmeans.shape[0]):
        cdist[k] = np.sum((classmeans[k,:] - imgmean[0:3])**2)
        

    #ipdb.set_trace()    
    c = np.int( np.argmin(cdist) )
    if c == 0:
        print("Class 0")
        plt.figure(1)
        #plt.subplot(1,2,1)
        plt.imshow(img)

        #for r in range(3):
            # img[:,:,r] = exposure.equalize_hist(img[:,:,r])
            #img[:,:,r] = (exposure.equalize_hist(img[:,:,r])*255).astype(np.uint8)
        #    img[:,:,r] = exposure.equalize_adapthist(img[:,:,r])

        #plt.subplot(1,2,2)
        #plt.imshow(img)
        plt.show()
            
        #img = (img-0.5)/0.5
        #x_test[0,:] = t.from_numpy(img).type(torch.FloatTensor).permute(2,0,1).cuda()
        
        #output = model_cl0(x_test)
        output = util.evaluate_model_tiled(model_cl0, x_test, 256)
        # ipdb.set_trace()

    else:
        output = util.evaluate_model_tiled(model_cl1, x_test, 256)
        # output = model_cl1(x_test)
  
    # ipdb.set_trace() 

    # results.append((output.cpu().squeeze(),shape))
    results.append((output,shape))
    test_ids.append(name[0])
   
    #idx = 0 
    #util.plotExample(inputs[idx,:], output[idx,0,:,:].data, output[idx,0,:,:].data, 0, 0, 0, 0, True)
    
    #ipdb.set_trace()
    if 1:
        plt.figure(1)
        plt.subplot(2,2,1)
        plt.imshow(inputs[0,:].cpu().permute(1,2,0).numpy()*0.5 + 0.5)
        plt.subplot(2,2,2)
        plt.imshow(output[0,:]) #0,:,:].cpu().data.numpy())
        plt.subplot(2,2,3)
        plt.imshow(output[1,:]) #0,1,:,:].cpu().data.numpy())
        #plt.subplot(2,2,4)
        #plt.imshow(output[0,2,:,:].cpu().data.numpy())
        plt.show()


    # plotExample(inputs[idx,:], masks[idx,:], labels_pred, epoch, i, lossFunc(output[idx,:].data.cpu(), masks[idx,:]), score, False, 'gallery')


# util.plot_all_results(model, testdataloader, 'gallery')



if 1:
    # upsample and encode
    new_test_ids = []
    rles = []
    for i,item in enumerate(results):
        print(i)
     
        # ipdb.set_trace() 
        preds_test_upsampled = item[0].astype(np.uint8)

        # output_t = (item[0] > 0.5).data.numpy().astype(np.uint8)
        # upsample
        # ipdb.set_trace()
     
        

        #preds_test_upsampled = resize(output_t[0], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)
        #preds_test_upsampled = np.stack((preds_test_upsampled,resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)))

        #preds_test_upsampled = np.stack((resize(output_t[0], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True),
        #                                resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True),
        #                                resize(output_t[2], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)), axis=0)

        labels = util.competition_loss_func(preds_test_upsampled,useCentroid=useCentroid)

        # ipdb.set_trace()

        rle = list(util.prob_to_rles(labels))
        rles.extend(rle)
        new_test_ids.extend([test_ids[i]] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

    # save to submission file
    util.save_submission_file(sub,'sub-dsbowl2018-0')







