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
args.numEpochs = 400 
args.learnWeights = True
args.dataAugm = True
args.imgWidth = 256



os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 0,1,2,3,4

runClass = 1 

args.modelName = 'model-cl' + str(runClass) + '-0'
args.submissionName = 'sub-dsbowl2018_cl' + str(runClass) + '-0'

# ***** LOAD DATA ********
TRAIN_PATH = './data/train_class' + str(runClass) + '.pth'
TEST_PATH = './data/test_class' + str(runClass) + '.pth'


# Class 0: 541
# Class 1: 124
trainSamples = 541 if (runClass == 0) else 124


splits = loadData.createKSplits(trainSamples, 5, random_state=0)
train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)


# ***** LOAD DATA ********
# TRAIN_PATH = './data/train.pth'
# TEST_PATH = './data/test.pth'
# splits = loadData.createKSplits(670, 5, random_state=0)
# train_data, val_data = loadData.readFromDisk(splits[0],TRAIN_PATH)


normalize = tsf.Normalize(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])
# normalize = tsf.Normalize(mean = [0.17071716,  0.15513969,  0.18911588], std = [0.03701544,  0.05455154,  0.03268249])

if args.dataAugm:
    st_trans = tsf.Compose([
        tsf.ToPILImage(),
	# tsf.Resize((256,256)) # 382
    ])

    s_trans = tsf.Compose([
        # tsf.CenterCrop(256)
        tsf.ToPILImage(),
        tsf.Resize((args.imgWidth,args.imgWidth)), # 382
        tsf.ToTensor(),
        normalize,
    ])

    t_trans = tsf.Compose([
        # tsf.CenterCrop(256),
        tsf.ToPILImage(),
        tsf.Resize((args.imgWidth,args.imgWidth)), # 382
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


test_trans = tsf.Compose([
    tsf.ToPILImage(),
    tsf.Resize((256,256)),
    tsf.ToTensor(),
    normalize
])    



dataset = loadData.Dataset(train_data, s_trans, t_trans, st_trans, args.dataAugm, args.imgWidth)
dataloader = torch.utils.data.DataLoader(dataset, num_workers = 2, batch_size = 4)

validset = loadData.Dataset(val_data, s_trans, t_trans, st_trans, args.dataAugm, args.imgWidth)
validdataloader = torch.utils.data.DataLoader(validset, num_workers = 2, batch_size = 4)



# ***** SET MODEL *****
# model = UNet(1, depth=5, merge_mode='concat').cuda(0) # Alternative implementation
model = UNet2(3,2,learn_weights=args.learnWeights) # Kaggle notebook implementation

model = nn.DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(),lr = 0.2*1e-3)
lossFunc = util.soft_dice_loss


# ***** TRAIN *****

def evaluate_model(model, lossFunc):
    running_accuracy = 0
    running_score = 0
    for i, data in enumerate(validdataloader, 0):
        inputs, masks, masks_multiLabel = data
        x_valid = torch.autograd.Variable(inputs).cuda()
        y_valid = torch.autograd.Variable(masks).cuda()

        # forward
        output = model(x_valid)
        loss = lossFunc(output, y_valid)

        # statistics
        running_accuracy += loss.data

        for j in range(inputs.shape[0]):
            # evalute competition loss function
            score, _ = util.competition_loss_func(output[j,:].data.cpu().numpy(),masks_multiLabel[j,0,:].numpy())
            running_score += score 

    return (1.0-running_accuracy/(i+1.0)), running_score/len(validdataloader.dataset)


def train_model(model, lossFunc, num_epochs=100):
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            inputs, masks, masks_multiLabel = data
            x_train = torch.autograd.Variable(inputs).cuda()
            y_train = torch.autograd.Variable(masks).cuda()
            optimizer.zero_grad()
         
            # forward
            output = model(x_train)
            loss = lossFunc(output, y_train)

            # train
            loss.backward()
            optimizer.step()
           
            # statistics
            running_loss += loss.data
           
            if i % args.iterPrint == args.iterPrint-1:    # print every iterPrint mini-batch
                print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / args.iterPrint))
                running_loss = 0.0

            # plot some segmented training examples
            if 0 and i % args.iterPlot == args.iterPlot-1:
                idx = 0
                #ipdb.set_trace()
                score, _ = util.competition_loss_func(output[idx,0,:].data.cpu().numpy(),masks_multiLabel[idx,0,:].numpy())
                util.plotExample(inputs[idx,:], masks[idx,0,:,:], output[idx,0,:,:].data, epoch, i, lossFunc(output[idx,:].data.cpu(), masks[idx,:]), score, False)

                for j in range(inputs.shape[0]):
                    # evalute competition loss function
                    score, _  = util.competition_loss_func(output[j,0,:].data.cpu().numpy(),masks_multiLabel[j,0,:].numpy())
                    print(score)

        acc, score = evaluate_model(model, lossFunc)
        print('acc: %.3f, score: %.3f' % (acc, score))

    return model
  
model = train_model(model, lossFunc, args.numEpochs)


util.save_model(model,args.modelName) 

#util.plot_results_for_images(model,dataloader)

# ***** EVALUATION ********
testset = loadData.TestDataset(TEST_PATH, test_trans)
testdataloader = t.utils.data.DataLoader(testset,num_workers=2,batch_size=1)

# make predictions for all test samples
model = model.eval()
results = []
test_ids = []
for i, data in enumerate(testdataloader):
    print(i)
    inputs, shape, name = data
    x_test = t.autograd.Variable(inputs, volatile=True).cuda()
    output = model(x_test)
    results.append((output.cpu().squeeze(),shape))
    test_ids.append(name[0])
    #idx = 0 
    #util.plotExample(inputs[idx,:], output[idx,0,:,:].data, output[idx,0,:,:].data, 0, 0, 0, 0, True)

# upsample and encode
new_test_ids = []
rles = []
for i,item in enumerate(results):
    print(i)
    output_t = (item[0] > 0.5).data.numpy().astype(np.uint8)
    # upsample
    preds_test_upsampled = resize(output_t[0], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)
    preds_test_upsampled = np.stack((preds_test_upsampled,resize(output_t[1], (item[1][0][0], item[1][0][1]),  mode='constant', preserve_range=True)))

    labels = util.competition_loss_func(preds_test_upsampled)

    rle = list(util.prob_to_rles(labels))
    rles.extend(rle)
    new_test_ids.extend([test_ids[i]] * len(rle))

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

# save to submission file
util.save_submission_file(sub,'sub-dsbowl2018-0')
