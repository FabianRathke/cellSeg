import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import measure
from skimage.transform import resize
from scipy import ndimage
import skimage.morphology as morph
from skimage.morphology import watershed
from skimage.feature import peak_local_max
import ipdb
import os.path
from skimage.measure import regionprops

from models import *

# ****** LOSS FUNCTION ******
def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1  = inputs.view(num,-1)
    m2  = targets.view(num,-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1)+10**-10) / (m1.sum(1) + m2.sum(1)+10**-10)
    score = 1 - score.sum()/num
    return score


def soft_dice_loss2(inputs, targets):
    ''' Loss function from from UnitBox:
        https://arxiv.org/pdf/1608.01471.pdf '''
    num = targets.size(0)
    m1  = inputs.view(num,-1)
    m2  = targets.view(num,-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1) - intersection.sum(1) + 1)
    score = - torch.log( score.sum() )
    return score





def plotExample(img, mask, pred, epoch, batch, loss, lossComp, interactive=False, folder = 'plots'):
    ''' Plots ground truth and prediction for a cell image '''
    if interactive:
        plt.ion()
    plt.figure(figsize=(15, 6))
    plt.subplot(131)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
    plt.subplot(132)
    plt.imshow(mask)
    plt.subplot(133)
    plt.imshow(pred)
    plt.suptitle("Epoch {} and batch {}: Loss = {:.2f}, CompEval = {:.2f}".format(epoch, batch, loss, lossComp))
    if interactive:
        plt.pause(0.05)
        plt.draw()
    else:
        plt.savefig("{}/epoch_{}_batch_{}.png".format(folder,epoch,batch))
        plt.close()


def competition_loss_func(inputs, targets = None):
    ''' https://www.kaggle.com/c/data-science-bowl-2018#evaluation
        evaluates IoU (intersection over union) for various thresholds and calculates
        TPs, FPs and FNs for all objects (detected vs. ground truth). The final score
        is averaged over all thresholds. '''

    # ipdb.set_trace()
    data_inputs = inputs.copy()
    thresholds = np.arange(0.5,1,0.05)
  
    if inputs.shape[0] > 1:
        diff = inputs[0,:]-inputs[1,:]
        diff[diff < 0.9] = 0
        diff[diff > 0.9] = 1
	#plt.subplot(221); plt.imshow(inputs[0,:]); plt.subplot(222); plt.imshow(inputs[1,:]); plt.subplot(223); plt.imshow(diff); plt.subplot(224); plt.imshow(targets); plt.show()
        labels = measure.label(diff)
        bodies = inputs[0,:]
        bodies[bodies > 0.9] = 1 # threshold
        inputs = morph.watershed(-ndimage.distance_transform_edt(diff), labels, mask=bodies)


        unique, counts = np.unique(inputs, return_counts=True)
        radi = np.sqrt(np.median(counts[1:-1])/np.pi)
        # get average size of nuclei
        
        current_label = 1
        corrected = np.zeros_like(inputs)
        for k in unique[1:-1]:
            label_k = (inputs == k).astype(int)
            rprop = regionprops(label_k)
            n_label_k = np.sum(label_k)
            if n_label_k > 30:
                n_hull    = rprop[0].convex_area - n_label_k
                if n_hull/n_label_k > 0.2:
                    # ipdb.set_trace()
                    # radi = np.sqrt((n_label_k)/np.pi)
                    fprint = np.int(np.maximum(2*np.floor(radi/2)+1,3))
                    distance = ndimage.distance_transform_edt(label_k)
                    diff_connectivity = measure.label(label_k)
                    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((fprint,fprint)),
                                        num_peaks_per_label=2, labels=diff_connectivity, )
                    markers = ndimage.label(local_maxi)[0]
                    label_k = morph.watershed(-distance, markers, mask=label_k)

                unique, counts = np.unique(label_k, return_counts=True)
                # print(unique)
                for u in unique[np.arange(len(unique))!=0]:
                    # corrected = corrected + (label_k == u).astype(int)*current_label
                    corrected[label_k == u] = current_label
                    #print(current_label)
                    current_label += 1

        # print(current_label)
        inputs = corrected

        #plt.figure(20),
        #plt.subplot(1,2,1)
        #plt.imshow(inputs), 

        #plt.subplot(1,2,2)
        #plt.imshow(corrected)
        #plt.show()
        #ipdb.set_trace()

    else:
        # multi-labels from binary input
        inputs =  measure.label(inputs)

    if targets is None:
        return inputs

    # number of labels in ground truth
    labels = np.unique(targets).astype('int32')
    labels_inputs = np.unique(inputs).astype('int32')
    matched_labels = np.zeros((len(labels)-1,2))
    IoU = np.zeros(len(labels)-1)
    # plot ground truth and predicted labels
    if 0:
        plt.subplot(121)
        plt.imshow(targets)
        plt.subplot(122)
        plt.imshow(inputs)

    for i in range(1,len(labels)):
        # check for predicted objects which overlaps the current ground truth object
        unique, counts = np.unique(inputs[targets==labels[i]], return_counts=True)
        pred_label = unique[np.argmax(counts)]
        matched_labels[i-1] = [labels[i],pred_label]
      
        # plot current cell and associated prediction
        if 0:
            plt.figure(figsize=(15, 6))
            plt.subplot(131)
            plt.imshow((targets==labels[i]).astype('int32'))
            plt.subplot(132)
            plt.imshow((inputs==pred_label).astype('int32'))
            plt.subplot(133)
            plt.imshow((targets==labels[i]).astype('int32') + (inputs==pred_label).astype('int32'))

        # calculate IoU
        if pred_label != 0:
            IoU[i-1] = np.sum((targets==labels[i])*(inputs==pred_label))/np.sum((targets==labels[i])+(inputs==pred_label))
        else:
            IoU[i-1] = -1

    score = 0
    for t in thresholds:
        TP = len(matched_labels[IoU > t, 1])
        FP = len(labels_inputs) - 1 - TP
        FN = len(labels) - 1 - TP
        score += TP/(TP + FP + FN + 10**-10)

    score /= len(thresholds)
    return score, inputs

def plot_all_results(model, dataloader, folder = 'gallery'):
    ''' predicts and plots all scans in dataloader using model '''
    for data in dataloader:
        
        ipdb.set_trace()

        inputs, masks, masks_multiLabel = data
        x_train = torch.autograd.Variable(inputs).cuda()
        output = model(x_train)

        for idx in range(inputs.shape[0]):
            score, labels_pred = competition_loss_func(output[idx,0,:].data.cpu().numpy(),masks_multiLabel[idx,0,:].numpy())
            plotExample(inputs[idx,:], masks[idx,:], labels_pred, epoch, i, lossFunc(output[idx,:].data.cpu(), masks[idx,:]), score, False, folder)


def rle_encoding(x):
    ''' From https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277 '''
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(labels):
    for i in range(1, labels.max() + 1):
        yield rle_encoding(labels == i)




def save_submission_file(sub,filename):
    ''' Save submission file. Follow the name convention "name-0". 
        If the file exists, then "name-1" etc. is the new filename. ''' 
     
    while os.path.exists(filename+'.csv'):
        namesplit = filename.split("-")
        namesplit[-1] =  str( int(namesplit[-1])+1 )
        filename = '-'.join(namesplit) 

    sub.to_csv(filename+'.csv', index=False)

def save_model(model, filename):
    ''' Save model file. Follow the name convention "name-0". 
        If the file exists, then "name-1" etc. is the new filename. '''
    
    while os.path.exists(filename+'.pt'):
        namesplit = filename.split("-")
        namesplit[-1] =  str( int(namesplit[-1])+1 )
        filename = '-'.join(namesplit) 
        
    torch.save(model, filename+'.pt')


def load_model(PATH):
    ''' Load saved model. '''
    
    return torch.load(PATH)

    
