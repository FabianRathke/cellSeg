import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage import measure
from skimage.transform import resize
from scipy import ndimage
import skimage.morphology as morph

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
    plt.subplot(1433)
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
    else:
        # multi-labels from binary input
        inputs =  measure.label(inputs)

    if targets == None:
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
        inputs, masks, masks_multiLabel = data
        x_train = torch.autograd.Variable(inputs).cuda()
        output = model(x_train)

        for idx in range(inputs.shape[0]):
            score, labels_pred = util.competition_loss_func(output[idx,0,:].data.cpu().numpy(),masks_multiLabel[idx,0,:].numpy())
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
    
    import os.path 
    while os.path.exists(filename+'.csv'):
        namesplit = filename.split("-")
        namesplit[-1] =  str( int(namesplit[-1])+1 )
        filename = '-'.join(namesplit) 

    sub.to_csv(filename+'.csv', index=False)


