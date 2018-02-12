import numpy as np
import matplotlib.pyplot as plt
import torch

# ****** LOSS FUNCTION ******
def soft_dice_loss(inputs, targets):
    num = targets.size(0)
    m1  = inputs.view(num,-1)
    m2  = targets.view(num,-1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
    score = 1 - score.sum()/num
    return score


def plotExample(img, mask, pred, epoch, batch, loss, interactive=False):
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
    plt.suptitle("Epoch {} and batch {}: Loss = {:.2f}".format(epoch, batch, loss))
    if interactive:
        plt.pause(0.05)
        plt.draw()
    else:
        plt.savefig("plots/epoch_{}_batch_{}.png".format(epoch,batch))
        plt.close()


#def competition_loss_func(inputs,targets)
#    ''' https://www.kaggle.com/c/data-science-bowl-2018#evaluation
#        evaluates IoU (intersection over union) for various thresholds and calculates
#        TPs, FPs and FNs for all objects (detected and ground truth). The final score
#        is averaged over all thresholds. '''
#    
#    thresholds = np.arange(0.5,1,0.05)
#    
#    # number of labels in ground truth
#    labels = np.unique(targets).astype('int32')
#    matched_labels = []
#    IoU = []
#
#    for i in range(1,len(labels)):
#        # check for predicted objects which overlaps the current ground truth object
#        unique, counts = np.unique(inputs[targets==labels[i]], return_counts=True)
#        pred_label = unique[np.argmax(counts)]
#        matched_labels.append([labels[i],pred_label])
#        
#        #plt.imshow((targets==labels[i]).astype('int32') + (inputs==pred_label).astype('int32'))
#
#        # calculate IoU
#        if pred_label != 0:
#            IoU.append(np.sum((targets==labels[i])*(inputs==pred_label))/(np.sum(targets==labels[i])+np.sum(inputs==pred_label)))
#        else:
#            IoU.append(-1)
