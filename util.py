import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from skimage import measure
from skimage.transform import resize
from scipy import ndimage
import skimage.morphology as morph
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
import ipdb
import os.path
from skimage.measure import regionprops
from IPython import embed
from models import *

# ****** LOSS FUNCTION ******
def soft_dice_loss(inputs, targets):
    num = targets.size(0) # number of batches
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


def cross_entropy(inputs, targets):
    ''' Cross Entropy Loss

        Targets holds binary masks for all classes and a weighting mask in the last layer '''

    numClasses = targets.size(1)-1
    # cross entropy w_{pixel}*\sum_{classes} target(pixel, class) * log(inputs(pixel, class))
    score = -t.sum(t.sum(t.log(inputs)*targets[:,0:numClasses,:],dim=1)*targets[:,-1,:,:])
    #ipdb.set_trace()
    return score/targets.size(0)


def plotExample(img, mask, mask_multi, pred, labels_pred, epoch, batch, loss, lossComp, interactive=False, folder = 'plots'):
    ''' Plots ground truth and prediction for a cell image '''
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_under([0.8, 0.8, 0.8])
    if interactive:
        plt.ion()
    plt.figure(figsize=(20, 10))
    plt.subplot(241)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5); plt.axis('off')
    plt.subplot(242)
    plt.imshow(mask[0,:]); plt.axis('off')
    plt.subplot(243)
    plt.imshow(mask[1,:]); plt.axis('off')
    plt.subplot(244)
    vmin = .001 if mask_multi.max() > 0 else 0
    plt.imshow(mask_multi, interpolation = 'none', cmap = cmap, vmin=vmin); plt.axis('off')
    plt.subplot(245)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5); plt.axis('off')
    plt.subplot(246)
    plt.imshow(pred[0,:]); plt.axis('off')
    plt.subplot(247)
    plt.imshow(pred[1,:]); plt.axis('off')
    plt.subplot(248)
    vmin = .001 if labels_pred.max() > 0 else 0
    plt.imshow(labels_pred, cmap = cmap, interpolation='none', vmin=vmin); plt.axis('off'); #plt.colorbar()
    plt.suptitle("Epoch {} and batch {}: Loss = {:.2f}, CompEval = {:.2f}".format(epoch, batch, loss, lossComp))
    if interactive:
        plt.pause(0.05)
        plt.draw()
    else:
        plt.savefig("{}/epoch_{}_batch_{}.png".format(folder,epoch,batch))
        plt.close()

def plotExampleTest(img, mask, pred, batch, fileSize, folder = 'plots'):
    ''' Plots ground truth and prediction for a cell image '''
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_under([0.8, 0.8, 0.8])
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5); plt.axis('off')
    plt.subplot(142)
    plt.imshow(mask[0,:]); plt.axis('off')
    plt.subplot(143)
    plt.imshow(mask[1,:]); plt.axis('off')
    plt.subplot(144)
    vmin = .001 if pred.max() > 0 else 0
    plt.imshow(pred, interpolation = 'none', cmap = cmap, vmin=vmin); plt.axis('off')
    plt.suptitle("id: {} ({:d} x {:d})".format(batch, fileSize[0], fileSize[1]))
    plt.savefig("{}/id_{}.png".format(folder,batch))
    plt.close()


def competition_loss_func(inputs, targets = None):
    ''' https://www.kaggle.com/c/data-science-bowl-2018#evaluation
        evaluates IoU (intersection over union) for various thresholds and calculates
        TPs, FPs and FNs for all objects (detected vs. ground truth). The final score
        is averaged over all thresholds. '''

    data_inputs = inputs.copy()
    thresholds = np.arange(0.5,1,0.05)
 
    if inputs.shape[0] > 1:
        if inputs.shape[0] > 2:
            diff = inputs[2,:]-inputs[1,:]
        else:
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
        for k in unique[1:]:
            label_k = (inputs == k).astype(int)
            rprop = regionprops(label_k)
            n_label_k = np.sum(label_k)
            if n_label_k > 30: # filter all cells smaller than 30 pixel
                n_hull = rprop[0].convex_area - n_label_k # size convex hull - actual predicted cell size 
                
                if n_hull/n_label_k > 0.2:
                    # radi = np.sqrt((n_label_k)/np.pi)
                    fprint = np.int(np.maximum(2*np.floor(radi/2)+1,3))
                    distance = ndimage.distance_transform_edt(label_k)
                    diff_connectivity = measure.label(label_k) # why not label_k ?? [Fabian]
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
        if 0 and np.sum(inputs-corrected):
            plt.figure(20),
            plt.subplot(1,2,1)
            plt.imshow(inputs), 

            plt.subplot(1,2,2)
            plt.imshow(inputs-corrected)
            plt.show(block=False)
            #ipdb.set_trace()

        inputs = corrected
    else:
        # multi-labels from binary input
        inputs = measure.label(inputs)

    # randomly shuffle labels for better visibility during plotting
    labels = np.unique(inputs)[1:]
    np.random.shuffle(labels)
    labels = np.concatenate((np.array([0]), labels))
    inputs = labels[inputs]

    if targets is None:
        return inputs

    # number of labels in ground truth
    labels = np.unique(targets).astype('int32')
    # number of predicted labels
    labels_inputs = np.unique(inputs).astype('int32')
    # keep track of the matching between ground truth and predicted labels
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
        # remove background as label
        if len(unique) > 1:
            counts = counts[unique!=0]
            unique = unique[unique!=0]
        pred_label = unique[np.argmax(counts)] # get the object with the largets overlap
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
    allScores = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        TP = len(matched_labels[IoU > t, 1])
        FP = len(labels_inputs) - 1 - TP
        FN = len(labels) - 1 - TP
        allScores[i] = TP/(TP + FP + FN + 10**-10)

    score = np.sum(allScores)/len(thresholds)
    # save all information in dict for debugging
    info = {"gt": targets, "pred": inputs, "matched_labels": matched_labels, "IoU": IoU, "allScores": allScores}

    return score, inputs, info

def plotLabels(img, info, epoch, batch, score, folder):
    ''' Plot labels in different colors, depending on being TP, FP, or FN'''
    cmap = matplotlib.cm.get_cmap('viridis')
    cmap.set_under([0.8, 0.8, 0.8])

    cmap1 = np.matlib.repmat(np.array([0, 0, 1, 1]),64,1) # blue
    cmap2 = np.matlib.repmat(np.array([1, 0, 0, 1]),64,1) # red
    cmap3 = plt.cm.RdYlGn(np.linspace(0., 1, 128))
    cmap3[0] = [0.8, 0.8, 0.8, 1]
    colors = np.vstack((cmap1, cmap2, cmap3))
    mymap = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    plt.figure(figsize=(22, 6))
    # plot Image
    plt.subplot(141)
    plt.imshow(img.permute(1, 2, 0) * 0.5 + 0.5); plt.axis('off')
    # plot ground truth
    plt.subplot(142)
    vmin = .001 if np.max(info['gt']) > 0 else 0
    plt.imshow(info['gt'], cmap = cmap, vmin = vmin); plt.axis('off'); plt.gca().set_title('Ground Truth')
    plt.subplot(143)
    vmin = .001 if np.max(info['pred']) > 0 else 0
    plt.imshow(info['pred'], cmap = cmap, vmin = vmin); plt.axis('off'); plt.gca().set_title('Prediction')
    plt.subplot(144);
    if info['pred'].max() > 0:
        # prepare data to plot
        pred = info['pred']
        plot = np.zeros_like(pred, dtype=float)
        FP = np.setdiff1d(np.unique(pred), info['matched_labels'][:,1])
        FP = FP[FP > 0]
        for i in FP:
            plot[pred==i] = -1

        FN = info['matched_labels'][info['matched_labels'][:,1] == 0, 0]
        for i in FN:
            plot[info['gt']==i] = -0.5

        TP = info['matched_labels'][info['IoU'] > 0, 0]
        for i in TP:
            plot[pred == info['matched_labels'][int(i)-1][1]] = info['IoU'][int(i-1)]
        plt.imshow(plot, interpolation = 'none', cmap = mymap, vmin = -1, vmax = 1); plt.axis('off'); plt.gca().set_title('Quality Prediction')
    else:
        plt.imshow(info['pred'], cmap = cmap)
   
    #plt.show(block=False)
    plt.suptitle("Epoch {} and batch {}: Competition-Loss = {:.2f}".format(epoch, batch, score))
    plt.savefig("{}/epoch_{}_batch_{}_quality.png".format(folder,epoch,batch))
    plt.close()


def plot_all_predictions(model, dataloader, folder = 'plots/gallery'):
    ''' predicts and plots all scans in dataloader using model '''
    for i, data in enumerate(dataloader):
        print(i)
        inputs, masks, masks_multiLabel = data
        x_train = torch.autograd.Variable(inputs).cuda()
        output = model(x_train)
        ipdb.set_trace()
    
        for idx in range(inputs.shape[0]):
            score, labels_pred, info = competition_loss_func(output[idx,:].data.cpu().numpy(), masks_multiLabel[idx,0,:].numpy())
            plotExample(inputs[idx,:], masks[idx,:],masks_multiLabel[idx,0,:], output.data[idx,:], labels_pred, i, idx, 0, score, False, folder)
            plotLabels(inputs[idx,:], info, i, idx, score, folder)


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


# ***** TRAIN *****

def evaluate_model(model, lossFunc, validdataloader):
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
            score, _, _ = competition_loss_func(output[j,:].data.cpu().numpy(),masks_multiLabel[j,0,:].numpy())
            running_score += score

    return (1.0-running_accuracy/(i+1.0)), running_score/len(validdataloader.dataset)


def train_model(model, optimizer, lossFunc, dataloader, validdataloader, args):
    for epoch in range(args.numEpochs):
        running_loss = 0; running_loss_comp = 0
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
#            if epoch > 5:
#                for j in range(inputs.shape[0]): # evalute competition loss function
#                    score, _, _  = competition_loss_func(output[j,:].data.cpu().numpy(),masks_multiLabel[j,0,:].numpy())
#                    running_loss_comp += score

            if i % args.iterPrint == args.iterPrint-1:    # print every iterPrint mini-batch
                print('[%d, %5d] loss: %.3f (score: %.3f)' %
                (epoch + 1, i + 1, running_loss / args.iterPrint, running_loss_comp/args.iterPrint))
                running_loss = 0.0; running_loss_comp = .0

            # plot some segmented training examples
            if 0 and i % args.iterPlot == args.iterPlot-1:
                idx = 0
                #ipdb.set_trace()
                score, _ = competition_loss_func(output[idx,0,:].data.cpu().numpy(),masks_multiLabel[idx,0,:].numpy())
                plotExample(inputs[idx,:], masks[idx,0,:,:], output[idx,0,:,:].data, epoch, i, lossFunc(output[idx,:].data.cpu(), masks[idx,:]), score, False)

        #if validdataloader and epoch > 5:
            #acc, score = evaluate_model(model, lossFunc, validdataloader)
            #print('acc: %.3f, score: %.3f' % (acc, score))

    return model
