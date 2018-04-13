import scipy.ndimage.morphology as morph
import skimage.morphology as morph2
import numpy as np
from scipy import ndimage
from skimage import measure
import ipdb

def fill_holes(labels):
    labels_filled = np.zeros_like(labels)
    idx = np.unique(labels)
    for j in idx[1:]:
        labels_j = np.zeros_like(labels)
        labels_j[labels==j] = 1
        labels_j = morph.binary_fill_holes(labels_j)
        labels_filled[labels_j] = j

    return labels_filled

def estimate_cell_size(labels):
    _, counts = np.unique(labels, return_counts=True)
    return np.median(counts[1:])

def post_splitting(labels, item_splits, threshold = 5):
    labels_new = labels.copy()
    
    _, counts = np.unique(labels, return_counts = True)
    sorted_list = sorted(list(zip(range(len(counts)),counts)), key=lambda x: x[1])
    tmp = np.where(labels==sorted_list[-2][0])

    fiberLabel = -1
    if (sorted_list[-2][1]-sorted_list[-3][1])/np.median(counts) > 5:
        if ((tmp[0].min() == 0 or tmp[0].max() == labels.shape[0]-1) and (tmp[1].min() == 0 or tmp[1].max() == labels.shape[1]-1)) or (tmp[0].min() == 0 and tmp[0].max() == labels.shape[0]-1) or (tmp[1].min() == 0 and tmp[1].max() ==  labels.shape[1]-1):
            print("Fiber detected")
            #labels[labels==sorted_list[-2][0]] = 0
            fiberLabel = sorted_list[-2][0]

    splits = np.zeros_like(labels).astype(np.float);
    new_label_counter = np.unique(labels)[-1]
    bounds = item_splits[2,:].numpy().copy()

    for j in np.unique(labels)[1:]:
        if j != fiberLabel:
            test = labels==j;
            test2 = bounds.copy()
            #test2[test2 < 0.1] = 0
            test2[test==False] = 0
            test3 = test-test2;
            test3[test3<0.55] = 0
            test3[test3>0.55] = 1

            all_labels = measure.label(test3)
            _, counts = np.unique(all_labels, return_counts=True)
            for k, count in enumerate(counts[1:]):
                if count < 30:
                    all_labels[all_labels==k] = 0

            if len(np.unique(all_labels)) > 2:
                addLabel = True
                if len(np.unique(all_labels)) == 3:
                    _, counts = np.unique(all_labels, return_counts = True)
                    counts.sort()
                    if counts[1]/counts[0] > threshold:
                        addLabel = False

                if len(np.unique(all_labels)) == 4:
                    _, counts = np.unique(all_labels, return_counts = True)
                    counts.sort()
                    if counts[2]/counts[0] > threshold:
                        addLabel = False

                if addLabel:
                    inputs = morph2.watershed(-ndimage.distance_transform_edt(test3), all_labels, mask=test)
                    splits += inputs
                    labels_new[labels_new==j] = 0
                    to_add = inputs+new_label_counter
                    to_add[to_add == new_label_counter] = 0
                    labels_new += to_add
                    new_label_counter += len(np.unique(all_labels))-1
                    print("Split cell {} in {}".format(j,len(np.unique(all_labels))-1))


    # relabel
    mask_ = np.zeros_like(labels_new)
    for i,idx in enumerate(np.unique(labels_new)):
        mask_[labels_new==int(idx)] = i

    return mask_
