import scipy.ndimage.morphology as morph
import numpy as np

def fill_holes(labels):
    labels_filled = np.zeros_like(labels)
    idx = np.unique(labels)
    for j in idx[1:]:
        labels_j = np.zeros_like(labels)
        labels_j[labels==j] = 1
        labels_j = morph.binary_fill_holes(labels_j)
        labels_filled[labels_j] = j

    return labels_filled

