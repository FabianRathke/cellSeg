
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import imageio

from skimage import exposure
import ipdb

# List files in directory

datasetPath = '../input/stage1_train/'

trainDirNames = next(os.walk(datasetPath))[1]

ntrain = len(trainDirNames)
# ntrain = 50
# ipdb.set_trace()

meanmat = np.zeros((ntrain,3))
stdmat  = np.zeros((ntrain,3))

meanmat_gray = np.zeros((ntrain,3))
stdmat_gray  = np.zeros((ntrain,3))

meanmat_color = np.zeros((ntrain,3))
stdmat_color  = np.zeros((ntrain,3))

labelColor = [] #np.zeros((ntrain,1))
imageColor = []

def press(event):
    print('press', event.key)
    
    classnum = int(event.key)
    if classnum == 0 or classnum == 1:
        labelColor.append(classnum)

    sys.stdout.flush()
    if event.key == 'x':
        visible = xl.get_visible()
        xl.set_visible(not visible)
        fig.canvas.draw()
    
    plt.close(fig)

c = 0
g = 0
for k in range(ntrain):
    
    name  = trainDirNames[k]
    path = datasetPath + name + '/images/' + name + '.png'

    image = imageio.imread(path)[:,:,0:3] 
    # image =  (exposure.equalize_adapthist(np.array(image))*255).astype(np.uint8)

    rows, cols, dims = image.shape

    # Gray 
    if np.sum(np.std(image, axis=2)) == 0.0:
        meanmat_gray[g,:] = np.mean(np.reshape(image,(rows*cols,dims)), axis=0)
        stdmat_gray[g,:] = np.std(np.reshape(image,(rows*cols,dims)), axis=0)
        g = g + 1
    else:

        # ipdb.set_trace()
        meanmat_color[c,:] = np.mean(np.reshape(image,(rows*cols,dims)), axis=0)
        stdmat_color[c,:] = np.std(np.reshape(image,(rows*cols,dims)), axis=0)
        
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect('key_press_event', press)
        ax.imshow(image)
        plt.show()    

        imageColor.append((image, labelColor[c]))
        
        print(labelColor[c])
        c = c+1
  
ipdb.set_trace()
import pandas as pd
df = pd.DataFrame(imageColor)
df.to_csv("imageColor.csv")

#############################################################################

meanmat_gray = meanmat_gray[0:g,:]
meanmat_color = meanmat_color[0:c,:] 

stdmat_gray = stdmat_gray[0:g,:]
stdmat_color = stdmat_color[0:c,:]

#meanmat[0,:] = np.mean(meanmat_gray, axis=0)

#print(meanmat)
#print(stdmat)

ipdb.set_trace()

#setmean = np.mean(meanmat,axis=0)
#setstd  = np.std(stdmat,axis=0)

#print("Dataset mean")
#print(setmean)
#print(setmean/255)
#print("Dataset std")
#print(setstd)
#print(setstd/255)


# ***** KMEANS CLUSTERING *****
from sklearn.cluster import KMeans

# X = meanmat_color # [datapoints X features]
X = np.concatenate( (meanmat_color, stdmat_color), axis=1)

n_clusters = 2
y_pred = KMeans(n_clusters=n_clusters).fit_predict(X)
#def equalHist(img):
#    img = exposure.equalize_adapthist(img)
#    #img = denoise_tv_chambolle(img, weight=0.02, multichannel=False)
#    return 

nClusters = np.zeros((n_clusters))
for k in range(n_clusters):
    nClusters[k] = np.sum(y_pred == k)


# Plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(2)
ax = Axes3D(fig, elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y_pred, edgecolor='k')


ndisp = 5
csorted = y_pred.argsort()
plt.figure(3)
for k in range(1,ndisp+1):
    if y_pred[csorted[k]] == 0:
        plt.subplot(2,ndisp,k)
        name  = trainDirNames[csorted[k]]
        path = datasetPath + name + '/images/' + name + '.png'
        image = imageio.imread(path)
        plt.imshow(image)

    if y_pred[csorted[-k]] == 1:
        plt.subplot(2,ndisp,k+ndisp)
        name  = trainDirNames[csorted[-k]]
        path = datasetPath + name + '/images/' + name + '.png'
        image = imageio.imread(path)
        plt.imshow(image)


classmean_color = np.zeros((n_clusters,3))
classn          = np.zeros((n_clusters))

# ipdb.set_trace()

#for i in range(meanmat_color.shape[0]):
#	for c in range(n_clusters):
#		if y_pred[i] == c:
#			classmean_color[c,:] += meanmat_color[i,:]
#			classn[c] += 1
	
for c in range(n_clusters):
    classmean_color[c,:] = np.mean(meanmat_color[y_pred == c,:], axis=0)

#classmean_color = classmean_color / (classn[:,None])

classmean_gray = np.mean(meanmat_gray, axis=0)


classmean = np.zeros((n_clusters+1,3))
classmean[0,:] = classmean_gray
classmean[1:,:] = classmean_color



ipdb.set_trace()
plt.show()

import pandas as pd 
df = pd.DataFrame(classmean)
df.to_csv("class3_means.csv")



plt.show()

# ***** SPLIT DATASET  *****
#
# targetpath = '../input_split/'
#
#if not os.path.exists(targetpath):
#    os.mkdir(targetpath)
#    for k in range(n_clusters):
#        os.mkdir(targetpath + 'set' + np.str(k))
#        os.mkdir(targetpath + 'set' + np.str(k) + '/stage1_train/')
#
#import errno
#import shutil 
#
#def copy(src, dest):
#    try:
#        shutil.rmtree(dest)
#        shutil.copytree(src, dest)
#    except OSError as e:
#        # If the error was caused because the source wasn't a directory
#        if e.errno == errno.ENOTDIR:
#            shutil.copy(src, dest)
#        else:
#            print('Directory not copied. Error: %s' % e)
#
#
#
#for k in range(n_clusters):
#    idx = csorted[y_pred == k]
#    for i in idx:
#        cp_to   = targetpath + 'set' + np.str(k) + '/stage1_train/' + trainDirNames[i] 
#
#        cp_from = datasetPath + trainDirNames[i]
#        
#        ipdb.set_trace()
#
#       copy(cp_from, cp_to)




#
# plt.figure(1)
# plt.imshow(image)
# plt.show()

# plt.figure(1)
# plt.scatter(X[:, 0], X[:, 1], c=y_pred)




































