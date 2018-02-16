
import os
import numpy as np

import matplotlib.pyplot as plt
import imageio


import ipdb

# List files in directory

datasetPath = '../input/stage1_train/'

trainDirNames = next(os.walk(datasetPath))[1]

ntrain = len(trainDirNames)
# ntrain = 100

meanmat = np.zeros((ntrain,4))
stdmat  = np.zeros((ntrain,4))
for k in range(ntrain):
    
    name  = trainDirNames[k]
    path = datasetPath + name + '/images/' + name + '.png'

    image = imageio.imread(path)

    # ipdp.set_trace()


    rows, cols, dims = image.shape
    meanmat[k,:] = np.mean(np.reshape(image,(rows*cols,dims)), axis=0)
    stdmat[k,:] = np.std(np.reshape(image,(rows*cols,dims)), axis=0)


print(meanmat)
print(stdmat)

setmean = np.mean(meanmat,axis=0)
setstd  = np.std(stdmat,axis=0)

print("Dataset mean")
print(setmean)
print(setmean/255)
print("Dataset std")
print(setstd)
print(setstd/255)


# ***** KMEANS CLUSTERING *****
from sklearn.cluster import KMeans

X = meanmat # [datapoints X features]

n_clusters = 2
y_pred = KMeans(n_clusters=n_clusters).fit_predict(X)

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


classmean = np.zeros((n_clusters,4))
classn    = np.zeros((n_clusters))



for i in range(ntrain):
	for c in range(n_clusters):
		if y_pred[i] == c:
			classmean[c,:] += meanmat[i,:]
			classn[c] += 1
	
classmean = classmean / (classn[:,None])


ipdb.set_trace()

import pandas as pd 
df = pd.DataFrame(classmean[:,:-1])
df.to_csv("class_means.csv")



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




































