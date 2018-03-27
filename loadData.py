import ipdb
from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t

from torchvision import transforms

import matplotlib.pyplot as plt

from skimage.filters import sobel
import skimage.measure as measure

import pandas as pd
from random import *
import PIL 

# from scipy import signal
from scipy import ndimage

def process(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        imgs = []
        for image in (file/'images').iterdir():
            img = io.imread(image)
            imgs.append(img)
        assert len(imgs)==1
        if img.shape[2]>3:
            assert(img[:,:,3]!=255).sum()==0
        img = img[:,:,:3]

        if has_mask:
            mask_files = list((file/'masks').iterdir())
            masks = None
            for ii,mask in enumerate(mask_files):
                mask = io.imread(mask)
                assert (mask[(mask!=0)]==255).all()
                if masks is None:
                    H,W = mask.shape
                    masks = np.zeros((len(mask_files),H,W))
                masks[ii] = mask
            tmp_mask = masks.sum(0)
            assert (tmp_mask[tmp_mask!=0] == 255).all()
            for ii,mask in enumerate(masks):
                masks[ii] = mask/255 * (ii+1)
            mask = masks.sum(0)
            item['mask'] = t.from_numpy(mask)
        item['name'] = str(file).split('/')[-1]
        item['img'] = t.from_numpy(img)
        datas.append(item)
    return datas


def process_split(file_path, clustermeans, cluster, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
        imgs = []
        for image in (file/'images').iterdir():
            img = io.imread(image)
            # ipdb.set_trace()
            rows, cols, dims = img.shape
            imgmean = np.mean(np.reshape(img.copy(),(rows*cols,dims)), axis=0)
            imgmean.shape = (imgmean.shape[0], 1)
            cdist = np.zeros((clustermeans.shape[0],1)) 
            for k in range(clustermeans.shape[0]):
                cdist[k] = np.sum((clustermeans[k,:] - imgmean[0:3])**2)

            c = np.int( np.argmin( cdist ) )
            if c == cluster:
                imgs.append(img)
            else:
                continue 
            # imgs.append(img)
        
        #ipdb.set_trace()
        if len(imgs) == 1:
            #assert len(imgs)==1
            if img.shape[2]>3:
                if (img[:,:,3]!=255).sum()!=0:
                    print(img)

                assert(img[:,:,3]!=255).sum()==0
            img = img[:,:,:3]

            if has_mask:
                mask_files = list((file/'masks').iterdir())
                masks = None
                for ii,mask in enumerate(mask_files):
                    mask = io.imread(mask)
                    assert (mask[(mask!=0)]==255).all()
                    if masks is None:
                        H,W = mask.shape
                        masks = np.zeros((len(mask_files),H,W))
                    masks[ii] = mask
                tmp_mask = masks.sum(0)
                assert (tmp_mask[tmp_mask!=0] == 255).all()
                for ii,mask in enumerate(masks):
                    masks[ii] = mask/255 * (ii+1)
                mask = masks.sum(0)
                item['mask'] = t.from_numpy(mask)
            item['name'] = str(file).split('/')[-1]
            item['img'] = t.from_numpy(img)
            datas.append(item)
    return datas



def crop_nparray(img, xy):
    return img[xy[1]:xy[3], xy[0]:xy[2], :]


class Dataset():
    def __init__(self,data,source_transform,target_transform,source_target_transform=None,augment=False,imgWidth=256):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform
 
        self.imgWidth = imgWidth
        self.augment = augment
        if self.augment:
            self.st_transform = source_target_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
    
        if self.augment == True:
            imgWidth = self.imgWidth

            # do cropping to imgWidth x imgWidth if *any* dimension is larger than imgWidth
            if np.any(np.asarray(img.shape[0:2]) > imgWidth):                
                # print("Crop")
                #plt.figure(1)
                #plt.subplot(2,2,1)
                #plt.imshow(img)
                #plt.subplot(2,2,3)
                #plt.imshow(mask.squeeze(axis=2))               

                # crop 
                xcoord = randint(0,img.shape[1] - imgWidth )
                ycoord = randint(0,img.shape[0] - imgWidth )
                mask = crop_nparray(mask,(xcoord, ycoord, xcoord+imgWidth, ycoord+imgWidth))
                img  = crop_nparray(img, (xcoord, ycoord, xcoord+imgWidth, ycoord+imgWidth))

                if np.any(np.asarray(img.shape[0:2]) != imgWidth): 
                    print(xcoord)
                    print(ycoord)
                    print(img.shape)
                    print("CROP ERROR")

                #plt.figure(1)
                #plt.subplot(2,2,2)
                #plt.imshow(img)
                #plt.subplot(2,2,4)
                #plt.imshow(mask.squeeze(axis=2))               
                #plt.show()

            raugment = randint(0,100)
            p = 50
         
            if 0 <= raugment <= 33:
                # mirror - left/right
                rint = randint(0, 100)
                if p  > rint:
                    # print("Flip LR")
                    mask = np.fliplr(mask)
                    img  = np.fliplr(img)
    
            elif 33 < raugment <= 66:
                # mirror - up/down
                rint = randint(0, 100)
                if p > rint:
                    # print("Flip UD")
                    mask = np.flipud(mask)
                    img = np.flipud(img)

            elif 66 < raugment <= 100:
                # transpose
                rint = randint(0, 100)
                if p > rint:
                    # print("Transpose")
                    mask = np.transpose(mask, (1,0,2))
                    img = np.transpose(img, (1,0,2))


            # rotate
            # take care of discretization artifacts.
      
  

        img = self.s_transform(img) 
        mask = self.t_transform(mask)*255
        # reassign labels
        mask_ = t.from_numpy(np.zeros_like(mask, dtype=np.float32))
        for i,idx in enumerate(np.unique(mask)):
            mask_[mask==int(idx)] = i
        mask = mask_
        # if there is at least one label
        if mask.sum() > 0:
            # edge mask
            edge_mask = t.from_numpy(sobel(mask[0,:,:]/mask.max()).astype('float32')).unsqueeze(0) # sobel filter
            edge_mask[edge_mask > 0] = 1 # binarize

            # centroids 
            # print(mask)
            centroid_mask = np.zeros((mask[0,:,:].shape)) #.squeeze(axis=2).shape)) 
            rp = measure.regionprops(mask[0,:,:].numpy().astype(int)) #) # .squeeze(axis=2))
            for props in rp:
                y0, x0 = props.centroid
                centroid_mask[np.int(y0), np.int(x0)] = 1

            struct1 = ndimage.generate_binary_structure(2,1)
            centroid_mask = ndimage.binary_dilation(centroid_mask, structure=struct1, iterations=3)*1
            centroid_mask = t.from_numpy(centroid_mask).unsqueeze(0).float()


        else:
            edge_mask = mask.clone()
            centroid_mask = mask.clone()

        #plt.figure(1)
        #plt.subplot(1,2,1)
        #plt.imshow(centroid_mask)
        #plt.subplot(1,2,2)
        #plt.imshow(mask[0,:,:])               
        #plt.show()

        # binary body mask
        mask_binary = mask.clone()
        mask_binary[mask > 0] = 1
        mask_stacked = t.cat((mask_binary,edge_mask,centroid_mask), 0)
      
        return img, mask_stacked, mask
    def __len__(self):
        return len(self.datas)


class TestDataset():
    def __init__(self,path,source_transform):
        self.datas = t.load(path)
        self.s_transform = source_transform
    def __getitem__(self, index):
        data = self.datas[index]
        name = data['name']
        img = data['img'].numpy()
        shape = t.IntTensor([img.shape[0], img.shape[1]])
        img = self.s_transform(img)
        return img, shape, name
    def __len__(self):
        return len(self.datas)


def createKSplits(l, K, random_state = 0):
    ''' createKSplits(l, K): returns a list with K entries, each holding a list of indices that constitute one splits. l is the number of data points '''
    arr = np.arange(l)
    np.random.seed(random_state)
    np.random.shuffle(arr)

    d = int (l/K)
    return [arr[d*i:d*(i+1)] if i < K-1 else arr[d*i:] for i in range(K)]


def readFromDisk(valIdx, path='/export/home/frathke/workspace/kaggle/cellSegmentation/data/train.pth'):
    data = t.load(path)
    # split into validation and training set
    trnIdx = np.setdiff1d(np.arange(len(data)),valIdx)
    
    data_train = [data[i] for i in trnIdx]
    data_val = [data[i] for i in valIdx]

    return data_train, data_val


def main():
   ''' Construct train and test data, can be skipped if already done. '''
   TRAIN_PATH = './data/train.pth'
   TEST_PATH = './data/test.pth'
   test = process('../input/stage1_test/', False)
   t.save(test, TEST_PATH)
   train_data = process('../input/stage1_train/')
   t.save(train_data, TRAIN_PATH)
  

def split_data():
    df = pd.read_csv('class_means.csv', sep=',',header=None, index_col=False)
    classmean = np.genfromtxt('class_means.csv', delimiter=',')[1:,1:] 

    # Class 0 - grayscale
    TEST_PATH = './data/test_class0.pth'
    print('Create test dataset for Class 0 (grayscale)')
    test = process_split('../input/stage1_test/', classmean, 0, False)
    t.save(test, TEST_PATH)
   
    #ipdb.set_trace()
    
    TRAIN_PATH = './data/train_class0.pth'
    print('Create training dataset for Class 0 (grayscale)')
    train_data = process_split('../input/stage1_train/', classmean, 0)
    t.save(train_data, TRAIN_PATH)
    
    # Class 1 - RGB
    TEST_PATH = './data/test_class1.pth'
    print('Create test dataset for Class 1 (RGB)')
    test = process_split('../input/stage1_test/', classmean, 1, False)
    t.save(test, TEST_PATH)
    
    TRAIN_PATH = './data/train_class1.pth'
    print('Create training dataset for Class 1 (RGB)')
    train_data = process_split('../input/stage1_train/', classmean, 1)
    t.save(train_data, TRAIN_PATH)


if __name__ == "__main__":
    # main()
    split_data()
