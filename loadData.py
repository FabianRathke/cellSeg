import ipdb
from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t
import sys

from torchvision import transforms

import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle
from skimage import exposure
from skimage.filters import sobel
from skimage.morphology import square, dilation
from scipy import ndimage

import util
import pandas as pd
from random import *
import PIL 

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
    def __init__(self,data, source_transform, target_transform, source_target_transform=None, augment=False, normalize=False, imgWidth=256, maskConf = [1, 0, 0]):
        self.s_transform = source_transform
        self.t_transform = target_transform
 
        self.imgWidth = imgWidth
        self.augment = augment

        if self.augment:
            self.st_transform = source_target_transform

        self.names = ['mask_binary', 'edge_mask', 'mask_bindary_diff']
        self.maskConf = maskConf
        
        if sum(maskConf) > 0:
            print("Using masks {}".format(", ".join([name for i,name in enumerate(self.names) if maskConf[i]])))
        else:
            print("Use 3 class mask (background, cells, boundary between cells)")

        if normalize:
            print("Perform histogram equalization")
            for i in range(len(data)):
                sys.stdout.write("\r" + str(i))
                data[i]['img'] = equalHist(data[i]['img'])
       
        self.datas = data

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
    
        if self.augment == True:
            imgWidth = self.imgWidth

            # do cropping to imgWidth x imgWidth if *any* dimension is larger than imgWidth
            if np.any(np.asarray(img.shape[0:2]) > imgWidth):                
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

            p = 50
            # mirror - left/right
            rint = randint(0, 100)
            if p  > rint:
                # print("Flip LR")
                mask = np.fliplr(mask)
                img  = np.fliplr(img)

            # mirror - up/down
            rint = randint(0, 100)
            if p > rint:
                # print("Flip UD")
                mask = np.flipud(mask)
                img = np.flipud(img)

            # transpose
            rint = randint(0, 100)
            if p > rint:
                # print("Transpose")
                mask = np.transpose(mask, (1,0,2))
                img = np.transpose(img, (1,0,2))
            
            # rotate
    
        img = self.s_transform(img) 
        mask = self.t_transform(mask)*255
        # reassign labels for filling holes that appeared during the cropping --> makes life easier in later functions
        mask_ = t.from_numpy(np.zeros_like(mask, dtype=np.float32))
        for i,idx in enumerate(np.unique(mask)):
            mask_[mask==int(idx)] = i
        mask = mask_
        mask_stacked = makeMask(mask, self.maskConf)

        return img, mask_stacked, mask
    
    def __len__(self):
        return len(self.datas)


def makeMask(mask, maskConf):
    if sum(maskConf) > 0:
        # if there is at least one cell
        if mask.sum() > 0:
            # edge mask
            edge_mask = t.from_numpy(sobel(mask[0,:,:]/mask.max()).astype('float32')).unsqueeze(0) # sobel filter
            edge_mask[edge_mask > 0] = 1 # binarize
        else:
            edge_mask = mask.clone()

        # binary body mask
        mask_binary = mask.clone()
        mask_binary[mask > 0] = 1
        # substract edges from mask
        mask_binary_diff = mask_binary - edge_mask
        mask_binary_diff[mask_binary < 0] = 0
        # stack resulting masks
        mask_stacked = eval("t.cat((" + ", ".join([name for i,name in enumerate(names) if maskConf[i]]) + "))")
    else:
        # three classes encoded as one-hot labels in three binary layers
        mask = mask[0,:].numpy()
        overlap = np.zeros_like(mask)
        for idx in np.unique(mask):
            if idx > 0:
                overlap += dilation(mask==idx, square(3))

        background = np.zeros_like(mask)
        background[mask == 0] = 1
        foreground = np.zeros_like(mask)
        foreground[mask > 0] = 1
        
        # add weight layer
        weight_dw = 1+1e-3-np.minimum(ndimage.distance_transform_edt(background)/15, 1) 
        overlap[overlap == 1] = 0
        # remove overlap from foreground mask
        foreground[overlap == 1] = 0
        background[overlap == 1] = 0
        # add 1 to foreground class to prevent division by zero, since we have images without any cells
        weight = (overlap/(np.sum(overlap)+500) + foreground/(np.sum(foreground)+1) + background/np.sum(background))*weight_dw
        #weight = weight*np.prod(mask.shape)/np.sum(weight)
        #ipdb.set_trace()

        mask_stacked = t.from_numpy(np.stack((background, foreground, overlap, weight))).type(t.FloatTensor)

    return mask_stacked


class TestDataset():
    def __init__(self,path,source_transform,normalize):
        self.datas = t.load(path)
        if normalize:
            print("Perform histogram equalization")
            for i in range(len(self.datas)):
                sys.stdout.write("\r" + str(i))
                self.datas[i]['img'] = equalHist(self.datas[i]['img'])
	    
            print("")	

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


def equalHist(img):
    img = exposure.equalize_adapthist(img.numpy())
    #img = denoise_tv_chambolle(img, weight=0.02, multichannel=False)
    return t.from_numpy(img*255).type(t.ByteTensor)


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
