from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t
import ipdb


from torchvision import transforms

import matplotlib.pyplot as plt

from skimage.filters import sobel

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


def crop_nparray(img, xy):
    return img[xy[1]:xy[3], xy[0]:xy[2], :]


class Dataset():
    def __init__(self,data,source_transform,target_transform,source_target_transform=None,augment=False):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform
  
        self.augment = augment
        if self.augment:
            self.st_transform = source_target_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()

        
        if self.augment == True:

            if np.all(np.asarray(img.shape) > 256):
                
                print(mask.shape)
                plt.figure(1)
                plt.subplot(2,2,1)
                plt.imshow(img)
                plt.subplot(2,2,3)
                plt.imshow(mask.squeeze(axis=2))               

                # randomly crop data and mask
                xcoord = randint(0,img.shape[0] - 256 + 1)
                ycoord = randint(0,img.shape[1] - 256 + 1)
                mask = crop_nparray(mask,(xcoord, ycoord, xcoord+256, ycoord+256))
                img  = crop_nparray(img, (xcoord, ycoord, xcoord+256, ycoord+256))


                # mirror

                # transposei

                # rotate


            else:
                toPIL = transforms.ToPILImage()
                img = toPIL(img)
                mask = toPIL(mask)

                img = self.st_transform(img)
                mask = self.st_transform(mask)
         

            
        #plt.subplot(2,2,2)
        #plt.imshow(img)
        #plt.subplot(2,2,4)
        #plt.imshow(mask.squeeze(axis=2))
        #print(mask.shape)
        #plt.show()
        # else:
        # reshape to 256x256

            # Rotation
            # rot  = randint(-20,20)
            # img  = img.rotate(rot)
            # mask = mask.rotate(rot, resample=PIL.Image.NEAREST)

        # Class specific normalization  
        # df = pd.read_csv('class_means.csv', sep=',',header=None, index_col=False)
        # classmean = np.genfromtxt('class_means.csv', delimiter=',')[1:,1:] 
        # rows, cols, dims = img.shape
        # imgmean = np.mean(np.reshape(img,(rows*cols,dims)), axis=0) 
        # c = np.argmin( np.sum((classmean - imgmean)**2, axis=1) )
        # img = img - classmean[c,:]
        # img.dtype = np.uint8
       
        img = self.s_transform(img) 
        mask = self.t_transform(mask)*255
        # if there is at least one label
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
        #mask_binary = mask_binary - edge_mask
        #mask_binary[mask_binary < 0] = 0
        # stack resulting masks
        mask_stacked = t.cat((mask_binary,edge_mask))
        #mask_stacked = edge_mask
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


def createKSplits(l, K, random_state=None):
    ''' createKSplits(l, K): returns a list with K entries, each holding a list of indices that constitute one splits. l is the number of data points '''
    arr = np.arange(l)
    if not random_state:
        np.random.seed()
    else:
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
  

if __name__ == "__main__":
    main()




