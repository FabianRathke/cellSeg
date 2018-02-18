from pathlib import Path
from skimage import io
import numpy as np
from tqdm import tqdm
import torch as t
import ipdb

import pandas as pd

def process(file_path, has_mask=True):
    file_path = Path(file_path)
    files = sorted(list(Path(file_path).iterdir()))
    datas = []

    for file in tqdm(files):
        item = {}
       i
       imgs = 
        []
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

class Dataset():
    def __init__(self,data,source_transform,target_transform):
        self.datas = data
#         self.datas = train_data
        self.s_transform = source_transform
        self.t_transform = target_transform

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()
    
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
        mask_binary = mask.clone()
        mask_binary[mask > 0] = 1
        return img, mask_binary, mask
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




