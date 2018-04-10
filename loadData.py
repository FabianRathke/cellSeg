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
import skimage.measure as measure

import util
import pandas as pd
from random import *
import PIL

from skimage import exposure
from skimage.measure import regionprops

from multiprocessing import Pool

# from scipy import signal
from scipy import ndimage

from skimage.color import rgb2lab
from collections import Counter

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

def applyHistEq(img):
    img['img'] = equalHist(img['img'])
    return img

def applyColorConv(img):
    img['img'] = colorspaceConversion(img['img'])
    return img


class Dataset():
    def __init__(self,data, source_transform, target_transform, source_target_transform=None, augment=False, histEq=False, imgWidth=256, maskConf = [1, 1, 0], use_centroid=0, scaleEq=False, cielab=False):
        self.datas = data
        self.useCentroid = use_centroid
        self.hist_eq = histEq

        self.s_transform = source_transform
        self.t_transform = target_transform
 
        self.imgWidth = imgWidth
        self.augment = augment

        if self.augment:
            self.st_transform = source_target_transform

        self.names = ['mask_binary', 'edge_mask', 'centroid_mask']
        self.maskConf = maskConf
        
        if sum(maskConf) > 0:
            print("Using masks: {}".format(", ".join([name for i,name in enumerate(self.names) if maskConf[i]])))
        else:
            print("Use 3 class mask (background, cells, boundary between cells)")
        
        #pool = Pool(8)

        if histEq:
            print("Perform histogram equalization")
            #data = pool.map(applyHistEq, data)
            for i in range(len(data)):
                sys.stdout.write("\r" + str(i))
                data[i]['img'] = equalHist(data[i]['img'])
            sys.stdout.write("\n")
       
        if cielab:    
            print("Perform color space conversion")
            #data = pool.map(applyColorConv, data)
            for i in range(len(data)):           
                sys.stdout.write("\r" + str(i))
                data[i]['img'] = colorspaceConversion(data[i]['img'])
            sys.stdout.write("\n")

                #d2 = rgb2lab(data[i]['img'])
                # remapp to 8-bit
                #d2[:,:,0] = d2[:,:,0]*(255/100)
                #data[i]['img'] = t.ByteTensor(d2 + [0,128,128])
                    
        #pool.close()       
        if scaleEq and len(data) > 0:
            #ipdb.set_trace()
            print("Perform cell size normalization")            

            # Compute average size of all nuclei
            mean_size = np.zeros((1,2))
            for i in range(len(data)):
                sys.stdout.write("\r" + str(i))             
                rp = regionprops(data[i]['mask'].numpy().astype(int))
                mean_i = np.abs([np.array([r.bbox[1]-r.bbox[3], r.bbox[0]-r.bbox[2]]) for r in rp]).mean(axis=0)
                mean_size += mean_i
            mean_size /= len(data)
            sys.stdout.write("\n")
            print(mean_size)

            for i in range(len(data)): 
                #ipdb.set_trace()
                #plt.figure(1), plt.subplot(1,2,1), plt.imshow(data[i]['mask'])
                
                rp = regionprops(data[i]['mask'].numpy().astype(int))
                mean_i = np.abs([np.array([r.bbox[1]-r.bbox[3], r.bbox[0]-r.bbox[2]]) for r in rp]).mean(axis=0)
                #print("before - after")
                #print(mean_i)
                                 
                scale_size = np.round((mean_size[0]/mean_i) * data[i]['mask'].shape).astype(int)  
                            
                pil_img = PIL.Image.fromarray(data[i]['mask'].numpy())
                pil_img = pil_img.resize(scale_size, PIL.Image.NEAREST)
                data[i]['mask'] = t.DoubleTensor(np.array(pil_img).astype(int))

                #rp = regionprops(data[i]['mask'].numpy().astype(int))
                #mean_i2 = np.abs([np.array([r.bbox[1]-r.bbox[3], r.bbox[0]-r.bbox[2]]) for r in rp]).mean(axis=0)
                #print(mean_i2)

                #plt.subplot(1,2,2), plt.imshow(data[i]['mask']), plt.show()

                pil_img = PIL.Image.fromarray(data[i]['img'].numpy())
                pil_img = pil_img.resize(scale_size, PIL.Image.BILINEAR)
                data[i]['img'] = t.ByteTensor(np.array(pil_img).astype(float))

                # plt.figure(1), plt.imshow(data[i]['img']), plt.show()
                   
        self.datas = data

    def __getitem__(self, index):
        data = self.datas[index]
        img = data['img'].numpy()
        mask = data['mask'][:,:,None].byte().numpy()

        if self.augment == True:
            imgWidth = self.imgWidth

            # do cropping to imgWidth x imgWidth if *any* dimension is larger than imgWidth
            if np.all(np.asarray(img.shape[0:2]) > imgWidth):                
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
         
            #if 0 <= raugment <= 33:
            # mirror - left/right
            rint = randint(0, 100)
            if p  > rint:
                mask = np.fliplr(mask)
                img  = np.fliplr(img)
    
            #elif 33 < raugment <= 66:
            # mirror - up/down
            rint = randint(0, 100)
            if p > rint:
                mask = np.flipud(mask)
                img = np.flipud(img)

            #elif 66 < raugment <= 100:
            # transpose
            rint = randint(0, 100)
            if p > rint:
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
        mask_stacked = makeMask(mask, self.maskConf, self.names)

        return img, mask_stacked, mask
    
    def __len__(self):
        return len(self.datas)


#    def printImageSize(self):
#        ipdb.set_trace()
#        print("test")

def makeMask(mask, maskConf, names, useCentroid=False):
    if sum(maskConf) > 0:
        # if there is at least one cell
        if mask.sum() > 0:
            # edge mask
            edge_mask = t.from_numpy(sobel(mask[0,:,:]/mask.max()).astype('float32')).unsqueeze(0) # sobel filter
            edge_mask[edge_mask > 0] = 1 # binarize

            # centroids 
            # print(mask)
            if useCentroid:
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
            if useCentroid:
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
        #ipdb.set_trace()
        # substract edges from mask
        #mask_binary_diff = mask_binary - edge_mask
        #mask_binary_diff[mask_binary < 0] = 0
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
    def __init__(self,path,source_transform,normalize,cielab):
        self.datas = t.load(path)
        if normalize:
            print("Perform histogram equalization")
            for i in range(len(self.datas)):
                if i == 20:
                    ipdb.set_trace()
                    print("test")
                sys.stdout.write("\r" + str(i))
                self.datas[i]['img'] = equalHist(self.datas[i]['img'])
	    
            print("")	

        if cielab:
            print("Perform RGB to CIELAB conversion")
            for i in range(len(self.datas)):  
                sys.stdout.write("\r" + str(i))
                self.datas[i]['img'] = colorspaceConversion(self.datas[i]['img'])
                sys.stdout.write("\n")

                # d2 = rgb2lab(data[i]['img'])
                # d2 = rgb2lab(self.datas[i]['img'])
                # d2[:,:,0] = d2[:,:,0]*(255/100)
                # self.datas[i]['img'] = t.ByteTensor(d2 + [0,128,128])

            # plt.figure(1)
            # plt.subplot(1,2,1)
            # plt.imshow(self.datas[0]['img'][:,:,0])
            # plt.subplot(1,2,2)
            # plt.imshow(img2[:,:,0])
            # plt.show()


        self.s_transform = source_transform
    def __getitem__(self, index):
        data = self.datas[index]
        name = data['name']
        img = data['img'].numpy()
        shape = t.IntTensor([img.shape[0], img.shape[1]])
        #if shape[0] > shape[1] and shape[1] < 200:
        #    img = np.concatenate((img,np.ones((shape[0],shape[0]-shape[1],3)).astype(np.uint8)*5),axis=1)
        #    shape[1] = shape[0]
        img = self.s_transform(img)
        return img, shape, name
    def __len__(self):
        return len(self.datas)

    def getImgShapes(self):
        shapes = []
        for data in self.datas:
            shapes.append((data['img'].shape[0], data['img'].shape[1]))

        return Counter(elem for elem in shapes)


def equalHist(img):
    img = exposure.equalize_adapthist(img.numpy())
    #img = denoise_tv_chambolle(img, weight=0.02, multichannel=False)
    return t.from_numpy(img*255).type(t.ByteTensor)

def colorspaceConversion(img):
    ''' Convert from RGB to CIELAB colorspace '''
        
    img = rgb2lab(img)
    img[:,:,0] = img[:,:,0]*(255/100)
    return t.ByteTensor(img + [0,128,128])

def createKSplits(l, K, random_state = 0):
    ''' createKSplits(l, K): returns a list with K entries, each holding a list of indices that constitute one splits. l is the number of data points '''
    arr = np.arange(l)
    np.random.seed(random_state)
    np.random.shuffle(arr)

    d = int (l/K)
    return [arr[d*i:d*(i+1)] if i < K-1 else arr[d*i:] for i in range(K)]


def readFromDisk(valIdx, path):
    # read files from disk
    if type(path) == list:
        data = t.load(path[0])
        for p in path[1:]:
            data += t.load(p)
    else:
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
