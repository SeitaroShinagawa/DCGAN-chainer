import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
#from sklearn.datasets import fetch_mldata

def clip(a):
  return 0 if a<0 else (255 if a>255 else a)

def array_to_img(im):
  im = im*255
  im = np.vectorize(clip)(im).astype(np.uint8)
  im=im.transpose(1,2,0)
  img=Image.fromarray(im)
  return img

def Binarize(x):
    return (np.sign(x-0.5)+1)*0.5

def load_img(image_path): #load as np.array
  img = Image.open(image_path)
  img = np.array(img,dtype=np.float32).transpose(2,0,1)/255
  return img

def save_img(img_array,save_path): #save from np.array
  img = array_to_img(img_array)
  img.save(save_path)

class RPGCharacters:
    """
    yurudorashiru RPG tsuku-ru data
    data   ?x4x64x64 (RGBA)
    """
    def __init__(self,binarize=False):
      if os.path.exists("RPGCharacters.npy"):
        sys.stdout.write("load RPGCharacter.npy\n")
        imgs_array = np.load("RPGCharacters.npy")
      else:        
        sys.stdout.write("create RPGCharacter.npy\n")
        image_root="/project/nakamura-lab07/Work/seitaro-s/RPGimages/3_sv_actors_20160915"
        img_list=[]
        with open(image_root+"/list.txt",'r') as f:
            for line in f:
                img_list.append(line.strip())
        
        imgs_array=[]
        for img_name in img_list:
            img = load_img(image_root+"/3_sv_actors/"+img_name)
            img = np.transpose(img,(1,2,0))

            img_bar_list = np.hsplit(img,9)
            imgs=[]
            for i in img_bar_list:
                img_list = np.vsplit(i,6)
                for j in img_list:
                    imgs.append(j)
            imgs = np.array(imgs,dtype=np.float32)
            imgs = np.transpose(imgs,(0,3,1,2))
            imgs_array.append(imgs)
        imgs_array = np.array(imgs_array,dtype=np.float32)
        ishape = imgs_array.shape
        imgs_array = np.reshape(imgs_array,(ishape[0]*ishape[1],ishape[2],ishape[3],ishape[4]))

        #RGBA --> RGB
        RGB_array = np.array([img_array[:-1] for img_array in imgs_array])
        imgs_array = RGB_array

        sys.stdout.write("save RPGCharacter.npy\n")
        np.save("RPGCharacters",imgs_array)
      self.train_array = imgs_array[:60000]
      self.test_array = imgs_array[60000:]

    def gen_train(self,batchsize,Random=True):
        if Random:
            indexes = np.random.permutation(self.train_array.shape[0])
        else:
            indexes = np.arange(self.train_array.shape[0])
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([self.train_array[x] for x in indexparts],dtype=np.float32)
            yield indexparts, image_batch
            num += 1 

    def gen_test(self,batchsize):
        indexes = np.arange(self.test_array.shape[0])
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([self.test_array[x] for x in indexparts],dtype=np.float32)
            yield indexparts, image_batch
            num += 1 


class MNIST:
    """
    MNIST handwritten recognition data
    train   60000x784 -> 60000x28x28
    test    60000x784 -> 10000x28x28
    train_label 60000x10
    y_label     10000x10
    """
    def __init__(self,binarize=False):
        mnist = fetch_mldata('MNIST original')
        x_all = mnist.data.astype(np.float32)/255
        y_all = mnist.target.astype(np.int32)
        x_train, x_test = np.split(x_all, [60000])
        y_train, y_test = np.split(y_all, [60000])
        if binarize==True:
            x_train = Binarize(x_train)
            x_test = Binarize(x_test)
        x_train = np.array(x_train).reshape(60000,28,28)
        x_test = np.array(x_test).reshape(10000,28,28)

        self.train = x_train
        self.test = x_test
        self.train_label = y_train
        self.test_label = y_test
        self.C=1
        self.width=28
        self.height=28

    def gen_train(self,batchsize,Random=True):
        if Random:
            indexes = np.random.permutation(60000)
        else:
            indexes = np.arange(60000)
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([[self.train[x]] for x in indexparts],dtype=np.float32)
            label_batch = np.asarray([[self.train_label[x]] for x in indexparts],dtype=np.float32)
            yield image_batch ,label_batch
            num += 1 
                
    def gen_test(self,batchsize):
        """
        Attention! minibatch calculation doesn't generate exact result.
        If you need exact result, make batchsize=1.
        """
        num = 0
        while batchsize*num < 10000:
            image_batch = np.asarray(self.test[batchsize*num:batchsize*(num+1)],dtype=np.float32)
            label_batch = np.asarray(self.test_label[batchsize*num:batchsize*(num+1)],dtype=np.float32)
            yield image_batch,label_batch
            num += 1 

