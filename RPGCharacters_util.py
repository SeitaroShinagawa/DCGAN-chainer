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
      if os.path.exists("RPGCharacters.npy") and os.path.exists("RPGCharacters_charaID.npy") and os.path.exists("RPGCharacters_charaID.npy"):
        sys.stdout.write("load PGCharacter.npy, RPGCharacters_charaID.npy, RPGCharacters_poseID.npy...\n")
        imgs_array = np.load("RPGCharacters.npy")
        label_character_id = np.load("RPGCharacters_charaID.npy")
        label_pose_id = np.load("RPGCharacters_poseID.npy")
      else:        
        sys.stdout.write("create RPGCharacter.npy, RPGCharacters_charaID.npy, RPGCharacters_poseID.npy...\n")
        image_root="/path/to/3_sv_actors_20160915"
        img_list=[]
        with open(image_root+"/list.txt",'r') as f:
            for line in f:
                img_list.append(line.strip())
        
        imgs_array=[]
        label_character_id=[] #(characterID,poseID)
        label_pose_id=[]
        for id_, img_name in enumerate(img_list):
            img = load_img(image_root+"/3_sv_actors/"+img_name)
            img = np.transpose(img,(1,2,0))

            img_bar_list = np.hsplit(img,9)
            imgs=[]
            for num_i, i in enumerate(img_bar_list):
                img_list = np.vsplit(i,6)
                for num_j, j in enumerate(img_list):
                    imgs.append(j)
                    poseid = 6*num_i+num_j
                    label_character_id.append([id_])
                    label_pose_id.append([poseid])
            imgs = np.array(imgs,dtype=np.float32)
            imgs = np.transpose(imgs,(0,3,1,2))
            imgs_array.append(imgs)
        imgs_array = np.array(imgs_array,dtype=np.float32)
        label_character_id = np.array(label_character_id,dtype=np.int32)
        label_pose_id = np.array(label_pose_id,dtype=np.int32)
        ishape = imgs_array.shape
        imgs_array = np.reshape(imgs_array,(ishape[0]*ishape[1],ishape[2],ishape[3],ishape[4]))

        RGB_array = np.array([img_array[:-1] for img_array in imgs_array]) #RGBA->RGB
        imgs_array = RGB_array
      
        sys.stdout.write("save RPGCharacter.npy, RPGCharacters_charaID.npy, RPGCharacters_poseID.npy\n")
        np.save("RPGCharacters.npy",imgs_array)
        np.save("RPGCharacters_charaID.npy",label_character_id)
        np.save("RPGCharacters_poseID.npy",label_pose_id)
      self.train_array = imgs_array[:60000]
      self.test_array = imgs_array[60000:]
      self.train_character_id = label_character_id[:60000]
      self.test_character_id = label_character_id[60000:]
      self.train_pose_id = label_pose_id[:60000]
      self.test_pose_id = label_pose_id[60000:]      
      self.height = 64
      self.width = 64
      self.C = 3 #color dimension
      self.train_size = len(self.train_array)
      self.test_size = len(self.test_array)
 
    def gen_train(self,batchsize,Random=True):
        if Random:
            indexes = np.random.permutation(self.train_array.shape[0])
        else:
            indexes = np.arange(self.train_array.shape[0])
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([self.train_array[x] for x in indexparts],dtype=np.float32) 
            chara_batch = np.asarray([self.train_character_id[x] for x in indexparts],dtype=np.int32)
            pose_batch = np.asarray([self.train_pose_id[x] for x in indexparts],dtype=np.int32)
            yield image_batch, chara_batch, pose_batch
            num += 1 

    def gen_test(self,batchsize):
        indexes = np.arange(self.test_array.shape[0])
        num = 0
        while batchsize*num < len(indexes):
            indexparts = indexes[batchsize*num:batchsize*(num+1)]
            image_batch = np.asarray([self.test_array[x] for x in indexparts],dtype=np.float32)
            chara_batch = np.asarray([self.test_character_id[x] for x in indexparts],dtype=np.int32)
            pose_batch = np.asarray([self.test_pose_id[x] for x in indexparts],dtype=np.int32)
            yield image_batch, chara_batch, pose_batch
            num += 1 

