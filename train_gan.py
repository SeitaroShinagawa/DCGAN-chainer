#!usr/bin/python

import os
import sys
import numpy as np
from RPGCharacters_util import RPGCharacters
from gan import Generator,Discriminator
from chainer import Variable,cuda,optimizers,serializers
from PIL import Image
import random
random.seed(0)

save_path=sys.argv[1]
if not os.path.exists(save_path):
  os.mkdir(save_path)
if not os.path.exists(save_path+"/model"):
  os.mkdir(save_path+"/model")

def clip(a):
  return 0 if a<0 else (255 if a>255 else a)

def array_to_img(im):
  im = im*255
  im = np.vectorize(clip)(im).astype(np.uint8)
  im=im.transpose(1,2,0)
  img=Image.fromarray(im)
  return img

def save_img(img_array,save_path): #save from np.array (3,height,width)
  img = array_to_img(img_array)
  img.save(save_path)

Gen = Generator()
Dis = Discriminator()

gpu = -1 
if gpu>=0:
    xp = cuda.cupy
    cuda.get_device(gpu).use()
    Gen.to_gpu()
    Dis.to_gpu()
else:
    xp = np

optG = Gen.make_optimizer()
optD = Dis.make_optimizer()
optG.setup(Gen)
optD.setup(Dis)

real = RPGCharacters()
trainsize=real.train_array.shape[0]
testsize=real.test_array.shape[0]

loss_fake_gen = 0.0
loss_fake_dis = 0.0
loss_real_dis = 0.0
batchsize = 64
max_epoch = 100
for epoch in range(max_epoch):
  n_fake_gen = 0
  n_fake_dis = 0
  n_real_dis = 0
  for data,charaid,poseid in real.gen_train(batchsize):
    rand_ = random.uniform(0,1)
    B = data.shape[0]
    if rand_ < 0.2:
        Dis.zerograds()

        x = Variable(xp.array(data))
        label_real = Variable(xp.ones((B,1),dtype=xp.int32))

        y, loss = Dis(x,label_real)
        loss_real_dis += loss.data
        loss.backward()
        optD.update()
        n_real_dis += B
    elif rand_ < 0.4:
        Dis.zerograds()

        z = Gen.generate_hidden_variables(B)
        x = Gen(Variable(xp.array(z)))
        label_real = Variable(xp.zeros((B,1),dtype=xp.int32))
        y, loss = Dis(x,label_real)
        loss_fake_dis += loss.data
        loss.backward()
        optD.update()
        n_fake_dis += B
    else:
        Gen.zerograds()
        Dis.zerograds()

        z = Gen.generate_hidden_variables(B)
        x = Gen(Variable(xp.array(z)))
        label_fake = Variable(xp.ones((B,1),dtype=xp.int32))
        y, loss = Dis(x,label_fake)
        loss_fake_gen += loss.data
        loss.backward()
        optG.update()
        n_fake_gen += B
    sys.stdout.write("\rtrain... epoch{}, {}/{}".format(epoch,n_real_dis+n_fake_dis+n_fake_gen,trainsize))
    sys.stdout.flush()
  
  z = Gen.generate_hidden_variables(batchsize)
  x = Gen(Variable(xp.array(z))) #(B,3,64,64) B:batchsize
  x.to_cpu()
  tmp = np.transpose(x.data,(1,0,2,3)) #(3,B,64,64)
  img_array=[]
  for i in range(3):
    img_array2=[]
    for j in range(0,batchsize,8):
      img=tmp[i][j:j+8]
      img=np.transpose(img.reshape(64*8,64),(1,0))
      img_array2.append(img)
    img_array2=np.array(img_array2).reshape(8*64,int(batchsize/8*64))
    img_array.append(np.transpose(img_array2,(1,0)))
  img_array = np.array(img_array)
  print("\nsave fig...")
  save_img(img_array,save_path+"/{}.png".format(str(epoch).zfill(3)))  
  print("fake_gen_loss:{}(all/{}) fake_dis_loss:{}(all/{}), real_dis_loss:{}(all/{})".format(loss_fake_gen/float(n_fake_gen),n_fake_gen,loss_fake_dis/float(n_fake_dis),n_fake_dis,loss_real_dis/float(n_real_dis),n_real_dis)) #losses are approximated values
  print('save model ...')
  prefix = save_path+"/model/"+str(epoch).zfill(3)
  if os.path.exists(prefix)==False:
    os.mkdir(prefix)        
  serializers.save_npz(prefix + '/Geights', Gen.to_cpu()) 
  serializers.save_npz(prefix + '/Goptimizer', optG)
  serializers.save_npz(prefix + '/Dweights', Dis.to_cpu())
  serializers.save_npz(prefix + '/Doptimizer', optD)
  Gen.to_gpu()
  Dis.to_gpu()

  real_belief_mean = 0.0
  fake_belief_mean = 0.0
  for j,(data,charaid,poseid) in enumerate(real.gen_test(1)):
        x = Variable(xp.array(data))
        label = Variable(xp.ones((1,1),dtype=xp.int32)) 

        y, loss = Dis(x,label,train=False)
        real_belief_mean += xp.sum(y.data)
        sys.stdout.write("\rtest real...{}/{}".format(j,testsize))
        sys.stdout.flush()
  print(" test real belief mean:{}({}/{})".format(real_belief_mean/testsize,real_belief_mean,testsize))
  for j,(data,charaid,poseid) in enumerate(real.gen_test(1)):
        z = Gen.generate_hidden_variables(1)
        x = Gen(Variable(xp.array(z)),train=False)
        label = Variable(xp.zeros((1,1),dtype=xp.int32))
        
        y, loss = Dis(x,label,train=False)
        fake_belief_mean += xp.sum(y.data) 
        sys.stdout.write("\rtest fake...{}/{}".format(j,testsize))
        sys.stdout.flush()
  print(" test fake belief mean:{}({}/{})".format(fake_belief_mean/testsize,fake_belief_mean,testsize))
        

