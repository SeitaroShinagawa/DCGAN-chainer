#!usr/bin/python

import os
import sys
import numpy as np
from RPGCharacters_util import RPGCharacters
from gan import Generator,Discriminator
from chainer import Variable,cuda,optimizers,serializers
import random
random.seed(0)

def clip(a):
  return 0 if a<0 else (255 if a>255 else a)

def array_to_img(im):
  im = im*255
  im = np.vectorize(clip)(im).astype(np.uint8)
  im=im.transpose(1,2,0)
  img=Image.fromarray(im)
  return img

def save_img(img_array,save_path): #save from np.array
  img = array_to_img(img_array)
  img.save(save_path)

save_path = "model"
if os.path.exists(save_path)==False:
    os.mkdir(save_path)    

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
  for i,(data,charaid,poseid) in enumerate(real.gen_train(batchsize)):
    rand_ = random.uniform(0,1)
    if rand_ < 0.8:
        Dis.zerograds()

        x = Variable(xp.array(data))
        label_real = Variable(xp.ones((batchsize,1),dtype=xp.int32))

        y, loss = Dis(x,label_real)
        loss_real_dis += loss.data
        loss.backward()
        optD.update()
        n_real_dis += batchsize
    elif rand_ < 0.9:
        Dis.zerograds()

        z = Gen.generate_hidden_variables(batchsize)
        x = Gen(Variable(xp.array(z)))
        label_real = Variable(xp.ones((batchsize,1),dtype=xp.int32))
        y, loss = Dis(x,label_real)
        loss_fake_dis += loss.data
        loss.backward()
        optD.update()
        n_fake_dis += batchsize
    else:
        Gen.zerograds()
        Dis.zerograds()

        z = Gen.generate_hidden_variables(batchsize)
        x = Gen(Variable(xp.array(z)))
        label_fake = Variable(xp.zeros((batchsize,1),dtype=xp.int32))
        y, loss = Dis(x,label_fake)
        loss_fake_gen += loss.data
        loss.backward()
        optG.update()
        n_fake_gen += batchsize
    z = Gen.generate_hidden_variables(batchsize)
    x = Gen(Variable(xp.array(z))) #(B,3,64,64) B:batchsize
    #(3,B/8*64,8*64)
    tmp = np.rollaxis(x.to_cpu().data,0,1) #(3,B,64,64)
    tmp = np.reshape(tmp,(3,B/8,64*8,64)) #(3,B/8,64*8,64)
    
    sys.stdout.write("\rtrain... epoch{}, {}/{}".format(epoch,i*batchsize,trainsize))
    sys.stdout.flush()
  print("\nfake_gen_loss:{}(all/{}) fake_dis_loss:{}(all/{}), real_dis_loss:{}(all/{})".format(loss_fake_gen/float(n_fake_gen),n_fake_gen,loss_fake_dis/float(n_fake_dis),n_fake_dis,loss_real_dis/float(n_real_dis),n_real_dis)) #losses are approximated values
  print('save model ...')
  prefix = save_path+"/"+str(epoch).zfill(3)
  if os.path.exists(prefix)==False:
    os.mkdir(prefix)        
  serializers.save_npz(prefix + '/Geights', Gen.to_cpu()) 
  serializers.save_npz(prefix + '/Goptimizer', optG)
  serializers.save_npz(prefix + '/Dweights', Dis.to_cpu())
  serializers.save_npz(prefix + '/Goptimizer', optD)
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
  print(" test real belief mean:{}({}/{})".format(real_belief_mean/testsize,real_belief_mean,testsize))
  for j,(data_i,data) in enumerate(real.gen_test(1)):
        z = Gen.generate_hidden_variables(1)
        x = Gen(Variable(xp.array(z)),train=False)
        label = Variable(xp.zeros((1,1),dtype=xp.int32))
        
        y, loss = Dis(x,label,train=False)
        fake_belief_mean += xp.sum(y.data) 
        sys.stdout.write("\rtest fake...{}/{}".format(j,testsize))
        sys.stdout.flush()
  print(" test fake belief mean:{}({}/{})".format(fake_belief_mean/testsize,fake_belief_mean,testsize))
        

