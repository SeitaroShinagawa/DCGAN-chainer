# DCGAN-chainer
DCGAN simple implementation using chainer [[Paper](https://arxiv.org/abs/1511.06434)]  

train_gan.py: main code  
gan.py: network definition (quoted from [[Chainerを使ってコンピュータにイラストを描かせる](http://qiita.com/rezoolab/items/5cc96b6d31153e0c86bc)]) (in Generator, tanh->sigmoid)  
RPGCharacters_util.py: Utility of dataset([[Yurudorashiru free image resource](http://yurudora.com/tkool/)])  

Requirements (via pip install):  
Chainer [[link](http://chainer.org/)] (verified this code works in version 1.16.0)  
pillow  
numpy  
scipy    

1. First, download dataset from "戦闘ユニット素材　ダウンロード(181.0MB)" in [[http://yurudora.com/tkool/](http://yurudora.com/tkool/)]  
(This dataset has about 62,000 images and each image is 64x64. It is same image size to the paper. So, this dataset is desirable to try simple GAN at first)    
2. After you get 3_sv_actors_20160915 directory, put it to the same place to this code.  
3. Run the code.
```    
python train_gan.py /path/to/save  
```  
(`/path/to/save` means where you want to save the model and generated images every epoch)  

**Random generated images**  

**epoch 0**  
![000.png](https://github.com/SeitaroShinagawa/DCGAN-chainer/blob/master/images/000.png "epoch 0")  

**epoch 50**  
![050.png](https://github.com/SeitaroShinagawa/DCGAN-chainer/blob/master/images/050.png "epoch 50")  

**epoch 99**  
![099.png](https://github.com/SeitaroShinagawa/DCGAN-chainer/blob/master/images/099.png "epoch 99")  



