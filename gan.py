
import chainer
import chainer.cuda
import chainer.functions as F
import chainer.links as L
import chainer.optimizers
import numpy as np


def init_normal(links, sigma):
    for link in links:
        shape = link.W.data.shape
        link.W.data[...] = np.random.normal(0, sigma, shape).astype(np.float32)

class Generator(chainer.Chain):

    n_hidden = 100
    sigma = 0.01

    def __init__(self):
        super(Generator, self).__init__(
            fc5=L.Linear(100, 512 * 4 * 4),
            norm5=L.BatchNormalization(512 * 4 * 4),
            conv4=L.Deconvolution2D(512, 256, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(256),
            conv3=L.Deconvolution2D(256, 128, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(128),
            conv2=L.Deconvolution2D(128, 64,  ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(64),
            conv1=L.Deconvolution2D(64,  3,   ksize=4, stride=2, pad=1))
        init_normal(
            [self.conv1, self.conv2, self.conv3,
             self.conv4, self.fc5], self.sigma)


    def __call__(self, z, train=True):
        n_sample = z.data.shape[0]
        test = not train
        h = F.relu(self.norm5(self.fc5(z), test=test))
        h = F.reshape(h, (n_sample, 512, 4, 4))
        h = F.relu(self.norm4(self.conv4(h), test=test))
        h = F.relu(self.norm3(self.conv3(h), test=test))
        h = F.relu(self.norm2(self.conv2(h), test=test))
        x = F.sigmoid(self.conv1(h))
        return x

    def make_optimizer(self):
        return chainer.optimizers.Adam(alpha=1e-4, beta1=0.5)

    def generate_hidden_variables(self, n): # n:batchsize
        return np.asarray(
            np.random.uniform(
                low=-1.0, high=1.0, size=(n, self.n_hidden)),
            dtype=np.float32)


class Discriminator(chainer.Chain):

    sigma = 0.01

    def __init__(self):
        super(Discriminator, self).__init__(
            conv1=L.Convolution2D(3,   64,  ksize=4, stride=2, pad=1),
            conv2=L.Convolution2D(64,  128, ksize=4, stride=2, pad=1),
            norm2=L.BatchNormalization(128),
            conv3=L.Convolution2D(128, 256, ksize=4, stride=2, pad=1),
            norm3=L.BatchNormalization(256),
            conv4=L.Convolution2D(256, 512, ksize=4, stride=2, pad=1),
            norm4=L.BatchNormalization(512),
            fc5=L.Linear(512 * 4 * 4, 1))
        init_normal(
            [self.conv1, self.conv2, self.conv3,
             self.conv4, self.fc5], self.sigma)

    def __call__(self, x, t, train=True):
        test = not train
        n_sample = x.data.shape[0]
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.norm2(self.conv2(h), test=test))
        h = F.leaky_relu(self.norm3(self.conv3(h), test=test))
        h = F.leaky_relu(self.norm4(self.conv4(h), test=test))
        y = self.fc5(h)
        return F.sigmoid(y), F.sigmoid_cross_entropy(y, t)

    def make_optimizer(self):
        return chainer.optimizers.Adam(alpha=1e-4, beta1=0.5)
