import time
import os
from networks import Network, Result, Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable

def coordinates(batch, x_dim = 32, y_dim = 32, scale = 1.0):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = x_dim * y_dim
    x_range = scale*(np.arange(x_dim)-(x_dim-1)/2.0)/(x_dim-1)/0.5
    y_range = scale*(np.arange(y_dim)-(y_dim-1)/2.0)/(y_dim-1)/0.5
    x_mat = np.matmul(np.ones((y_dim, 1)), x_range.reshape((1, x_dim)))
    y_mat = np.matmul(y_range.reshape((y_dim, 1)), np.ones((1, x_dim)))
    r_mat = np.sqrt(x_mat*x_mat + y_mat*y_mat)
    x_mat = np.tile(x_mat.flatten(), batch).reshape(1, batch * n_points, 1).astype(np.float32)
    y_mat = np.tile(y_mat.flatten(), batch).reshape(1, batch * n_points, 1).astype(np.float32)
    r_mat = np.tile(r_mat.flatten(), batch).reshape(1, batch * n_points, 1).astype(np.float32)
    return torch.from_numpy(np.concatenate((x_mat, y_mat, r_mat), axis=2)).float()

def combine(coord, z):
    '''
    Combines coordinates and latent vector into one input tensor, also converts it to a torch tensor

    Coordinates have shape [1, batch * width * height, 3]
    Latent vector should have shape [1, batch * width * height, z_dim]
    '''

    out = torch.cat((coord, z), 2)
    return out

def create_latent(latent, batch, x_dim, y_dim):
    #latent vectors from encoder of shape [batch, z_dim]
    z = latent.repeat(x_dim * y_dim, 1)
    return z.view(1, batch * x_dim * y_dim, -1)

class CPPN(nn.Module):
    def __init__(self, z_dim, n_channel, z_channel, dropout=0.5):
        super(CPPN, self).__init__()
        self.z_dim = z_dim
        self.n_channel = n_channel
        self.z_channel = z_channel

        kernel_size = 5 #

        # [3 + z_dim] -> [3 + z_dim, 1] -> [n_channel, 1]
        # you add 3 because x, y, and r data is added alongside the latent vector

        self.net = nn.Sequential(
            nn.ConvTranspose1d(z_dim + 3, z_channel * 4, kernel_size, 1, 2),
            nn.BatchNorm1d(z_channel * 4),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(z_channel * 4, z_channel * 2, kernel_size, 1, 2),
            nn.BatchNorm1d(z_channel * 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(z_channel * 2, z_channel, kernel_size, 1, 2),
            nn.BatchNorm1d(z_channel),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.ConvTranspose1d(z_channel, n_channel, kernel_size, 1, 2),
            nn.Sigmoid()
        )

    def forward(self, X):
        out = X.view(-1, self.z_dim + 3, 1)
        out = self.net(out)
        return out

default_params = {
    'z_channel': 16,
    'z_dim': 256,
    'encoder dropout': 0.5,
    'decoder dropout': 0.3,
    'batch size': 200,
    'epochs': 20,
    'learning rate': 0.01
}

class CPPNNetwork(Network):

    def __init__(self, dataset, hyper_params = {}, cuda = False, width = 32, height = 32):

        self.width = 32
        self.height = 32

        RGB = 3

        self.use_cuda = cuda
        self.params = default_params
        self.params.update(hyper_params)
        self.n_channel = RGB

        self.save_path = "saved_networks/CPPN"

        self.dataloader = DataLoader(dataset = dataset,
                                     shuffle = True,
                                     batch_size = self.params['batch size'],
                                     drop_last = True)

        self.encoder = Encoder(width,
                               RGB,
                               self.params['z_channel'],
                               self.params['z_dim'],
                               self.params['encoder dropout'],
                               self.use_cuda)
        self.decoder = CPPN(self.params['z_dim'],
                            RGB,
                            self.params['z_channel'],
                            self.params['decoder dropout'])

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, self.params['learning rate'])

        if self.use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def train_epoch(self, epoch = 0):
        TINY = 1e-15
        before_time = time.clock()

        self.encoder.train()
        self.decoder.train()

        for img, label in self.dataloader:
            self.optimizer.zero_grad()

            X = Variable(img)
            if self.use_cuda:
                X = X.cuda()

            z_mu, z_var = self.encoder(X)
            z = self.encoder.sample(self.params['batch size'], z_mu, z_var)
            latent = create_latent(z, self.params['batch size'], self.width, self.height)
            coords = Variable(coordinates(self.params['batch size'], self.width, self.height))
            if self.use_cuda:
                coords = coords.cuda()

            inp = combine(coords, latent)

            if self.use_cuda:
                inp = inp.cuda()

            X_reconstructed = self.decoder(inp)#.view(self.params['batch size'], self.n_channel, self.width, self.height)
            reconstruction_loss = F.binary_cross_entropy(X_reconstructed + TINY, X + TINY, size_average = False)
            KL_loss = z_mu.pow(2).add_(z_var.exp()).mul_(-1).add_(1).add_(z_var)
            KL_loss = torch.sum(KL_loss).mul_(-0.5)

            total_loss = reconstruction_loss + KL_loss
            total_loss.backward()
            self.optimizer.step()

            return

        duration = time.clock() - before_time

        def loss_reporting(loss):
            return "{}".format(loss.data[0])

        report = Result(duration, total_loss, epoch, loss_reporting)
        return report



    def save(self, name):
        pass

    def load(self, name):
        pass

    def sample(self, *img):
        self.encoder.eval()
        self.decoder.eval()

        #todo image support

        z = torch.FloatTensor(self.params['batch size'], self.params['z_dim']).normal_()
        z = z.numpy()
        latent = create_latent(z, self.params['batch size'], self.width, self.height)
        coords = coordinates(self.params['batch size'], self.width, self.height)
        inp = Variable(combine(coords, latent))
        if self.use_cuda:
            inp = inp.cuda()

        result = self.decoder(inp)
        return result.data.cpu()



