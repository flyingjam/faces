import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from networks import Network
from torch.utils.data import DataLoader

from utils import conv2dsize


class Encoder(nn.Module):
    def __init__(self, side_length, n_channel, z_channel, z_dim, dropout = 0.5, kernel_size=5, use_cuda = False):
        super(Encoder, self).__init__()

        '''
        side_length is the length in pixels of the square input image
        n_channel is the number of channels (3 for RGB, 1 for grayscale)
        z_channel is the quantized number of output channels that the convolution use
        z_dim is the size of the latent vector
        '''

        self.n_channel = n_channel
        self.z_channel = z_channel
        self.z_dim = z_dim
        self.use_cuda = use_cuda


        self.out_dimension = side_length
        for i in range(3):
            self.out_dimension = conv2dsize(self.out_dimension, kernel_size, stride=2)

        '''
        Networks calculate mu and the log var in order to use the reparameterization trick
        '''
        self.mu = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel * 4),
            nn.LeakyReLU()
        )

        self.var = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size=kernel_size, stride=2),
            nn.BatchNorm2d(z_channel * 4),
            nn.LeakyReLU()
        )



        self.mu_out = nn.Linear(z_channel * 4 * self.out_dimension, z_dim)
        self.var_out = nn.Linear(z_channel * 4 * self.out_dimension, z_dim)


    def forward(self, X):

        mu_out = self.mu(X)
        mu_out = mu_out.view(mu_out.size(0), -1)
        mu_out = self.mu_out(mu_out)

        var_out = self.var(X)
        var_out = var_out.view(var_out.size(0), -1)
        var_out = self.var_out(var_out)

        return mu_out, var_out

    def sample(self, batch_size, mu, var):
        eps = Variable(torch.randn(batch_size, self.z_dim))
        if self.use_cuda:
            eps = eps.cuda()
        return mu + torch.exp(var/2) * eps

class Decoder(nn.Module):

    def __init__(self, side_length, n_channel, z_channel, z_dim, droppout = 0.5, kernel_size = 5):
        super(Decoder, self).__init__()

        self.z_dim = z_dim
        self.n_channel = n_channel
        self.side_length = side_length

        self.convnet = nn.Sequential(
            nn.ConvTranspose2d(z_dim, z_channel * 2, kernel_size, 2, output_padding=0),
            nn.BatchNorm2d(z_channel * 2),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(z_channel * 2, z_channel, kernel_size, 2, output_padding=1),
            nn.BatchNorm2d(z_channel),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.ConvTranspose2d(z_channel, n_channel, kernel_size, 2, output_padding=1),
            nn.Sigmoid()
        )



    def forward(self, X):
        out = X.view(X.size(0), self.z_dim, 1, 1)
        out = self.convnet(out)
        return out


default_params = {
    'z_channel': 16,
    'z_dim': 256,
    'encoder dropout': 0.5,
    'decoder dropout': 0.3,
    'kernel': 5,
    'batch size': 200,
    'epochs': 20,
    'learning rate': 0.01
}

class VAENetwork(Network):

    def __init__(self, dataset, hyper_params = {}, cuda=False):
        side_length =32
        RGB = 3


        self.use_cuda = cuda
        self.params = default_params
        self.params.update(hyper_params)

        self.dataloader = DataLoader(dataset=dataset,
                                     shuffle=True,
                                     batch_size=self.params['batch size'],
                                     drop_last=True)

        self.encoder = Encoder(side_length,
                               RGB,
                               self.params['z_channel'],
                               self.params['z_dim'],
                               self.params['encoder dropout'],
                               self.params['kernel'],
                               self.use_cuda)
        self.decoder =Decoder(side_length,
                              RGB,
                              self.params['z_channel'],
                              self.params['z_dim'],
                              self.params['decoder dropout'],
                              self.params['kernel'])

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(parameters, self.params['learning rate'])

        if self.use_cuda:
            self.encoder.cuda()
            self.decoder.cuda()

    def train_epoch(self, epoch = 0):

        TINY = 1e-15 #BCE is NaN with an input of 0
        before_time = time.clock()

        self.encoder.train()
        self.decoder.eval()

        loss_sum = 0
        total = 0
        for img, label in self.dataloader:
            self.optimizer.zero_grad()

            X = Variable(img)
            if self.use_cuda:
                X = X.cuda()

            z_mu, z_var = self.encoder(X)
            z = self.encoder.sample(self.params['batch size'], z_mu, z_var)

            X_reconstructed = self.decoder(z)
            reconstruction_loss = F.binary_cross_entropy(X_reconstructed + TINY, X + TINY, size_average=True)
            KL_loss = z_mu.pow(2).add_(z_var.exp()).mul_(-1).add_(1).add_(z_var)
            KL_loss = torch.sum(KL_loss).mul_(-0.5)
            total_loss = reconstruction_loss + KL_loss
            total_loss.backward()
            self.optimizer.step()

        duration = time.clock() - before_time

        #TODO: Structured way to return training results
        return duration, total_loss.data[0], X_reconstructed





