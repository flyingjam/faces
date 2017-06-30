import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import conv2dsize

class Encoder(nn.Module):
    def __init__(self, side_length, n_channel, z_channel, z_dim, dropout = 0.5, kernal_size=5, use_cuda = False):
        super(Encoder, self).__init__()

        self.n_channel = n_channel
        self.z_channel = z_channel
        self.z_dim = z_dim
        self.use_cuda = use_cuda


        self.out_dimension = side_length
        for i in range(3):
            self.out_dimension = conv2dsize(self.out_dimension, kernal_size, stride=2)

        self.mu = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size=kernal_size, stride=2),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size=kernal_size, stride=2),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size=kernal_size, stride=2),
            nn.BatchNorm2d(z_channel * 4),
            nn.LeakyReLU()
        )

        self.var = nn.Sequential(
            nn.Conv2d(n_channel, z_channel, kernel_size=kernal_size, stride=2),
            nn.BatchNorm2d(z_channel),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel, z_channel * 2, kernel_size=kernal_size, stride=2),
            nn.BatchNorm2d(z_channel * 2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(z_channel * 2, z_channel * 4, kernel_size=kernal_size, stride=2),
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




