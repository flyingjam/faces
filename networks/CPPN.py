import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    return np.concatenate((x_mat, y_mat, r_mat), axis=2)

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


