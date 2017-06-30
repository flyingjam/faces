import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, n_channel):
        super(Encoder, self).__init__()

        self.n_channel = n_channel



    def forward(self, X):
        pass