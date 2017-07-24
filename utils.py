from math import floor

def conv2dsize(in_side, kernal_size, stride = 1, padding = 0, dilation = 1):
    '''Helper function that calculates the output dimensions of a square convolution'''
    return floor((in_side + 2 * padding - dilation * (kernal_size - 1) - 1)/stride + 1)

def conv2dtranposesize(in_side, kernal_size, stride = 1, padding = 0, output_padding = 0, dilation = 1):
    return (in_side - 1) * stride - 2*padding + kernal_size + output_padding