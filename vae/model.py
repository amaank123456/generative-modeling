import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    """
    Sequential(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (3): ReLU()
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): ReLU()
        (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    """
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        ##################################################################
        # TODO 2.1: Set up the network layers. First create the self.convs.
        # Then create self.fc with output dimension == self.latent_dim
        ##################################################################
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
        )
        outsize1 = self.calculate_conv_out_size(input_shape[-1], 1, 1, 3, 1)
        outsize2 = self.calculate_conv_out_size(outsize1, 1, 1, 3, 2)
        outsize3 = self.calculate_conv_out_size(outsize2, 1, 1, 3, 2)
        outsize4 = self.calculate_conv_out_size(outsize3, 1, 1, 3, 2)
        self.flat_dim = 256*outsize4*outsize4
        self.fc = nn.Linear(self.flat_dim, latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.1: Forward pass through the network, output should be
        # of dimension == self.latent_dim
        ##################################################################
        out = self.convs(x)
        out = out.view(x.shape[0], self.flat_dim)
        return self.fc(out)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
    
    # Function that calculates the output size from a convolution layer (assumes the image has height = width)
    def calculate_conv_out_size(self, inp_size, padding, dilation, kernel_size, stride):
        return (inp_size + 2*padding - dilation * (kernel_size-1) - 1)//stride + 1

class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        ##################################################################
        # TODO 2.4: Fill in self.fc, such that output dimension is
        # 2*self.latent_dim
        ##################################################################
        self.fc = nn.Linear(self.flat_dim, 2*self.latent_dim)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, x):
        ##################################################################
        # TODO 2.4: Forward pass through the network, should return a
        # tuple of 2 tensors, mu and log_std
        ##################################################################
        out = self.convs(x)
        out = out.view(x.shape[0], self.flat_dim)
        out = self.fc(out)
        mu, log_std = out.chunk(2, dim=-1)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        return mu, log_std


class Decoder(nn.Module):
    """
    Sequential(
        (0): ReLU()
        (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (2): ReLU()
        (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (4): ReLU()
        (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (6): ReLU()
        (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    """
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        ##################################################################
        # TODO 2.1: Set up the network layers. First, compute
        # self.base_size, then create the self.fc and self.deconvs.
        ##################################################################
        in_size1 = self.calculate_conv_in_size(self.output_shape[-1], 1, 1, 3, 1)
        in_size2 = self.calculate_conv_transpose_in_size(in_size1, 1, 1, 4, 2, 0)
        in_size3 = self.calculate_conv_transpose_in_size(in_size2, 1, 1, 4, 2, 0)
        in_size4 = self.calculate_conv_transpose_in_size(in_size3, 1, 1, 4, 2, 0)
        self.base_size = in_size4
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.fc = nn.Linear(latent_dim, 256 * self.base_size * self.base_size)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################

    def forward(self, z):
        #TODO 2.1: forward pass through the network, 
        ##################################################################
        # TODO 2.1: Forward pass through the network, first through
        # self.fc, then self.deconvs.
        ##################################################################
        out = self.fc(z)
        out = out.view(out.shape[0], 256, self.base_size, self.base_size)
        return self.deconvs(out)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
    
    # Function that calculates the output size from a convolution layer (assumes the image has height = width)
    def calculate_conv_transpose_in_size(self, out_size, padding, dilation, kernel_size, stride, output_padding):
        return (out_size + 2*padding - dilation * (kernel_size-1) - output_padding - 1)//stride + 1

    # Function that calculates the input size from a convolution layer (assumes the image has height = width)
    def calculate_conv_in_size(self, out_size, padding, dilation, kernel_size, stride):
        return (out_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    # NOTE: You don't need to implement a forward function for AEModel.
    # For implementing the loss functions in train.py, call model.encoder
    # and model.decoder directly.
