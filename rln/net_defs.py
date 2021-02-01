import torch
import torch.optim as optim
import math
import torch.nn.functional as F


class CircPad(torch.nn.Module):
    def __init__(self, pad_size):
        super(CircPad, self).__init__()
        if type(pad_size) == tuple:
            self.padding = pad_size
        else:
            self.padding = tuple(pad_size for i in range(6))

    def forward(self, x):
        x = F.pad(x, self.padding, mode='circular')
        return x

    def __repr__(self):
        return f"{type(self).__name__}(pad_size={self.padding})"


class ImageCrop(CircPad, torch.nn.Module):
    # opposite of CircPad, so same constructor
    def __init__(self, pad_size):
        super(ImageCrop, self).__init__(pad_size)

    def forward(self, x):
        # get bounds in each direction
        xl, xh, yl, yh, zl, zh = self.padding

        # crop out the padded areas
        return x[:, :,  xl:-xh, yl:-yh, zl:-zh]


class ForwardBlock(torch.nn.Module):
    """Run one conv layer"""

    def __init__(self, ks, channels_in, channels_out, use_activ=True):
        super(ForwardBlock, self).__init__()

        # do one conv layer
        self.pad = CircPad(ks // 2)
        self.conv = torch.nn.Conv3d(in_channels=channels_in,
                                    out_channels=channels_out, kernel_size=ks, stride=1)

        self.use_activ = use_activ
        if self.use_activ:
            self.activ = torch.nn.PReLU(num_parameters=channels_out)

    def forward(self, x):

        # apply operations
        x = self.pad(x)
        x = self.conv(x)

        if self.use_activ:
            x = self.activ(x)
        return x


class OutputBlock(torch.nn.Module):
    """Flatten output channels using 1x1x1 convolutions"""

    def __init__(self, ks, channels_in, channels_out):
        super(OutputBlock, self).__init__()

        # do one conv layer
        self.convflat = torch.nn.Conv3d(in_channels=channels_in,
                                        out_channels=channels_out, kernel_size=1, stride=1)

    def forward(self, x):

        # just apply convolution
        x = self.convflat(x)

        return x


class DownBlock(torch.nn.Module):
    """Downsample and apply convolution"""

    def __init__(self, ks, channels_in, channels_out):
        super(DownBlock, self).__init__()

        # do a downsample
        self.convdown = torch.nn.Conv3d(in_channels=channels_in,
                                        out_channels=channels_out, kernel_size=2, stride=2)
        self.activ_down = torch.nn.PReLU(num_parameters=channels_out)

    def forward(self, x):
        # apply downsampling
        x = self.convdown(x)
        x = self.activ_down(x)

        return x


class UpBlock(torch.nn.Module):
    """Upsample and apply convolution"""

    def __init__(self, ks, channels_in, channels_out):
        super(UpBlock, self).__init__()

        # do an upsample
        self.conv_up = torch.nn.ConvTranspose3d(in_channels=channels_in,
                                                out_channels=channels_out, kernel_size=2, stride=2)
        self.activ_up = torch.nn.PReLU(num_parameters=channels_out)


    def forward(self, x, xpass):
        # apply upsampling to low-level
        x = self.conv_up(x)
        x = self.activ_up(x)
        # concatenate the high- and low-level info
        x = torch.cat([x, xpass], 1)

        return x

class RecNetBase():
    """ 
        Simple base class for a network used in an RLN (and the RLN itself). 
        Defines some useful helpers for model serialization
    """

    def name(self):
        """ 
            Get string representation for model
        """
        return str(type(self).__name__)

    def serialize(self, running_dict=None):
        """
            Dump model parameters to dict
        """
        net_params = running_dict or dict()

        # make sure we load this correctly later
        net_params['name'] = self.name()
        net_params['type'] = type(self)

        return net_params

    def unserialize(self, net_params):
        """ 
            Load a model parameters from dict
        """
        #! Load params from dict BEFORE constructing layers
        assert self.name() == net_params['name'], "Net types do not match!"

        



