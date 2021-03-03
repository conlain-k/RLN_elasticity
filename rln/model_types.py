import torch
import torch.optim as optim
import math
import torch.nn.functional as F

from rln.net_defs import *


class RecurrentLocalizationNet(RecNetBase, torch.nn.Module):
    """ Wrapper to turn a simple predictor net into a recurrent model """

    def __init__(self, H, dim_size, model_type, reuse_net, net_params, kernel_size=3, npasses=5, nprev=0, enforce_avg=True, y_REF=1):
        torch.nn.Module.__init__(self)
        RecNetBase.__init__(self)

        # set up RLN parameters
        self.ds = dim_size
        self.nprev = nprev
        self.npasses = npasses
        self.model_type = model_type
        self.reuse_net = reuse_net
        self.enforce_avg = enforce_avg

        # set reference val to use later
        self.setReferenceValue(y_REF)

        # how many channels should each net have for input?
        self.input_channels = H * (1 + self.nprev)

        # how many channels should be output?
        self.output_channels = 1

        # if we have are given a set of hyperparams, use that instead of defaults
        if net_params is not None:
            self.unserialize(net_params)
        else:
            self.model_params = None # no hyperparams to load
        
        print(f"Given ds of {dim_size}, but cached was {self.ds}. Going with new one!")
        self.ds = dim_size

        # should we tie weights across each iteration?
        if reuse_net:
            # (RLN-t) reuse network across iterations
            self.model = model_type(dim_size=dim_size,
                                    channels_in=self.input_channels, channels_out=self.output_channels, kernel_size=kernel_size, net_params=self.model_params)
        else:
            # (full RLN) construct new network for each iteration
            self.modlist = torch.nn.ModuleList()
            for p in range(npasses):
                newmod = model_type(dim_size=dim_size,
                                    channels_in=self.input_channels, channels_out=self.output_channels, kernel_size=kernel_size, net_params=self.model_params)
                self.modlist.append(newmod)

        # capture model params for later use
        if self.model_params is None:
            self.model_params = self.getModel(0).serialize()

    def serialize(self, running_dict=None):
        net_params = RecNetBase.serialize(self, running_dict)

        # cache my parameters
        net_params["model_params"] = self.model_params
        net_params["nprev"] = self.nprev
        net_params["npasses"] = self.npasses
        net_params["ds"] = self.ds
        net_params["reuse_net"] = self.reuse_net
        net_params["model_type"] = self.model_type
        net_params["enforce_avg"] = self.enforce_avg

        return net_params

    def unserialize(self, net_params):
        RecNetBase.unserialize(self, net_params)

        # overwrite my parameters
        self.model_params = net_params["model_params"]
        self.nprev = net_params["nprev"]
        self.npasses = net_params["npasses"]
        self.ds = net_params["ds"]
        self.reuse_net = net_params["reuse_net"]
        self.model_type = net_params["model_type"]
        self.enforce_avg = net_params["enforce_avg"]

    def _combineInputs(self, x, y, yprev):
        """Combine x and y into product input"""
        yy = [y]
        if yprev is not None:
            # prepend current iterate
            yy = [y] + yprev
        # same as simple net but use two iterations
        x = torch.cat([x * yi for yi in yy], dim=1)

        return x

    def getModel(self, ind):
        """
            Decide which model to use for iteration <ind>        
        """
        assert ind < self.npasses
        if self.reuse_net:
            return self.model
        else:
            return self.modlist[ind]

    def setReferenceValue(self, y_REF):
        # what constant value to use as reference solution?
        self.y_REF = float(y_REF)

    def forward(self, x, accumloss=False, y_target=None, loss_fn=None, return_all=False):
        """ 
            This function does a lot of work. If accumloss is True, then it just passes x through the network and returns the predicted y. 
            If accumloss is True, then it will also compute a running loss for backprop (using loss_fn). 
            If return_all is True, then the entire time series will be stored and returned. 
        """
        # set reference tensor and use as first iterate
        y = yR = x.new_ones(
            (x.shape[0], 1, self.ds, self.ds, self.ds), requires_grad=True) * self.y_REF

        # accumulate running loss if we're in training mode
        running_loss = None

        # make sure these inputs are being used correctly
        if accumloss:
            assert y_target is not None
            assert loss_fn is not None
            assert (not return_all)
        else:
            assert y_target is None
            assert loss_fn is None

        # we will return a series eventually
        if return_all:
            y_series = [yR]

        # extra args
        yprev = None

        # set up previous iterations if needed
        if self.nprev > 0:
            # zero out fake iterates, will be replaced by real ones in later iterations
            yprev = [yR * 0 for i in range(self.nprev)]

        # apply sequence of passes, appending output each time
        for i in range(self.npasses):

            curr_model = self.getModel(i)

            # use product inputs for high-level repr
            xy = self._combineInputs(x, y, yprev)

            # apply net to get perturb
            perturb = curr_model(xy)

            # cache current y (as next step's previous)
            if self.nprev > 0:
                # shift everyone down, free first slot and empty last
                yprev[1:] = yprev[:-1]
                yprev[0] = y

            # subtract off average to enforce that perturbation is zero-mean
            if self.enforce_avg:
                perturb = perturb - perturb.mean()

            # perform update step
            y = yR + perturb

            # accumulate loss if needed
            if accumloss:
                if running_loss is None:
                    running_loss = loss_fn(y, y_target)
                else:
                    running_loss += loss_fn(y, y_target)

            # store series if needed
            if return_all:
                y_series.append(y)

        if accumloss:
            # divide by num passes to get something interpretable
            running_loss = running_loss / self.npasses
            return y, running_loss
        elif return_all:
            return torch.stack(y_series)
        else:
            return y


class TinyVNet(RecNetBase, torch.nn.Module):
    def __init__(self, dim_size, channels_in, channels_out, kernel_size=3, net_params=None):
        torch.nn.Module.__init__(self)
        RecNetBase.__init__(self)

        self.ks = kernel_size
        # use bigger kernel for first conv to capture more spatial extent
        self.ks_first = self.ks #+ 2

        # set number of channels for each layer
        self.din = channels_in
        self.dout = channels_out
        # fine and coarse scale number of channels
        self.d1 = 16
        self.d2 = self.d1 * 2

        # minimum factor to pad to (to make downsampling nicer)
        self.min_factor = 4

        # get size to pad to, making sure we have extra padding on each side
        pad_to = int(4 * math.ceil((dim_size + 4) / 4.0))

        padding = pad_to - dim_size

        pad_l = int(padding // 2)
        pad_h = int(padding - pad_l)

        self.pad_size = (pad_l, pad_h, pad_l, pad_h, pad_l, pad_h)

        self.pad_to_multiple = CircPad(self.pad_size)

        if net_params is not None:
            self.unserialize(net_params)

        # initial filter to extract features
        self.conv_begin_1 = ForwardBlock(self.ks_first, self.din, self.d1)
        self.conv_begin_2 = ForwardBlock(self.ks, self.d1, self.d1)

        # downsample and filter in lower space
        self.down_block = DownBlock(self.ks, self.d1, self.d2)

        # filter in low_space
        self.conv_down_1 = ForwardBlock(self.ks, self.d2, self.d2)
        self.conv_down_2 = ForwardBlock(self.ks, self.d2, self.d2)

        # upsample and filter in higher space
        self.up_block = UpBlock(self.ks, self.d2, self.d2)

        # filter again in high, combined space
        self.conv_end_1 = ForwardBlock(self.ks, self.d1 + self.d2, self.d2)
        self.conv_end_2 = ForwardBlock(self.ks, self.d2, self.d1, use_activ=False)


        # # prepare output
        # inception block at end instead of output block
        self.output_block = OutputBlock(self.ks,
                                        self.d1, self.dout)

        self.crop = ImageCrop(self.pad_size)

    def serialize(self, running_dict=None):
        net_params = RecNetBase.serialize(self, running_dict)

        #  save num channels
        net_params['d1'] = self. d1
        net_params['din'] = self.din
        net_params['d2'] = self. d2
        net_params['ks'] = self. ks

        return net_params

    def unserialize(self, net_params):
        #! Load params from dict BEFORE constructing layers
        RecNetBase.unserialize(self, net_params)

        #  load num channels
        self.d1 = net_params['d1']
        self.din = net_params['din']
        self.d2 = net_params['d2']
        self.ks = net_params['ks']

    def forward(self, x):
        # pad before processing
        x = self.pad_to_multiple(x)

        # pass through a first conv layer and store
        x = self.conv_begin_1(x)
        x = self.conv_begin_2(x)

        # downsample
        x_down = self.down_block(x)
        x_down = self.conv_down_1(x_down)
        x_down = self.conv_down_2(x_down)

        # upsample filtered
        x = self.up_block(x_down, xpass=x)

        # make sure we don't keep x2 around
        x2 = None

        # combine low- and high-pass information
        x = self.conv_end_1(x)
        x = self.conv_end_2(x)
        x = self.output_block(x)

        # crop after processing
        x = self.crop(x)

        return x


class SimpleLayerNet(RecNetBase, torch.nn.Module):
    def __init__(self, dim_size, channels_in, channels_out, kernel_size=3, net_params=None):
        torch.nn.Module.__init__(self)
        RecNetBase.__init__(self)
        self.ks = kernel_size

        self.din = channels_in
        self.d1 = 16
        self.d2 = 16
        self.d3 = 16
        self.dout = channels_out

        if net_params is not None:
            self.unserialize(net_params)

        # 3 conv layers plus a flattening
        self.conv_1 = torch.nn.Conv3d(
            in_channels=self.din, out_channels=self.d1, kernel_size=self.ks, stride=1)
        self.activ_1 = torch.nn.PReLU(num_parameters=self.d1)

        # second conv layer
        self.conv_2 = torch.nn.Conv3d(
            in_channels=self.d1, out_channels=self.d2, kernel_size=self.ks, stride=1)
        self.activ_2 = torch.nn.PReLU(num_parameters=self.d2)

        self.conv3 = torch.nn.Conv3d(
            in_channels=self.d2, out_channels=self.d3, kernel_size=self.ks, stride=1)

        # flatten to single channel
        self.convflat = torch.nn.Conv3d(
            in_channels=self.d3, out_channels=self.dout, kernel_size=1, stride=1)

        self.pad = CircPad(self.ks // 2)

    def serialize(self, running_dict=None):
        net_params = RecNetBase.serialize(self, running_dict)

        #  save num channels
        net_params['din'] = self.din
        net_params['d1'] = self.d1
        net_params['d2'] = self.d2
        net_params['d3'] = self.d3
        net_params['dout'] = self.dout
        net_params['ks'] = self. ks

        return net_params

    def unserialize(self, net_params):
        #! Load params from dict BEFORE constructing layers
        RecNetBase.unserialize(self, net_params)

        #  load num channels
        self.din = net_params['din']
        self.d1 = net_params['d1']
        self.d2 = net_params['d2']
        self.d3 = net_params['d3']
        self.dout = net_params['dout']
        self.ks = net_params['ks']

    def forward(self, x):
        # apply first conv layer
        x = self.pad(x)
        x = self.conv_1(x)
        x = self.activ_1(x)

        # # second conv
        x = self.pad(x)
        x = self.conv_2(x)
        x = self.activ_2(x)

        # third conv
        x = self.pad(x)
        x = self.conv3(x)

        # flatten channels
        x = self.convflat(x)

        return x
