import time
from enum import Enum

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from rln.helpers import count_parameters
from rln.model_types import *

import sys, os

# which mode is the net running in (used in runpass)
class runMode(Enum):
    TRAIN = 1
    VALID = 2

def custom_loss(output, target):
    """ Use scaled RMSE: use MSE loss function, but square-root and re-scale to output as percentage """
    mseloss = torch.nn.MSELoss()
    target_mean = torch.mean(target)

    loss = mseloss(output.squeeze(), target.squeeze())
    loss = torch.sqrt(loss) / target_mean
    loss = loss * 100
    return loss

class RLN_localizer():
    """ Wrapper for CNN regression model for two-phase elasticity """

    def __init__(self, H, dim_size, learning_rate=0.002, load_model=None, use_cuda=True, npasses=1, nprev=2, reuse_net=True, savename=None):
        self.H = H
        self.ds = dim_size

        # use cuda if we can
        if use_cuda and torch.cuda.is_available():
            self.DEVICE = torch.device("cuda:0")
        else:
            self.DEVICE = torch.device("cpu")

        print("Device is {}".format(self.DEVICE))

        # net initialization info
        self.checkpt = None
        net_params = None
        self.n_epochs_elapsed = 0
        self.helper = None

        # what type of net to build?
        model_type = TinyVNet
        # model_type = SimpleLayerNet

        # load checkpoint if one given
        if load_model is not None:
            print("Loading saved model {}".format(load_model))
            self.checkpt = torch.load(load_model, map_location=self.DEVICE)
            self.n_epochs_elapsed = int(self.checkpt['n_epochs_elapsed'])
            net_params = self.checkpt['net_params']
            model_type = net_params['model_type']

            # if no savename provided, try using one from the checkpoint
            if savename is None:
                savename = self.checkpt.get('save_name')

        # construct net
        self.net = RecurrentLocalizationNet(H, dim_size, model_type=model_type, reuse_net=reuse_net, npasses=npasses, nprev=nprev,
                                            net_params=net_params)

        # now load state dict
        if self.checkpt is not None:
            self.net.load_state_dict(self.checkpt['model_state_dict'])

        # send to right location
        self.net = self.net.float().to(self.DEVICE)

        # Optimizer
        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=learning_rate)

        # now initialize optimizer
        if self.checkpt is not None:
            self.optimizer.load_state_dict(
                self.checkpt['optimizer_state_dict'])

        # LR scheduler
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, factor=0.5, patience=2, verbose=True)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=60, eta_min=1e-8)
            
        if self.checkpt is not None:
            self.scheduler.load_state_dict(
                self.checkpt['scheduler_state_dict'])

        # how often to checkpoint?
        self.savefreq = 5

        # do mini-batch training
        self.batch_size = 32

        # what value to clip gradient norms to?
        self.grad_clip_mag = 2 * self.net.npasses

        # what's the best loss we've seen?
        self.best_loss = np.infty

        # print diagnostic info
        num_params = count_parameters(self.net)
        print(f"Model structure: \n{str(self.net)}")
        print(f"Net has {num_params} parameters, {self.net.npasses} passes")
        print()

        # if name is none, come up with one
        self.savename = savename or f"{self.net.name()}_{num_params}_{self.net.npasses}"

    def runpass(self, data_loader, mode):
        """ run one iteration on the given data loader, only doing backprop if mode is TRAIN """
        running_loss = 0

        # print some diagnostic info
        nbatches = len(data_loader)
        printfreq = max(int(nbatches//5), 1)
        bstr = "training" if mode == runMode.TRAIN else "validation"
        print(f"Running {nbatches} batches ({bstr}): ", end='', flush=True)

        for ind, batch in enumerate(data_loader):
            if (ind % printfreq == printfreq - 1):
                print(ind + 1, end=',', flush=True)

            # empty gradients if training
            if mode == runMode.TRAIN:
                self.optimizer.zero_grad()

            # build minibatch
            mini_x, mini_y = batch

            # torch is weird about these
            mini_x = mini_x.squeeze(1).float().to(self.DEVICE)
            mini_y = mini_y.squeeze(1).float().to(self.DEVICE)

            # run forward pass
            out, rloss = self.net(mini_x, accumloss=True,
                                  y_target=mini_y, loss_fn=custom_loss)

            # do backprop
            if mode == runMode.TRAIN:
                rloss.backward()
                # clip gradients just to keep them sane
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.grad_clip_mag)
                self.optimizer.step()

            # accumulate loss (averaged over batches)
            running_loss += float(rloss.item()) / nbatches

        # newline at end of epoch
        print()
        return running_loss

    def fit(self, dataset_train, dataset_valid, nepochs=20):
        """ Fit network to given training dataset"""
        if self.helper:
            self.helper.fit(dataset_train)

        # make sure model exists
        os.makedirs("models/", exist_ok=True)

        # make dataloader for training data
        loader_train = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)

        loader_valid = DataLoader(
            dataset_valid, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0)

        # keep track of our timing and progress
        start_time = time.time()
        start = self.n_epochs_elapsed
        stop = self.n_epochs_elapsed + nepochs

        train_losses = [None] * nepochs
        valid_losses = [None] * nepochs

        # alright lets start epochs
        for epoch in range(start, stop):
            self.n_epochs_elapsed = epoch

            # run training pass
            self.net.train()
            torch.set_grad_enabled(True)
            tloss = self.runpass(data_loader=loader_train, mode=runMode.TRAIN)


            # compute validation loss after training
            self.net.eval()
            # run validation pass
            with torch.no_grad():
                vloss = self.runpass(
                    data_loader=loader_valid, mode=runMode.VALID)

            dt = time.time() - start_time

            # print debug stuff for epochs
            printstr = f"Epoch {epoch + 1:3d}: elapsed time: {dt:4.2f}, LR: {self.optimizer.param_groups[0]['lr']}, train loss: {tloss:.4f}, validation loss: {vloss:.4f}"

            print(printstr)

            # do a sched step every epoch
            if type(self.scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(vloss)
            else:
                self.scheduler.step()

            # should we save the model?
            save_model = (epoch % self.savefreq == self.savefreq - 1)

            # always save if validation loss decreased
            if vloss <= self.best_loss:
                self.best_loss = vloss
                save_model = True

            if save_model:
                print(
                    f"Saving model, best loss so far is {self.best_loss:.4f}")
                self.saveModel(f"models/{self.savename}_{epoch+1}.model")

            # cache losses for output later
            train_losses[epoch] = tloss
            valid_losses[epoch] = vloss

        return train_losses, valid_losses

    def saveModel(self, name):
        """ save all model parameters to a file """
        state_dict = dict()
        state_dict['model_state_dict'] = self.net.state_dict()
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        state_dict['scheduler_state_dict'] = self.scheduler.state_dict()
        state_dict['n_epochs_elapsed'] = self.n_epochs_elapsed + 1

        state_dict['net_params'] = self.net.serialize()
        state_dict['save_name'] = self.savename

        torch.save(state_dict, name)

    def predict(self, X_test):
        """ Predict y for a given x (no backprop) """

        # get data into right form
        if not torch.is_tensor(X_test):
            X_test = torch.as_tensor(X_test).float().to(self.DEVICE)

        # dump into loader for batching
        loader_test = DataLoader(
            X_test, batch_size=self.batch_size, shuffle=False, pin_memory=False, num_workers=0)

        # store output for each batch
        out = [None] * len(loader_test)

        nbatches = len(loader_test)
        printfreq = max(int(nbatches//5), 1)

        print("Running {} batches (testing): ".format(
            nbatches), end='', flush=True)

        # now evaulate network on each batch
        self.net.eval()
        with torch.no_grad():
            for ind, batch in enumerate(loader_test):
                if (ind % printfreq == printfreq - 1):
                    print(ind + 1, end=',', flush=True)
                pred = self.net(batch, accumloss=False)
                out[ind] = pred.detach().cpu().float()

        # flatten batches
        ret = torch.cat(out, 0).numpy().astype(np.float32).squeeze()

        return ret

    def get_evolution(self, x, y, mz):
        """ Get evolution of predicted y for a given x between iterations in x-y slice at z-voxel mz """
        print("mz is", mz)
        x = torch.as_tensor(x).float().to(self.DEVICE)
        x = x.reshape(1, 2, self.ds, self.ds, self.ds)
        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(x, accumloss=False, return_all=True)

        xp = x.detach().cpu().numpy().squeeze()[1, :, :, mz]
        y_pred = y_pred.detach().cpu().numpy().squeeze()

        print(y_pred.shape)
        

        return y_pred

    def computePredictionLoss(self, y1, y2):
        """ Compute loss between two predictions (no backprop) """
        self.net.eval()
        with torch.no_grad():
            y1 = torch.as_tensor(y1).squeeze().float().to(self.DEVICE)
            y2 = torch.as_tensor(y2).squeeze().float().to(self.DEVICE)

            print(y1.shape, y2.shape, y1.type(), y2.type())
            loss = custom_loss(y1, y2).item()
            return loss
