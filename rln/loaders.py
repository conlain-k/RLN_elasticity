import torch
from torch.utils.data import Dataset, ConcatDataset
import os
import sys
import numpy as np

mdir_default = 'data/c_2/microstructures'
rdir_default = 'data/c_2/responses'


def loadDataset(base, which, mdir='data', rdir='data', discretize=False, *args, **kwargs):
    X = f"data/{base}_{which}.npy"
    y = f"data/{base}_{which}_strain.npy"
    # Wrap in concat because that makes things magically work?
    dataset = LocalizationDataset(X, y, discretize=discretize, *args, **kwargs)
    return dataset


class LocalizationDataset(Dataset):
    """ Load set of microstructure and strain responses from files """
    def __init__(self, mh_file, resp_file, reqgrad=True, normalize=False, discretize=True, H=2, ds = 31, full_tensor = False):
        print(f"Loading dataset with micro file {mh_file} and responses {resp_file}")
        # TODO take these procedurally
        self.H = H
        self.ds = ds

        self.mf = mh_file
        self.rf = resp_file

        self.normalize = normalize
        self.discretize = discretize
        self.full_tensor = full_tensor

        self.reqgrad = reqgrad

        self.mh = self._loadMicro()
        self.resp = self._loadResp()[:,:,:]

        print(self.mh.shape)
        print(self.resp.shape)

        self.mh = self.mh.astype(np.float32)
        self.resp = self.resp.astype(np.float32)

    def _loadMicro(self):
        mh = np.array(np.load(self.mf)).astype(np.float32)

        # skip discretization if we already have
        if self.discretize:
            mh = self.discretize_mh(mh)

        return mh

    def _loadResp(self):
        resp = np.load(self.rf).astype(np.float32)

        num_samp = resp.shape[0]

        # channel 0 is instance, channel 1,2,3 are x,y,z, channel 4 is component
        # swap component to the front of x,y,z
        if self.full_tensor:
            print(resp.shape)
            resp = resp.reshape(num_samp, self.ds,self.ds,self.ds, -1)
            print(resp.shape)
            resp = np.transpose(resp, axes=[0, 4,1,2,3])
    

            print(resp.shape)
            # take only xx component for now
            resp = resp[:,0]
            print(np.average(resp))

            print(resp.shape)
        else:
            resp = resp.reshape(-1,self.ds,self.ds,self.ds)


        if self.normalize:
            resp = resp / np.average(resp)

        return resp

    def __len__(self):
        """ Return the total number of samples """
        return self.mh.shape[0]

    def __getitem__(self, index):
        """ Get a requested structure-strain pair (or set thereof) """
        if torch.is_tensor(index):
            index = index.tolist()

        # Load data and get label
        X = self.mh[index]
        y = self.resp[index]

        # Convert to tensors if needed
        X = torch.as_tensor(X,).float().requires_grad_(self.reqgrad)
        y = torch.as_tensor(y,).float().requires_grad_(self.reqgrad)

        return X, y


    def discretize_mh(self, m_flat):
        """ Convert set of phase ids into indicator basis """
        # ensure each microstructure is 3d
        m_reshaped = m_flat.reshape((-1, self.ds, self.ds, self.ds))
        # output m^h for each microstructure
        m_big = np.zeros(
            (m_reshaped.shape[0], self.H, self.ds, self.ds, self.ds))

        # make sure we have the right number of phases
        unique_vals = np.unique(m_flat)
        assert(unique_vals.size == self.H)

        # get indicator basis
        for ind_m, mh_i in enumerate(m_reshaped):
            for ind_h, hi in enumerate(unique_vals):
                m_big[ind_m, ind_h, mh_i == hi] = 1

        return m_big



