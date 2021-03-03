import torch

import numpy as np


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def dataset_to_np(dataset):
    # makes unzipping easier somehow?
    X, y = zip(*[(detachData(di[0]),
                  detachData(di[1])) for i, di in enumerate(dataset)])
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

# get every entry in a torch dataset
def detachData(data):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.asarray(data)
    assert not torch.is_tensor(data)
    return data

def computeImgErr(true, pred):
    ret = np.abs(true - pred) / np.average(true)
    return ret

