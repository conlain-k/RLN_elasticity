from numpy.fft import fftn, ifftn, fftshift
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
from matplotlib import pyplot as plt


from rln.helpers import *


class AVG_regressor():
    """ Just return average strain field """

    def __init__(self, H=2, dim_size=51):
        self.H = H
        self.ds = dim_size

        self.avg = 0

    def load_state_dict(self, dict):
        self.avg = dict['avg']

    def state_dict(self):
        return {'avg': self.avg}

    def try_add_figure(self, *args, **kwargs):
        return None

    def fit(self, dataset, *args, **kwargs):
        X_fit, y_fit = dataset_to_np(dataset)
        self.avg = np.average(y_fit)
        print(f"Avg is {self.avg:.8f}")

    def predict(self, X_test):
        X_test = X_test.reshape(-1, self.H, self.ds, self.ds, self.ds)

        y_pred = np.ones(
            (X_test.shape[0], self.ds, self.ds, self.ds)) * self.avg

        return y_pred


class OLS_regressor():
    def __init__(self, H=2, dim_size=51):
        self.H = H
        self.ds = dim_size
        self.stencil = np.zeros(
            (self.H, self.ds, self.ds, self.ds), dtype=np.complex128)

    def try_add_figure(self, *args, **kwargs):
        return None

    def fit(self, dataset, *args, **kwargs):
        # unzip dataset
        X_fit, y_fit = dataset_to_np(dataset)

        mh = X_fit.reshape(-1, self.H, self.ds, self.ds, self.ds)
        mh_ft = fftn(mh, axes=(2, 3, 4))
        resp = y_fit.reshape(-1, self.ds, self.ds, self.ds)
        resp_ft = fftn(resp, axes=(1, 2, 3))

        s0 = (slice(None), )

        # for each each frequency, fit in fourier space
        for ijk in np.ndindex(mh.shape[2:]):
            # average-term: fit both phases
            if np.sum(ijk) == 0:
                s1 = s0
            else:
                # else only fit high phase
                s1 = (slice(-1, 0, -1),)

            M = mh_ft[s0 + s1 + ijk]

            R = resp_ft[s0 + ijk]

            res, _, _, _ = np.linalg.lstsq(M, R, rcond=None)
            self.stencil[s1 + ijk] = res.squeeze()

        print("avg is {}".format(self.stencil[:, 0, 0, 0]))

    def load_stencil(self, stenc, avg):
        self.stencil[1, :, :, :] = fftn(stenc)
        self.stencil[0, :, :, :] = np.zeros_like(stenc)
        self.stencil[0, 0, 0, 0] = avg

    def get_stencil(self):

        stencil_real = ifftn(self.stencil, axes=(1, 2, 3)).real

        print("min, max stencil val is {}, {}".format(
            np.min(stencil_real), np.max(stencil_real)))
        return stencil_real

    def plot_stencil(self):
        stencil_real = self.get_stencil()

        fig, ax = plt.subplots(1, 1)

        stencil_show = stencil_real[1, :, :, 0]
        stencil_show = np.roll(stencil_show, self.ds // 2, axis=0)
        stencil_show = np.roll(stencil_show, self.ds // 2, axis=1)
        im0 = ax.imshow(stencil_show, cmap='coolwarm', origin='lower')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Influence coefficients, phase 1")
        cb0 = fig.colorbar(im0, cax=addCBaxis(ax))
        plt.tight_layout()

        return stencil_show

    def predict_batch(self, batch):
        fft_mul = np.squeeze(fftn(batch, axes=(2, 3, 4)))

        stenc = self.stencil[np.newaxis]

        # convolve with stencil
        resp_ft = stenc * fft_mul
        resp_ft = np.sum(resp_ft, axis=1)

        resp_real = ifftn(resp_ft, axes=(1, 2, 3)).real.squeeze()
        return resp_real

    def predict(self, X_test):
        batches = np.array_split(X_test, 5, axis=0)
        resps = [self.predict_batch(batch) for batch in batches]
        return np.concatenate(resps, axis=0)
