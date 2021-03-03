from PIL import Image

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid

import numpy as np

from rln.helpers import computeImgErr

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'savefig.dpi': 300})

def arr2png(arr, path, size=1080):
    # min zero
    arr = arr - np.min(arr)
    # max one
    arr = arr / np.max(arr)

    im = Image.fromarray(np.uint8(cm.viridis(arr)*255))
    im = im.resize((size, size))
    im.save(path)
    return im

def addCBaxis(ax):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return cax


def plotResponses(actual, predicted, errors, worst_z=0):
    min_v = np.min([actual])
    max_v = np.amax([actual])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    c1 = axes[0].imshow(predicted[:, :, worst_z], vmin=min_v,
                        vmax=max_v, cmap='gray_r', origin='lower', aspect='auto')
    c2 = axes[1].imshow(actual[:, :, worst_z], vmin=min_v,
                        vmax=max_v, cmap='gray_r', origin='lower', aspect='auto')
    c3 = axes[2].imshow(errors[:, :, worst_z], vmin=0,
                        cmap='coolwarm', origin='lower', aspect='auto')

    [ax_.axis('off') for ax_ in axes.ravel()]

    axes[0].set_title("Predicted Strain Response")

    axes[1].set_title("FEA Strain Response")

    axes[2].set_title("Local Strain Error")

    plt.colorbar(c1, cax=addCBaxis(axes[0]), label="Normalized Strain")
    plt.colorbar(c2, cax=addCBaxis(axes[1]), label="Normalized Strain")
    plt.colorbar(c3, cax=addCBaxis(axes[2]), label="Strain Error (ASE)")

    plt.tight_layout(rect=[0, 0, 1, 0.9])

    return fig


def show_instance(mh, resp, savefig=False, mz=None, impath="images"):
    fig = plt.figure(figsize=(8, 6))

    axes = ImageGrid(fig, 111,
                     nrows_ncols=(1, 2),
                     axes_pad=0.15,
                     share_all=True,
                     cbar_mode=None,
                     cbar_pad=0, cbar_size='0%')

    ds = resp.shape[0]
    if mz is None:
        mz = ds//2
    axes[0].imshow(mh[1, :, :, mz], cmap='viridis', origin='lower')
    axes[1].imshow(resp[:, :, mz], cmap='gray_r', origin='lower')

    axes[0].set_title("Microstructure")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_title("Response")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    fig.tight_layout()

    if savefig:
        plt.savefig(f"{impath}/example_io.png", dpi=300)
        # arr2png(mh[:, :, mz, 1], f"{impath}/mh.png", ds * 32)
        # arr2png(resp[:, :, mz], f"{impath}/resp.png", ds * 32)

    return fig


def analyze_results(test_responses, test_predictions, savefig=False, impath="images", ind_override=None, mi_override=None):
    """ Compute error metrics and make plots """
    # print aggregate results
    print("min, max, avg predicted strain is {:.6f}, {:.6f}, {:.6f}".format(np.amin(test_predictions),
                                                                            np.amax(test_predictions), np.average(test_predictions)))

    # compute normalized errors
    rel_errs = computeImgErr(test_responses, test_predictions)

    # compute MASE
    MASE = np.average(100 * rel_errs, axis=(1, 2, 3))

    # average over each instance
    print("All instances: max local percent error is {:.3f}".format(
        np.amax(rel_errs) * 100))
    print("All instances: avg local percent error is {:.3f}".format(
        np.average(rel_errs) * 100))

    worst_pred = np.argmax(rel_errs)
    worst_ind, mx, my, mz = np.unravel_index(worst_pred, rel_errs.shape)

#    worst_ind = 4936
 #   mx, my, mz = (12, 18, 30)

    # print info for the worst instance
    worst_resp_pred = np.squeeze(test_predictions[worst_ind])
    worst_resp_actual = np.squeeze(test_responses[worst_ind])
    print("Worst instance: avg local percent error is {:.3f}".format(
          np.average(rel_errs[worst_ind]) * 100))

    print('Worst prediction is in instance {} at ({}, {}, {}), values there are predicted: {:.6f}, actual: {:.6f}'.format(
        worst_ind, mx, my, mz, worst_resp_pred[mx, my, mz], worst_resp_actual[mx, my, mz]))

    print(MASE.shape)

    print(f"mean: {np.average(MASE)}, std: {np.std(MASE)}")

    max_ASE = np.amax(MASE)

    spacing = 0.5

    bins = np.arange(0, max_ASE+spacing, spacing)

    fig = plt.figure(figsize=(8, 6))
    hist = plt.hist(MASE, bins=bins, edgecolor='k', linewidth=0.5, density=True)
    plt.xlabel("Percent MASE")
    plt.ylabel("Frequency")
    plt.title("MASE histogram across test set")
    plt.tight_layout()

    np.save(f"{impath}/mase.npy", MASE)

    # plot error histogram
    if savefig:
        plt.savefig(f"{impath}/err_hist.png", dpi=300)

    # plot response for worst instance
    respfig = plotResponses(worst_resp_actual, worst_resp_pred, rel_errs[worst_ind], worst_z=mz)
    if savefig:
        plt.savefig(f"{impath}/strain.png", dpi=300)

    # now plot errors
    errfig, ax = plt.subplots(1, 1)
    im = ax.imshow(rel_errs[worst_ind, :, :, mz], vmin=0,
                   cmap='coolwarm', origin='lower')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Local strain relative error")
    cb = errfig.colorbar(im)
    errfig.tight_layout()
    if savefig:
        plt.savefig(f"{impath}/err.png", dpi=300)

    return worst_ind, mz, errfig, respfig


def main():
    predictions = np.load("output/lhs_micro_big_21_50_predictions.npy") 
    true = np.load("data/lhs_micro_big_21_50_test_strain.npy")
    true = true[:,:,0]
    true = true.reshape(-1,51,51,51)

    true = true/ np.average(true)

    analyze_results(true, predictions, savefig=True)


if __name__ == "__main__":
    main()
