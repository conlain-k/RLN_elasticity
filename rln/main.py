import argparse
import sys
import os
# we needs numpy
import numpy as np

from rln.conv import RLN_localizer
from rln.helpers import dataset_to_np
from rln.loaders import loadDataset
from rln.analyze_data import computeImgErr


# setup parser
parser = argparse.ArgumentParser(description="Solve linear elasticity via MKS")
parser.add_argument('--eval', action='store_true',
                    help='If true, skip fitting and just evaluate (default False)')

parser.add_argument('--no_cuda', action='store_true',
                    help='If true, disable CUDA (default False)')

parser.add_argument('--noreuse', action='store_true',
                    help='If false, reuse same net across iterations. If true, create <npasses> different proximal networks')
parser.add_argument('--model_type',
                    choices=['FLN', 'RLN-t', 'RLN'],
                    default="RLN-t",
                    help='Which model type to use (default "RLN-t").')
parser.add_argument('--CR', nargs=1, help='Contrast ratio for dataset to load (default "10")',
                    choices=['10', '50'], default='10')

parser.add_argument('--load', nargs=1, help='Saved model to load')

########################################
# DEFAULT CONFIGURATION
# The choices below will reconstruct the results from the paper

dim_size = 31  # use 31x31x31 microstructures
H = 2  # use 2-phase microstructure
npasses = 5  # 5 passes per iteration
nepochs = 60  # 60 epochs for training

########################################


def setup_FLN(model_to_load, CR):
    # 1 pass of the proximal operator
    return RLN_localizer(
        H, dim_size, load_model=model_to_load,
        use_cuda=True, reuse_net=True, npasses=1, savename=f"FLN_c{CR}")


def setup_RLN_t(model_to_load, CR):
    # same as FLN but run for several passes
    return RLN_localizer(
        H, dim_size, load_model=model_to_load,
        use_cuda=True, reuse_net=True, npasses=npasses, savename=f"RLN_t_c{CR}")


def setup_RLN(model_to_load, CR):
    # same as RLN-t but don't reuse network
    return RLN_localizer(
        H, dim_size, load_model=model_to_load,
        use_cuda=True, reuse_net=False, npasses=npasses, savename=f"RLN_full_c{CR}")


def main():
    print("Starting run!")
    seed = None
    np.random.seed(seed)

    args = parser.parse_args()

    # should we load model parameters from a file?
    model_to_load = args.load
    if model_to_load is not None:
        model_to_load = model_to_load[0]

    no_cuda_timers = args.no_cuda

    # train or test?
    eval_mode = args.eval

    CR = args.CR
    if CR is not None:
        CR = CR[0]
    else:
        CR = "10"  # default to CR 10
    dataset_file_base = f"31_c{CR}"

    # build the correct model
    if args.model_type == 'FLN':
        model = setup_FLN(model_to_load, CR)
    elif args.model_type == "RLN-t":
        model = setup_RLN_t(model_to_load, CR)
    else:
        model = setup_RLN(model_to_load, CR)

    # train if we're not in eval mode
    if not eval_mode:
        print("Loading train data! This may take a bit ...")

        dataset_train = loadDataset(dataset_file_base, "train", normalize=True)

        dataset_valid = loadDataset(dataset_file_base, "valid", normalize=True)

        print("Data loaded!")
        print("Training on {} instances!".format(len(dataset_train)))
        print("Validating on {} instances!".format(len(dataset_valid)))
        print("Fitting model!")

        model.fit(dataset_train, dataset_valid, nepochs=nepochs)

        print("Clearing out training memory!")
        # now clear memory for the testing set
        del dataset_train
        del dataset_valid

    # load testing dataset
    dataset_test = loadDataset(dataset_file_base, "test", normalize=True)

    # dump them back into numpy for analysis later
    X_test, y_test = dataset_to_np(dataset_test)
    print("Test set size:", X_test.shape, y_test.shape)

    print("Testing model!")
    # start timer if we have CUDA
    if not no_cuda_timers:
        import torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

    # run model on test instances
    y_hat = model.predict(X_test)

    # stop timer if we have CUDA
    if not no_cuda_timers:
        end.record()

        # Wait for everything to finish running for timing
        torch.cuda.synchronize()
        print(f"Predicting took: {start.elapsed_time(end)} ms")

    # make sure predictions match the right shape
    print(y_hat.shape, y_test.shape)

    print("Testing RMSE percent loss is {}".format(
        model.computePredictionLoss(y_hat, y_test)))

    # compute MASE errors
    rel_errs = computeImgErr(y_hat, y_test)

    # compute MASE
    MASE = np.average(100 * rel_errs, axis=(1, 2, 3))
    print(f"Testing MASE error is mean: {np.average(MASE)}, std: {np.std(MASE)}")

    os.makedirs("data/predictions/", exist_ok=True)
    np.save(f"data/predictions/{dataset_file_base}_predictions.npy", y_hat)

    return model


if __name__ == "__main__":
    main()
