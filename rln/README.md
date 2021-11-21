# Software overview

This file presents a brief overview to the repository's software design. Hopefully this is useful for anyone trying to replicate or modify this project.

All RLN-related code is stored in this directory. Below is a brief discussion of each file. For further reference, see the in-code documentation.

### 

- conv.py
    - Contains implementation for a callable, trainable model under the `RLN_localizer` class. Also contains code for fitting, evaluating, saving/loading and some visualization. Holds a PyTorch network internally. Design is similar to `scikit-learn`, but not exactly the same. NOTE: this file contains a large amount of boilerplate and custom implementation that is not really useful. In more recent work I've switched to [PyTorch-Lightning](https://www.pytorchlightning.ai/), which removes the need for most of this file
- helpers.py
    - Contains a few helper functions related to data manipulation and processing. This file was used more during development, and its functions might be best moved to the few places they are actually used.
- __init__.py
    - Empty file required to make the `rln` directory a proper package.
- loaders.py
    - Contains code for data preprocessing and loading. Note that `.npy` files are used for input/output. In later work I have switched to `hdf5` (via [h5py](https://www.h5py.org/)) and have found it much nicer. 
- model_types.py
    - Contains the actual RLN implementation, along with a large amount of boilerplate code for data processing and model/hyperparameter saving/loading. Again, a lot of this can be swapped out for PT Lightning. The `RecurrentLocalizationNet` class represents the full recurrent model implementation, whereas `TinyVNet` and `SimpleLayerNet` are two possible proximal operators that are called within the RLN. 
- net_defs.py
    - Contains different building blocks used in all of the network designs, along with a base class (`RecNetBase`) for model/hyperparameter saving/loading. 