# RLN_elasticity
Pytorch code used for the paper [Recurrent Localization Networks applied to the Lippmann-Schwinger Equation](https://doi.org/10.1016/j.commatsci.2021.110356). All software is contained in the `rln` directory, and a sample driver file is provided in `main.py`. Future development focused on generalizing this approach will be conducted in a separate repository.

## Data availability
The data for this paper is freely available on (Mendeley data)[https://data.mendeley.com/datasets/v6dt8dwrh8/2]. A temporary version that is more shell-friendly is also (available on Dropbox)[https://www.dropbox.com/sh/pma0npf1wr86n9i/AADFc7xWNOe6WilrJQbSHC8Va], but may not be permanently available. A script to automatically download the Dropbox version and prepare it for evaluation is available in this repository's (data directory)[./data].

## Software overview
A brief discussion of this repository's software design is available in the (rln readme file)[./rln/README.md]

## Sample uses

### Evaluate each pretrained model configuration on contrast-10 datasets

FLN:
```
python main.py --eval --load models_trained/FLN_c10.model --model_type FLN --CR 10
```

RLN-t:
```
python main.py --eval --load models_trained/RLN_t_c10.model --model_type RLN-t --CR 10
```

RLN:
```
python main.py --eval --load models_trained/RLN_full_c10.model --model_type RLN --CR 10
```

### Train models from scratch
Train FLN using same configuration as paper: 
```
python main.py --model_type FLN --CR 10
```
