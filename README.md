# RLN_elasticity
Pytorch code used for the paper _Recurrent Localization Networks applied to the Lippmann-Schwinger Equation_ ([accepted manuscript here](https://arxiv.org/abs/2102.00063)). All software is contained in the `rln` directory, and a sample driver file is provided in `main.py`. Future development will be focused on generalizing this approach and will be conducted in a separate repository.

## Sample uses

### Evaluate each model on contrast-10 datasets

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
