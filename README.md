# CONC: Complex-noise-resistant Open-set Node Classification with Adaptive Noise Detection
## Introduction
The source code and models for our paper ROG_PL: Robust Open-Set Graph Learning via Region-Based Prototype Learning
## Framework
![image](https://github.com/Iewoaixil/CONC/blob/main/framework.jpg)

## Installation
Before to execute CONC, it is necessary to install the following packages:

* torch
* torch_geometric
* networkx
* matplotlib
* scipy
* numpy
* sklearn
* pyparsing

## Overall Structure

The project is organised as follows:

* `datasets/`contains the necessary dataset files;
* `idx/` contains the noisy dataset index;
* `config/` contains the necessary dataset config;
* `utils/`contains the necessary processing subroutines;
* * `model/`contains the model related files.
* `Results/`save run results.


## Basic Usage

### For train
```shell
python train.py
```

### For test
```shell
python test.py
```


