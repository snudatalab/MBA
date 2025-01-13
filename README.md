# MBA

This project is a pytorch implementation of Multi-Behavior Sequence Aware Recommendation via Graph Convolution Networks (MBA).

## Overview

MBA is an accurate multi-behavior recommendation framework that exploits interaction transfer between behaviors, behavior-aware attention networks, and a novel sampling for BPR.
This project provides executable source code with adjustable hyperparameters as arguments and preprocessed datasets used in the paper.

## Prerequisites

Our implementation is based on Python 3.9 and Pytorch 1.9.0. Please see the full list of packages required for our codes in `requirements.txt`.

## Datasets

We use 3 datasets in our work: Tmall, Jdata, and Beibei. Preprocessed data are included in the `data` directory. Unpack zip files in `data` directory before running the code.

| Dataset | Users | Items | View | Collect | Cart | Buy |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Tmall | 41,738 | 11,953 | 1,813,498 | 221,514 | 1,996 | 181,428 | 
| Jdata | 93,334 | 24,624 | 1,600,973 | 39,663 | 41,262 | 234,691 | 
| Beibei | 21,716 | 7,997 | 2,412,586 | - | 642,622 | 304,576 | 


## Running the code

You can train the model by running `python src/main.py`.
You can change the hyperparameters by modifying the arguments of `main.py`.
Also, you can run a demo script `demo.sh` that reproduces the experimental results of MBA in Beibei dataset.
