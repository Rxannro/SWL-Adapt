# SWL-Adapt
Official implementation of SWL-Adapt.

![SWL-Adapt_framework](https://user-images.githubusercontent.com/105044070/167120684-c0d2e0d9-872a-4acd-81b2-a887f3f0db1e.png)

## Dependencies

* python 3.7
* torch == 1.8.0 (with suitable CUDA and CuDNN version)
* higher (https://pypi.org/project/higher/)
* numpy, torchmetrics, scipy, pandas, argparse, sklearn

## Datasets

| Dataset | Download Link |
| -- | -- |
| RealWorld | https://sensor.informatik.uni-mannheim.de/#dataset_realworld |
| OPPORTUNITY | https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition |
| PAMAP2 | https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring |

## Quick Start

Data preprocessing is included in main.py. Download the datasets and run SWL-Adapt as follows. This gives the performance of each evaluation with each user in the set of new users as the new user, and their average.
```
python --data_path /path/to/dataset --dataset [realWorld, OPPORTUNITY, or PAMAP2] main.py 
```
