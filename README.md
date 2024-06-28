## CKSP (Cross-species Knowledge Sharing and Preserving)

This repository is an official PyTorch implementation of the paper "CKSP: Cross-species Knowledge Sharing and Preserving for Universal Animal Activity Recognition"

## Requirements

This is my experiment environment
- python3.7
- pytorch+cuda10.2

## Details
### 1. Dataset
We used three public datasets collected from horses (Kamminga et al., 2019a), sheep (Kleanthous et al., 2022a) and cattle (C. Li et al., 2021), respectively.
Data links:
Horses: https://data.4tu.nl/articles/_/12687551/1
Sheep: https://github.com/nkleanthous2015/Sheep_activity_Data
Cattle: https://zenodo.org/records/5849025#.ZE-y_3ZByHu

### 2. Train the model
The training, validation, and testing processes can be found in train.py.
The parameters can be revised in training_script.sh.
