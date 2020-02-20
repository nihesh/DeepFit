Deep Fit
========

This is the Python-3.6 implementation of our paper *Deep Robust Multi-Model Fitting* submitted to *ECCV 2020*

Overview
--------

At a granular level, our implementation has 3 components:

| Component | Description | Point of Entry |
| --------- | ----------- | -------------- |
| Data Preparation | Processes raw dataset and generates .mat files | data_preparation.py |
| NN Training | Reads the data from .mat files and starts training the model | train.py |
| Inference Generation | Uses the trained model and generates inferences on test data | inference.py |

Dependencies
------------

* torch-1.2.0
* torchvision-0.4.0
* numpy-1.17.2
* sklearn-0.21.3
* matplotlib-3.0.2

Data Preparation
----------------

> `$ python3 data_preparation.py`

In this phase, we read the raw dataset folder and convert the raw data into structured .mat files, one for each sample. 
Each .mat file is a dictionary containing the following keys:

| Key | Value |
| --- | ----- |
| data | 2D numpy array of shape [num_points x dim] containing the data |
| label | 1D numpy array of shape [num_points] containing label information. 0 corresponds to gross outliers and 1...C corresponds to different structures |

The script arguments can be modified within `data_preparation.py`. The following arguments can be tuned:

| Argument | Functionality | Values |
| -------- | ------------- | ------ |
| data_root | Path to the raw data folder | - |
| type | The raw folders are processed differently depending on the type of the dataset | SyntheticHomography<br> SyntheticLine<br> Adelaide-RMF |

Model Training
--------------

> `$ python3 train.py`

The script arguments can be modified within `train.py`. The following arguments can be tuned:

| Argument | Functionality | Values |
| -------- | ------------- | ------ |
| data_root | Path to the processed dataset folder - folder containing .mat files | String |
| type | Type of the model to be fitted | Homography<br> Line<br> Plane<br> Fundamental<br> |
| num_samples | Number of training samples to consider for training | Integer | 
| topk | The top k hypothesis to be selected after sorting based on density or residual | Integer |
| num_hypothesis | Number of hypothesis generated for density or residual estimation | Integer |
| feature_map_scale | Factor by which number of feature maps are scaled down in pointnet | Float |
| batch_size | Batch size for pointnet | Integer |
| num_workers | Number of threads for data loader | Integer |
| learning_rate | Learning rate of the model | Float |
| epochs | Number of training epochs | Integer |
| save_path | Path where simulation results and plots should be saved | String |
