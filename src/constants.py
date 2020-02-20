# Author : Nihesh Anderson 
# Date 	 : 14 Feb, 2020
# File 	 : Constants.py

"""
This file contains a list of constant variables across all the scripts in this repository. They need not be tuned
for experimentation purposes
"""

# Seed value for random number generator to ensure reproducibility
SEED = 0 													
# Small value for floating point calculations					
EPS = 1e-9
# Possible types of datasets supported by our implementation
DATASET_TYPES = ["SyntheticHomography", "SyntheticLine", "AdelaideRMF_Homography", "AdelaideRMF_FM"]
# Root folders for raw datasets
DATASET_ROOT = [
	"./data/raw/synthetic_multi_homography",
	"./data/raw/multi_2Dlines_dataset",
	"./data/raw/adelaidermf_homography",
	"./data/raw/adelaidermf_fm"
]
# MSS size for various structures - these must be the standard values - oversampling is not allowed
MSS = {
	"Homography": 5,	
	"Line": 2,
	"Plane": 3,
	"Fundamental": 7
}