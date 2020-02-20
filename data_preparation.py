# Author : Nihesh Anderson
# Date 	 : 14 Feb, 2020

"""
[ENTRY - Data Preparation]
Data preparation script that processes raw data to structured .mat files
"""

from src.constants import DATASET_TYPES, DATASET_ROOT

# Script Arguments
DATA_ROOT = DATASET_ROOT[3]
TYPE = DATASET_TYPES[3]

# Arguments that need not be tuned
TARGET = "./data/processed/" + TYPE

import os
import scipy.io
import numpy as np
import src.utils as utils
import torch

def PrepareAdelaideRMF(data_root, target):

	# Read the list of files from data_root path
	files = os.listdir(data_root)
	files.sort()

	# Read the mat file and save it to TARGET
	for i in range(len(files)):

		mat = scipy.io.loadmat(os.path.join(data_root, files[i]))

		# Data is contained in a key named 'data' and the corresponding ground truth information is in 'label'
		# data shape is 6 x num_correspondences (in homogenous coordinates)
		# gt shape is 1 x num_correspondences
		data = mat["data"]
		assert((not (data[2, :] == 0).any()) and (not (data[5, :] == 0).any()))
		data = data[[0, 1, 3, 4], :]
		gt = mat["label"]

		# Reshape the matrices into standard form
		data = data.transpose(1, 0)
		gt = gt.reshape(-1)

		# Normalise the input points
		left_pts, _ = utils.normalise2dpts(torch.tensor(data[:, :2]).float().cuda().unsqueeze(0))
		right_pts, _ = utils.normalise2dpts(torch.tensor(data[:, 2:]).float().cuda().unsqueeze(0))
		left_pts = left_pts.cpu().detach().numpy()[0]
		right_pts = right_pts.cpu().detach().numpy()[0]
		data = np.concatenate([left_pts, right_pts], axis = 1)

		# Create output mat file
		output_mat = {}
		output_mat["data"] = data
		output_mat["label"] = gt

		# Save mat file
		scipy.io.savemat(os.path.join(target, files[i]), output_mat, appendmat = False)

def PrepareSyntheticHomography(data_root, target):

	# Read the list of files from data_root path
	files = os.listdir(data_root)
	files.sort()

	# Read the mat file and save it to TARGET
	for i in range(len(files)):

		mat = scipy.io.loadmat(os.path.join(data_root, files[i]))

		# Data is contained in a key named 'X' and the corresponding ground truth information is in 'gt_data'
		# data shape is num_dim x num_correspondences
		# gt shape is 1 x num_correspondences
		data = mat["X"]
		gt = mat["gt_data"]

		# Reshape the matrices into standard form
		data = data.transpose(1, 0)
		gt = gt.reshape(-1)

		# Normalise the input points
		left_pts, _ = utils.normalise2dpts(torch.tensor(data[:, :2]).float().cuda().unsqueeze(0))
		right_pts, _ = utils.normalise2dpts(torch.tensor(data[:, 2:]).float().cuda().unsqueeze(0))
		left_pts = left_pts.cpu().detach().numpy()[0]
		right_pts = right_pts.cpu().detach().numpy()[0]
		data = np.concatenate([left_pts, right_pts], axis = 1)

		# Create output mat file
		output_mat = {}
		output_mat["data"] = data
		output_mat["label"] = gt

		# Save mat file
		scipy.io.savemat(os.path.join(target, files[i]), output_mat, appendmat = False)

def PrepareSyntheticLine(data_root, target):

	# Read the list of files from data_root path
	files = os.listdir(data_root)
	files.sort()

	# Read the mat file and save it to TARGET
	for i in range(len(files)):

		mat = scipy.io.loadmat(os.path.join(data_root, files[i]))

		# Data is contained in a key named 'data' and the corresponding ground truth information is in 'gt_data'
		# data shape is num_dim x num_correspondences
		# gt shape is 1 x num_correspondences
		data = mat["data"]
		gt = mat["gt_data"]

		# Reshape the matrices into standard form
		gt = gt.reshape(-1)

		# Create output mat file
		output_mat = {}
		output_mat["data"] = data
		output_mat["label"] = gt

		# Save mat file
		scipy.io.savemat(os.path.join(target, files[i]), output_mat, appendmat = False)

if(__name__ == "__main__"):

	os.system("rm -rf " + TARGET)
	os.makedirs(TARGET)

	if(TYPE == "SyntheticHomography"):
		PrepareSyntheticHomography(DATA_ROOT, TARGET)
	elif(TYPE == "SyntheticLine"):
		PrepareSyntheticLine(DATA_ROOT, TARGET)
	elif(TYPE == "AdelaideRMF_Homography"):
		PrepareAdelaideRMF(DATA_ROOT, TARGET)
	elif(TYPE == "AdelaideRMF_FM"):
		PrepareAdelaideRMF(DATA_ROOT, TARGET)
	else:
		print("Dataset type not found")
		exit(0)
