# Author: Nihesh Anderson
# File	: CustomDataset.py
# Date  : 13 Feb, 2020

import os
from torch.utils.data import Dataset
import random
import scipy.io
from src.dgsac import DGSAC_Features, Residual_Features
from src.constants import SEED
import torch
import time

class DataReader(Dataset):

	def __init__(self, data_root, structure, num_samples, topk, num_hypothesis = 1000, verbose = False):

		"""
		Homography dataset reader - All operations within the dataset happens in doubles but values are returned in float by __getitem__()
		Input:
			data_root 			 - Path to the folder containing the homography .mat files
			structure 		 	 - The type of geometric structure the data corresponds to
							  	   Supported structures - "Homography"
			num_samples 		 - Number of data files to be loaded
			topk 				 - Number of best hypothesis to be considered in DGSAC encoding
			num_hypothesis 		 - Generates num_hypothesis many hypothesis for computing DGSAC features
			verbose 			 - Gives verbose results including feature generation computation time
		"""

		# Class variable assignment
		self.topk = topk
		self.structure = structure
		self.num_hypothesis = num_hypothesis
		self.verbose = verbose

		# Read files from the dataset folder and crop it to num_samples
		self.files = os.listdir(data_root)
		self.files.sort()
		self.files = self.files[:num_samples]

		# Append root path too each file name
		for i in range(len(self.files)):
			self.files[i] = os.path.join(data_root, self.files[i])
		random.Random(SEED).shuffle(self.files)

	def __len__(self):

		return len(self.files)

	def __getitem__(self, idx):

		file = self.files[idx]
		# Load the .mat file from disk
		mat = scipy.io.loadmat(file)

		# Read data and ground truth label
		data = torch.tensor(mat["data"]).double().cuda()
		label = torch.tensor(mat["label"]).cuda().view(-1)

		if(self.verbose):
			print("Computing features for every point using Density/Residual")
	
		feature_in_t = time.time()
		encoding = DGSAC_Features(data, self.topk, self.structure, self.num_hypothesis, self.verbose) 
		# encoding = Residual_Features(data, self.topk, self.structure, self.num_hypothesis)
		feature_out_t = time.time()

		if(self.verbose):
			print("{:40} : {time}".format(
					"Overall feature computation time",
					time = feature_out_t - feature_in_t
				))
			print("Feature generation complete")

		return data.float(), encoding.float(), label
		