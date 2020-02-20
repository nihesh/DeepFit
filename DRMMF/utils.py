
# Author : Nihesh Anderson
# Date 	 : Feb 18, 2020
# File 	 : utils.py

import numpy as np
import torch
import scipy.signal
import math

def residual_to_density(residual):

	"""
	This module takes residual matrix as input and computes the corresponding density matrix
	Input:
		residual 	- [num_hypothesis x num_correspondences] 2D tensor
	Output:
		density 	- [num_hypothesis x num_coorrespondences] 2D tensor
	"""

	EPS = 1e-3

	num_hypothesis = residual.shape[0]
	num_correspondences = residual.shape[1]

	# Sort the residuals based on magnitude
	residual, preferences = torch.sort(residual, dim = 1)
	residual = residual.cpu().detach().numpy()
	residual = residual + EPS

	# Smoothen the residual vector as proposed in "Density Guided Sampling and Consensus"
	filter_width = math.ceil(0.025 * num_correspondences)
	edge = filter_width // 2

	# Step 1: Pad the residual vector with edge values along dim 1 borders
	residual = np.concatenate([np.flip(residual[:, :edge], axis = 1), residual, residual[:, -edge:]], axis = 1)

	# Step 2: Convolve along dim 1 using box filter
	box_filter = np.ones((1, filter_width)) / filter_width
	residual = scipy.signal.convolve2d(residual, box_filter, mode = "valid")
	residual = torch.tensor(residual).double().cuda()
	
	# Step 3: Compute density_i_j = j / residual_i_j
	numerator = torch.tensor([[i for i in range(1, num_correspondences + 1)]]).double().cuda().repeat(num_hypothesis, 1)
	density = (numerator / residual)

	# Reorder the densities to negate residual sorting effect
	density = density.scatter(1, preferences, density)
	
	return density