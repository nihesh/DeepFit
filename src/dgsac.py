# Author : Nihesh Anderson
# Date 	 : 13 Feb, 2020
# File 	 : dgsac.py

import numpy as np
import cv2
import scipy.signal
from src.constants import EPS, MSS
import src.utils as utils
import torch
import src.geometry as geometry
import time
import math

"""
Implementation of dgsac and accelerated hypothesis generation ideas
To support new structures, include the following methods to geometry.py:
	FitStructure(mss) - Similar to FitHyperplane, FitHomography, etc
	StructureResidual(data, hypothesis) - Similar to HyperplaneResidual, HomographyResidual, etc
Now, add a new TYPE in train.py and refer this type to these methods in Residual() and GenerateRandomHypothesis() methods
"""

def GenerateRandomHypothesis(data, weights, structure, num_hypothesis):

	"""
	This module generates num_correspondences many hypothesis fitted on MSS by fixing one point and generating the rest uniformly at random
	Input:
		data 			- [num_correspondences x dim] data tensor (dim = 4 for homography, 3 for planes, etc)
		weights 		- [num_correspondences] weight list or numpy array
		structure 		- string containing the type of structure to fit (See dataset init method for more information)
		num_hypothesis  - Generates num_hypothesis many hypothesis for computing DGSAC features
	Output:
		hypothesis 		- [num_hypothesis x d0 x d1 x ...] tensor denoting the parameters of the structure (d_i = 3 x 3 homography, 4 for planes, etc) 
	"""

	weights = np.asarray(weights)

	# Define sizes
	num_correspondences = data.shape[0]
	mss_size = MSS[structure]

	# Permute the data so that the remainder of fixed points are sampled uniformly at random
	permutation = np.asarray([i for i in range(num_correspondences)])
	np.random.shuffle(permutation)
	data = data[permutation]

	batch_mss = []

	i = 0
	# Every iteration generates the set of points in mss - subsequently, hypothesis is fitted in a vectorized fashion
	while(i < num_hypothesis):

		fixed_pt = i % num_correspondences

		# Set weight of the fixed point to 0
		old_fixed_wt = weights[fixed_pt]
		weights[fixed_pt] = 0
		scale = np.sum(weights)

		# Sample mss without replacement
		mss = np.random.choice(num_correspondences, mss_size - 1, replace = False, p = weights / scale)
		mss = np.concatenate([mss, [fixed_pt]])
		mss = data[mss]
		batch_mss.append(mss.unsqueeze(0))

		# Restore weight of fixed pt
		weights[fixed_pt] = old_fixed_wt

		i += 1

	batch_mss = torch.cat(batch_mss, dim = 0)

	# Fit geometric structure on mss and add the matrix to the collection
	if(structure == "Homography"):
		hypothesis = geometry.FitHomography(batch_mss)
	elif(structure == "Line" or structure == "Plane"):
		hypothesis = geometry.FitHyperplane(batch_mss)
	elif(structure == "Fundamental"):
		hypothesis = geometry.FitFundamentalMatrix(batch_mss)
	else:
		print("Error: Structure not found")
		exit(0)

	return hypothesis

def Residual(data, hypothesis, structure):

	"""
	Returns a matrix representing the residuals for all ordered pairs - an interface for computing residuals for different structures
	Input:
		data 		- [num_correspondences x dim] data matrix (dim = 4 for homography, 3 for planes, etc)
		hypothesis	- [num_hypothesis x d0 x d1 x ....] tensor denoting the parameters of the structure (d_i = 3 x 3 homography, 4 for planes, etc) 
		structure 	- string containing the type of structure to fit (See dataset init method for more information)
	Output:
		residuals 	- [num_correspondences x num_hypothesis] residual matrix

	Output Explanation:
		residual[i][j] is the residual of ith correspondence and jth hypothesis
		It is computed by transforming the input point and measuring the L2 distance between the transformed point and correspondence
	"""

	if(structure == "Homography"):
		residual = geometry.HomographyResidual(data, hypothesis)
	elif(structure == "Line" or structure == "Plane"):
		residual = geometry.HyperplaneResidual(data, hypothesis)
	elif(structure == "Fundamental"):
		residual = geometry.FundamentalMatrixResidual(data, hypothesis)
	else:
		print("Error: Structure not found")
		exit(0)

	return residual

def residual_to_density(residual):

	"""
	This module takes residual matrix as input and computes the corresponding density matrix
	Input:
		residual 	- [num_hypothesis x num_correspondences] 2D tensor
	Output:
		density 	- [num_hypothesis x num_coorrespondences] 2D tensor
	"""

	global EPS

	num_hypothesis = residual.shape[0]
	num_correspondences = residual.shape[1]

	# Sort the residuals based on magnitude
	residual, preferences = torch.sort(residual, dim = 1)
	residual = residual.cpu().detach().numpy()
	residual = residual + EPS

	# Smoothen the residual vector as proposed in "Density Guided Sampling and Consensus"
	filter_width = int(0.025 * num_correspondences)
	filter_width = filter_width + (not (filter_width & 1))	 		# Ensures filter width is odd

	# Step 1: Pad the residual vector with edge values along dim 1 borders
	residual = np.pad(residual, [(0, 0), (filter_width // 2, filter_width // 2)], mode = "edge")

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

def DGSAC_Features(data, topk, structure, num_hypothesis, verbose):

	"""
	[This module doesn't use tensor operations everywhere]
	This module takes point correspondences as input and returns dgsac features
	Input:
		data 				 - [num_correspondences x dim] data matrix (dim = 4 for homography, 3 for planes, etc)
		topk 				 - Number of nearest hypothesis to consider, for creating the encoding
		structure 			 - string containing the type of structure to fit (Homography, Plane, Line, etc)
		num_hypothesis 		 - Generates num_hypothesis many hypothesis for computing DGSAC features
		verbose 			 - prints computation time of different modules in this section
	Output:
		encoding 			 - [num_correspondences x num_correspondences] encoding matrix for each data point

	Output explanation:
		For the entire data matrix, we compute the density vector (from ordered residuals).
		We pick the top k density hypothesis. 
		encoding[i][j] = | density[i][j] 			if the jth hypothesis is in top-k
						 | 0 						otherwise
		In each row of the encoding matrix, there are exactly k non zero entries
	"""

	num_correspondences = data.shape[0]

	# Uniform sampling
	weights = [1 for i in range(num_correspondences)]

	# Generate num_correspondences many hypothesis fitted on random subsets of mss correspondences  - bottleneck
	hyp_gen_in_t = time.time()
	hypothesis = GenerateRandomHypothesis(data, weights, structure, num_hypothesis)
	hyp_gen_out_t = time.time()
	
	if(verbose):
		print("{:40} : {time}".format(
				"Time taken to generate random hypothesis",
				time = hyp_gen_out_t - hyp_gen_in_t
			))

	# generate residual vector for all the correspondences - residual[i][j] denotes the residual between ith point and jth hypothesis
	residual = Residual(data, hypothesis, structure)

	# Transpose residual and density vectors (residual_to_density operates on the transposed version of our convention)
	residual = residual.permute(1, 0)
	density = residual_to_density(residual)
	density = density.permute(1, 0)

	density = -density
	density, preferences = torch.sort(density, dim = 1)
	# Restoring original density
	density = -density

	# Step 4: Extract top k preferred hypothesis
	preferences = preferences[:, :topk]
	density = density[:, :topk]
	density[:, :] = 1

	# Step 5: Create encoding
	encoding = torch.zeros([num_correspondences, num_hypothesis]).double().cuda()
	encoding = encoding.scatter(1, preferences, density)

	return encoding

def Residual_Features(data, topk, structure, num_hypothesis, verbose):

	"""
	This module takes point correspondences as input and returns residual features
	Input:
		data 				 - [num_correspondences x 4] data tensor
		topk 				 - Number of nearest hypothesis to consider, for creating the encoding
		structure 			 - string containing the type of structure to fit (See dataset init method for more information)
		num_hypothesis 		 - Generates num_hypothesis many hypothesis for computing DGSAC features
		verbose 			 - prints computation time of different modules in this section
	Output:
		encoding 			 - [num_correspondences x num_correspondences] encoding matrix for each data point

	Output explanation:
		For the entire data matrix, we compute the residual vector (from ordered residuals).
		We pick the top k density hypothesis. 
		encoding[i][j] = | residual[i][j] 			if the jth hypothesis is in top-k
						 | 0 						otherwise
		In each row of the encoding matrix, there are exactly k non zero entries
	"""

	num_correspondences = data.shape[0]

	# Uniform sampling
	weights = [1 for i in range(num_correspondences)]

	# Generate num_correspondences many hypothesis fitted on random subsets of mss correspondences 
	hyp_gen_in_t = time.time()
	hypothesis = GenerateRandomHypothesis(data, weights, structure, num_hypothesis)
	hyp_gen_out_t = time.time()

	if(verbose):
		print("{:40} : {time}".format(
				"Time taken to generate random hypothesis",
				time = hyp_gen_out_t - hyp_gen_in_t
			))

	# generate residual vector for all the correspondences - residual[i][j] denotes the residual between ith point and jth hypothesis
	residual = Residual(data, hypothesis, structure)

	# Sort the residuals based on magnitude
	residual, preferences = torch.sort(residual, dim = 1)

	# Step 4: Extract top k preferred hypothesis
	preferences = preferences[:, :topk]
	residual = residual[:, :topk]
	residual[:, :] = 1		

	# Step 5: Create encoding - set 1 if the jth hypothesis is within top k of ith sample
	encoding = torch.zeros([num_correspondences, num_hypothesis]).double().cuda()
	encoding = encoding.scatter(1, preferences, residual)

	return encoding