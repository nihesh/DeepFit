# Author : Nihesh Anderson 
# Date 	 : 15 Feb, 2020
# File 	 : geometry.py

import cv2
import numpy as np 
import torch
import src.utils as utils
import scipy.spatial.distance as distance

def FitHyperplane(mss):

	"""
	Takes MSS as input and fits a hyperplane over it
	Input:
		mss 		 - [batch_size x mss x dim] 2D tensor
	Output:
		hyperplane	 - [batch_size x dim + 1] 1D tensor
	"""

	batch_size = mss.shape[0]
	mss_size = mss.shape[1]

	x = mss
	y = torch.ones([batch_size, mss_size, 1]).double().cuda()

	# Find least squares solution - (X'X)^-1 X'Y 
	weights = torch.bmm(torch.bmm(torch.inverse(torch.bmm(x.permute(0, 2, 1), x)), x.permute(0, 2, 1)), y).view(batch_size, -1)
	# Hyperplane Form: w'X - 1 = 0
	line = torch.cat([weights, torch.tensor([-1]).double().cuda().view(1, 1).repeat(batch_size, 1)], dim = 1)

	utils.nan_check(line)

	return line

def HyperplaneResidual(data, hypothesis):

	"""
	Returns the residual tensor residual[i][j] denoting the residual between ith sample and jth hypothesis
	Input:
		data 		- [num_correspondences x dim] data tensor
		hypothesis	- [num_hypothesis x (dim + 1)] homography tensor
	Output:
		residuals 	- [num_correspondences x num_hypothesis] residual tensor
	"""

	num_correspondences = data.shape[0]
	degree = data.shape[1]

	ones = torch.ones([num_correspondences, 1]).double().cuda()
	# [num_correspondences x (dim + 1)] 2D data vector
	data = torch.cat([data, ones], dim = 1)
	# [(dim + 1) x num_hypothesis] 2D hypothesis vector
	hypothesis = hypothesis.permute(1, 0)
	# Compute hypothesis norm foor first dim weights - standard way to compute perpendicular distance from point to hyperplane
	hyp_norm = torch.norm(hypothesis[:degree, :], 2, dim = 0).view(1, -1)

	distance = torch.mm(data, hypothesis)
	residual = distance / hyp_norm

	utils.nan_check(residual)

	return residual

def FitHomography(mss):

	"""
	Takes MSS as input and fits a homography over it
	Implementation idea: https://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
	[Implemented in pytorch]
	Input:
		mss 		 - [batch_size x mss x 4] 2D tensor
	Output:
		homography   - [batch_size x 3 x 3] 2D tensor
	"""

	batch_size = mss.shape[0]
	num_correspondences = mss.shape[1]

	mss_src, T_src = utils.normalise2dpts(mss[:, :, :2])
	mss_target, T_target = utils.normalise2dpts(mss[:, :, 2:])

	# Define variables x, y, x' and y' as per the stack exchange formulation
	x = mss_src[:, :, 0].unsqueeze(2)
	y = mss_src[:, :, 1].unsqueeze(2)
	x_prime = mss_target[:, :, 0].unsqueeze(2)
	y_prime = mss_target[:, :, 1].unsqueeze(2)
	zeros = torch.zeros([batch_size, num_correspondences]).double().cuda().unsqueeze(2)
	ones = torch.ones([batch_size, num_correspondences]).double().cuda().unsqueeze(2)

	# Create odd rows of matrix P - say upper half (We reorder for convenience - the solution remains unchanged)
	P_upper = torch.cat([
			-x, -y, -ones, zeros, zeros, zeros, x * x_prime, y * x_prime, x_prime
		], dim = 2)
	P_lower = torch.cat([
			zeros, zeros, zeros, -x, -y, -ones, x * y_prime, y * y_prime, y_prime
		], dim = 2)
	constant = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1]).double().cuda().view(1, 1, 9).repeat(batch_size, 1, 1)
	P = torch.cat([P_upper, P_lower, constant], dim = 1)

	# Create Y column vector which is just the same as transpose of constant tensor defined earlier
	Y = torch.zeros([batch_size, P.shape[1], 1]).double().cuda()
	Y[:, -1, 0] = 1

	# Solve the linear system using DLT least squares -> PH = Y
	homography = torch.bmm(torch.bmm(torch.inverse(torch.bmm(P.permute(0, 2, 1), P)), P.permute(0, 2, 1)), Y).view(batch_size, 9)

	# reshape homography
	homography = homography.view(batch_size, 3, 3)
	# Convert back the homography to negate normalisation effect
	homography = torch.bmm(torch.inverse(T_target), homography)
	homography = torch.bmm(homography, T_src)
	# Ensure homography[2][2] = 1
	homography = homography / homography[:, -1, -1].view(batch_size, 1, 1)

	utils.nan_check(homography)

	return homography

def HomographyResidual(data, hypothesis):

	"""
	Returns the residual matrix residual[i][j] denoting the residual between ith sample and jth hypothesis - symmetric transfer error
	[Implemented using tensor ops]
	Input:
		data 		- [num_correspondences x 4] data matrix
		hypothesis	- [num_hypothesis x 3 x 3] homography matrix
	Output:
		residuals 	- [num_correspondences x num_hypothesis] residual matrix
	"""

	num_correspondences = data.shape[0]
	num_hypothesis = hypothesis.shape[0]

	# First two features in axis = 1 corresponds to source point and the remaining 2 corresponds to target point
	data_src = data[:, :2]
	data_target = data[:, 2:]

	# Convert to homogenous form
	ones = torch.ones([num_correspondences, 1]).double().cuda()
	data_src = torch.cat([data_src, ones], dim = 1)
	data_target = torch.cat([data_target, ones], dim = 1)

	# Prepare data_src and and data_target to compute homography transformation
	data_src = data_src.unsqueeze(1).repeat(1, num_hypothesis, 1)
	data_target = data_target.unsqueeze(1).repeat(1, num_hypothesis, 1)

	# Transform points using the proposed hypothesis set. data_forward[i][j] is the transformation of point i under hypothesis j
	data_forward = torch.bmm(hypothesis, data_src.permute(1, 2, 0)).permute(2, 0, 1)
	data_backward = torch.bmm(torch.inverse(hypothesis), data_target.permute(1, 2, 0)).permute(2, 0, 1)

	# Normalise the transformed points
	data_forward = data_forward / data_forward[:, :, -1].unsqueeze(2)
	data_backward = data_backward / data_backward[:, :, -1].unsqueeze(2)

	# Compute residual - L2 distance between forward transformed source and target
	residual1 = data_forward - data_target
	residual1 = torch.pow(residual1, 2).sum(dim = 2)

	# Compute residual - L2 distance between backward transformed target and source
	residual2 = data_backward - data_src
	residual2 = torch.pow(residual2, 2).sum(dim = 2)
	residual = residual1 + residual2

	utils.nan_check(residual)

	return residual

def FitFundamentalMatrix(mss):

	"""
	Takes MSS as input and fits a homography over it
	[Implemented in pytorch]
	Input:
		mss 		 - [batch_size x mss x 4] 2D tensor
	Output:
		fundamental  - [batch_size x 3 x 3] 2D tensor
	"""

	batch_size = mss.shape[0]
	num_correspondences = mss.shape[1]	

	mss_src, T_src = utils.normalise2dpts(mss[:, :, :2])
	mss_target, T_target = utils.normalise2dpts(mss[:, :, 2:])

	ones = torch.tensor([1]).double().cuda().view(1, 1, 1).repeat(batch_size, num_correspondences, 1)
	mss_src = torch.cat([mss_src, ones], dim = 2)
	mss_target = torch.cat([mss_target, ones], dim = 2)

	row = torch.cat([
		mss_src * mss_target[:, :, 0].view(batch_size, num_correspondences, 1), 
		mss_src * mss_target[:, :, 1].view(batch_size, num_correspondences, -1),
		mss_src 
	], dim = 2)


	outer_product = row.unsqueeze(3) * row.unsqueeze(2)
	mat = outer_product.sum(dim = 1)


	U = []
	S = []
	VT = []

	for i in range(batch_size):

		_, V = torch.symeig(mat[i])
		u, s, v = torch.svd(V[:, 0].view(3, 3))
		vt = v.permute(1, 0)
		s[2] = 0

		U.append(u.unsqueeze(0))
		S.append(torch.diag(s).unsqueeze(0))
		VT.append(vt.unsqueeze(0))

	U = torch.cat(U, dim = 0)
	S = torch.cat(S, dim = 0)
	VT = torch.cat(VT, dim = 0)

	F = torch.bmm(torch.bmm(U, S), VT)
	F = torch.bmm(T_src, torch.bmm(F.permute(0, 2, 1), T_target.permute(0, 2, 1)))
	F = F / F[:, 2, 2].view(batch_size, 1, 1)

	utils.nan_check(F)

	return F

def FundamentalMatrixResidual(data, hypothesis):

	"""
	Returns the residual matrix residual[i][j] denoting the residual between ith sample and jth hypothesis
	[Implemented using tensor ops]
	Input:
		data 		- [num_correspondences x 4] data matrix
		hypothesis	- [num_hypothesis x 3 x 3] fundamental matrix
	Output:
		residuals 	- [num_correspondences x num_hypothesis] residual matrix
	"""

	num_correspondences = data.shape[0]
	num_hypothesis = hypothesis.shape[0]

	ones = torch.zeros([num_correspondences, 1]).double().cuda()
	pts1 = torch.cat([data[:, :2], ones], dim = 1)
	pts2 = torch.cat([data[:, 2:], ones], dim = 1)

	epi_lines1 = torch.bmm(pts1.unsqueeze(0).expand(num_hypothesis, -1, -1), hypothesis)
	epi_lines1 = epi_lines1 / torch.norm(epi_lines1[:, :, :2], dim = 2).unsqueeze(2)
	d1 = torch.abs(torch.sum(epi_lines1 * pts2.unsqueeze(0), dim = 2))

	epi_lines2 = torch.bmm(pts2.unsqueeze(0).expand(num_hypothesis, -1, -1), hypothesis.permute(0, 2, 1))
	epi_lines2 = epi_lines2 / torch.norm(epi_lines2[:, :, :2], dim = 2).unsqueeze(2)
	d2 = torch.abs(torch.sum(epi_lines2 * pts1.unsqueeze(0), dim = 2))	

	residual = torch.max(d1, d2)
	residual = residual.permute(1, 0)

	utils.nan_check(residual)

	return residual

if(__name__ == "__main__"):

	pass 