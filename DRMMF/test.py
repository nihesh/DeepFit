# Author : Nihesh Anderson
# Date 	 : 18 Feb, 2020
# File 	 : test.py

import utils
import scipy.io
import torch
import numpy as np

def t_residual_to_density(src, target):

	"""
	Input:
		src 		- Path to residual mat file
		target 		- Path to density mat file
	"""

	EPS = 1e-6

	src_mat = scipy.io.loadmat(src)
	target_mat = scipy.io.loadmat(target)

	r = torch.tensor(src_mat["R_hat"]).double().cuda()
	d = torch.tensor(target_mat["D_hat"]).double().cuda()

	d_hat = utils.residual_to_density(r)
	
	# Save output of python implementation to tests folder
	target_mat = {}	
	target_mat["D_hat"] = d_hat.cpu().detach().numpy()
	scipy.io.savemat("./tests/residual_to_density/my_estimate_D_hat.mat", target_mat)

	print("Number of non matching entries", (torch.abs(d_hat - d) > EPS).sum().item())
	print("Total entries", d.shape[0] * d.shape[1])

	# If the number of violations is at most 10% of the total number of entries, we are okay.
	return (torch.abs(d_hat - d) > EPS).sum() <= 0.1 * d.shape[0] * d.shape[1]

if(__name__ == "__main__"):

	# Test for residual_to_density function
	assert t_residual_to_density("./tests/residual_to_density/R_hat.mat", "./tests/residual_to_density/D_hat.mat")