# Author : Nihesh Anderson
# Date 	 : 14 Feb, 2020
# File 	 : utils.py

"""
Collection of handy tools that supports main script
"""

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import math
import torch
from src.constants import EPS
import sklearn.metrics

def nan_check(vector):

	"""
	Asserts none of the elements in vector is nan
	"""

	assert(not torch.isnan(vector).any() and not (vector == float("inf")).any())

def shift_and_scale_norm(data):

	"""
	Centres the data and makes sure feature variance is 1.
	Input:
		data 		- 2D tensor
	Output:
		data 		- Rescaled 2D tensor

	Output Explanation:
		data = data - mean / std
	"""

	# Compute mean and std across features
	mean = torch.mean(data, dim = 0).view(1, -1)
	std = torch.max(torch.std(data, dim = 0).view(1, -1), torch.tensor([EPS]).double().cuda())

	# Shift and scale the data
	data = (data - mean) / std

	return data
	
def VisualiseEmbedding(data, embedding, label, save_path):

	"""
	Visualises the 2D TSNE representation of the given embedding, colour coded using the corresponding label
	Input:
	 	data  	  - [num_samples x dim] 2D numpy array representing the input dataa
		embedding - [num_samples x num_hypothesis] 2D numpy array representing the encoding
		label 	  - [num_samples] 1D numpy array denoting the class label of the embedding
		save_path - path to which figures have to be saved
	Output:
		None
	"""

	plt.clf()

	fig, ax = plt.subplots(1, 2)
	fig.set_size_inches(8.5, 5)

	# Reduce the dimension of embedding and data vector to 2 using TSNE
	embedding = TSNE(n_components = 2).fit_transform(embedding)
	embedding = np.asarray(embedding)

	if(data.shape[1] > 2):
		data = TSNE(n_components = 2).fit_transform(data)
	data = np.asarray(data)

	# Scatter the embedding
	ax[0].scatter(embedding[:, 0], embedding[:, 1], c = label)
	ax[1].scatter(data[:, 0], data[:, 1], c = label)

	# Set title
	ax[0].set_title("Embedding space")
	ax[1].set_title("Data space")

	fig.savefig(save_path)

	plt.close("all")

def normalise2dpts(data):

	"""
	Centres the data to origin and ensures that the magnitude of every data vector is sqrt(2)
	Input:
		data 		- [batch_size x num_correspondences x 2] 2D tensor
	Output:
		data  		- [batch_size x num_correspondences x 2] 2D tensor - normalised data vector 
		T 			- [batch_size x 3 x 3] Transformation matrix - scale + shift operations
	"""

	batch_size = data.shape[0]
	num_correspondences = data.shape[1]

	# Compute the centroid of all the points
	mean = torch.mean(data, dim = 1).view(batch_size, 1, 2)
	# Shift centroid to origin
	data = data - mean
	# Compute magnitude of each point
	magnitude = torch.pow(torch.pow(data, 2).sum(dim = 2), 0.5)
	mean_magnitude = torch.mean(magnitude, dim = 1)
	# Rescale the data
	scale = (math.sqrt(2) / mean_magnitude).view(-1, 1, 1)
	data = data * scale

	# Compute transformation matrix
	scale = scale.view(batch_size)
	mean = mean.view(batch_size, 2)
	T = torch.zeros([batch_size, 3, 3]).double().cuda()
	T[:, 2, 2] = 1
	T[:, 0, 0] = T[:, 1, 1] = scale
	T[:, 0, 2] = -scale * mean[:, 0]
	T[:, 1, 2] = -scale * mean[:, 1]

	return data, T

def get_structure_distribution(labels):

	"""
	This module operates on batches of labels and returns a list of 2D tensors, where the ith item in the list and jth tensor 
	denotes the probability distribution of the jth structure and ith batch. output[i][j][k] == 1 iff kth point belongs to the jth
	structure in the ith batch
	Assumes the labelling is continuous from 0....C where C is the number of structures and 0 is for outliers

	Input:
		labels 					 - [batch_size x num_correspondences] 2D tensor denoting the structure membership of all the points in a batch
	Output:
		structure_distribution   - [batch_size x num_structures x num_samples] list of 2D tensors as described above
		num_structures			 - [batch_size] 1D numpy array denoting the number of structures in a batch
	"""

	batch_size = labels.shape[0]
	num_structures_collect = []

	structure_distribution = []
	for i in range(batch_size):

		num_structures = torch.max(labels[i]).item()
		num_correspondences = labels[i].shape[0]
		st_dist_now = torch.zeros([num_structures + 1, num_correspondences]).float().cuda()

		x_idx = labels[i].view(-1)
		y_idx = torch.linspace(0, num_correspondences - 1, num_correspondences).long().cuda().view(-1)

		st_dist_now[x_idx.cpu().detach().numpy(), y_idx.cpu().detach().numpy()] = 1
		st_dist_now = st_dist_now[1:, :]

		structure_distribution.append(st_dist_now)
		num_structures_collect.append(num_structures)

	return structure_distribution, np.asarray(num_structures_collect)

def find_approximate_match(predicted, state, ground_truth):

	"""
	This function matches the predicted distribution with one of the ground truth distributions that doesn't favour choosing picked points, given by state
	Intuitively, it tries to find the best remaining structure greedily based on a reward computed as predicted * ground_truth[i] - state * ground_truth[i] - (1 - ground_truth[i]) * predicted[i]
	Input:
		predicted 				- [batch_size x num_correspondences] 2D tensor describing the probability distribution
		state 					- [batch_size x num_correspondences] 2D tensor describing the state
		ground_truth 			- [batch_size x num_structures x num_correspondences] 
	Output:	
		align_distribution		- [batch_size x num_correspondences] 2D tensor with which predicted distribution should be aligned with
	"""

	batch_size = predicted.shape[0]
	num_correspondences = predicted.shape[1]

	matched_distribution = []

	for i in range(batch_size):

		reward = ground_truth[i] * (predicted[i] - state[i]).view(1, num_correspondences) - (1 - ground_truth[i]) * predicted[i].view(1, num_correspondences)
		reward = reward.sum(dim = 1)
		
		max_reward, best = torch.max(reward, 0)

		to_align_with = ground_truth[i][best].unsqueeze(0)
		matched_distribution.append(to_align_with)

	matched_distribution = torch.cat(matched_distribution, dim = 0)
	state = torch.min(state + matched_distribution, torch.tensor([1]).float().cuda())

	return matched_distribution, state

if(__name__ == "__main__"):

	pass