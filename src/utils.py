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

def compute_silhouette_order(data, label):

	"""
	Computes the order in which clusters are to be detected by the network
	We try to ensure that easy clusters are detected first, followed by hard clusters.
	Clusters with high mean silhouette scores are defined as easy clusters. 
	Input:
		data - [batch_size x num_correspondences x dim] 3D tensoor
		label - [batch_size x num_correspondences] 2D tensor
	Output:
		true_distribution, true_state - list of size max number of clusters where ith item in the list is a 2D tensor
										of shape [num_sets_in_batch_with_i_structures x num_correspondences]
		num_structures 				  - [batch_size] 1D numpy array denoting the number of structures in the ith batch
	""" 

	batch_size = data.shape[0]
	num_correspondences = data.shape[1]

	# Compute pairwise distance between all the points
	distance = data.unsqueeze(2) - data.unsqueeze(1)
	distance = torch.pow(torch.pow(distance, 2).sum(dim = 3), 0.5)

	distance = distance.cpu().detach().numpy()
	label = label.cpu().detach().numpy()
	
	num_structures = []
	distribution = []
	state = []
	max_labels = 0

	for i in range(batch_size):
		
		sh = sklearn.metrics.silhouette_samples(distance[i], label[i])
		
		# Compute mean silhouette score for all the labels except label 0 - outlier class
		cur_labels = list(set(label[i]))
		num_labels = len(cur_labels) - 1
		# Add the number of structures in the current batch to the collection
		num_structures.append(num_labels)
		# Compute max structures oover all batches
		max_labels = max(max_labels, num_labels)

		mean_score = []
		for l in cur_labels:

			# Skip if outlier
			if(l == 0):
				continue

			# Compute mean silhouette score of all samples belonging to structure l
			mean_score.append([np.mean(sh[label[i] == l]), l])

		# Sort structure silhouette scores from high to low
		mean_score.sort(reverse = True)

		# Create the probability matrix where row[j][k] = 1 iff mean_score[j][1] == label[k]
		# state vector is just the cumulative sum of probability vectors
		_dis = torch.zeros([num_labels, num_correspondences]).float().cuda()
		_state = torch.zeros([num_labels, num_correspondences]).float().cuda()
		cur_state = torch.zeros([num_correspondences]).float().cuda()
		for j in range(num_labels):
			_state[j] = cur_state
			_dis[j, np.argwhere(label[i] == mean_score[j][1]).reshape(-1)] = 1
			cur_state += _dis[j]
		distribution.append(_dis)
		state.append(_state)

	# Convert num_structures collection to numpy array
	num_structures = np.asarray(num_structures)

	# Reorder distribution and state to compute loss in a vectorized fashion
	true_distribution = []
	true_state = []

	for i in range(max_labels):
		structure_i = []
		state_i = []
		for j in range(batch_size):
			if(distribution[j].shape[0] <= i):
				continue
			structure_i.append(distribution[j][i].unsqueeze(0))
			state_i.append(state[j][i].unsqueeze(0))
		structure_i = torch.cat(structure_i, dim = 0)
		state_i = torch.cat(state_i, dim = 0)

		true_distribution.append(structure_i)
		true_state.append(state_i)

	return true_distribution, true_state, num_structures

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

if(__name__ == "__main__"):

	pass