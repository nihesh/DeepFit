# Author : Nihesh Anderson
# File 	 : compressor.py
# Date 	 : 19 Feb, 2020

import torch
import torch.nn as nn
from src.constants import EPS
import torch.nn.functional as F
import models.pointnet as pointnet

class Compressor(nn.Module):

	"""
	A fully connected neural network to convert each feature vector to a probability distribution from which points can be sampled
	We use sigmoid activation to convert the unbounded values to probabilities
	"""

	def __init__(self, feature_dim):

		"""
		Input:
			feature_dim 		- Size of the feature space generated by pointnet
		"""

		super(Compressor, self).__init__()

		# Number of nodes in the layer between input and final representation
		frac = int(max(5, 0.1 * feature_dim)) 

		self.fc1 = nn.Linear(feature_dim, frac)
		self.fc2 = nn.Linear(frac, 1)

		self.relu = nn.LeakyReLU(True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, embedding):

		"""
		Forward propagation step for autoencoder
		Input:
			embedding 		- [batch_size x num_correspondences x dim] 3D tensor representing scaled pointnet embedding 
		Output:
			distribution 	- [batch_size x num_correspondences] 2D tensor which is an unnormalised distribution over the points
		"""

		# Encoder
		embedding = self.relu(self.fc1(embedding))
		embedding = self.sigmoid(self.fc2(embedding))

		return embedding