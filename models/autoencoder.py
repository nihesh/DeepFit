# Author : Nihesh Anderson
# File 	 : autoencoder.py
# Date 	 : 19 Feb, 2020

import torch
import torch.nn as nn
from src.constants import EPS
import torch.nn.functional as F
import models.pointnet as pointnet

class AutoEncoder(nn.Module):

	"""
	A fully connected dense auto encoder to identify the set of proposed hypothesis from the given state vector
	"""

	def __init__(self, input_dim, latent_dim):

		"""
		Input:
			input_dim 		- Size of the input space
			latent_dim 		- Size of the latent space
		"""

		super(AutoEncoder, self).__init__()
		assert(input_dim >= latent_dim)

		# Number of nodes in the layer between input and latent representation
		frac = int(0.1 * (input_dim - latent_dim) + latent_dim) 

		self.enc1 = nn.Linear(input_dim, frac)
		self.enc2 = nn.Linear(frac, latent_dim)

		self.dec1 = nn.Linear(latent_dim, frac)
		self.dec2 = nn.Linear(frac, input_dim)

		self.relu = nn.LeakyReLU(True)
		self.sigmoid = nn.Sigmoid()

	def forward(self, state):

		"""
		Forward propagation step for autoencoder
		Input:
			state 	- state[i] is proportional to the number of times sample i has been a part of inlier set - 0 in the first iteration
					  [batch_size x num_correspondences] 2D tensor 
		Output:
			[batch_size x num_correspondences] 2D tensor - A transformation of state, where state[i] is expected to be close to 1 if i has not been considered as an inlier, 0 otherwise
		"""

		# Encoder
		state = self.relu(self.enc1(state))
		state = self.relu(self.enc2(state))

		# Decodoer
		state = self.relu(self.dec1(state))
		state = self.relu(self.dec2(state))

		return state
