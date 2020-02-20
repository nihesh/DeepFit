# Author : Nihesh Anderson 
# Date 	 : 15 Feb, 2020
# File	 : model.py

import torch
import torch.nn as nn
from src.constants import EPS
import torch.nn.functional as F

class Discrete_Sampler(nn.Module):

	"""
	Samples from a discrete distribution using Gumbel-softmax trick
	Implementation of the following paper - https://www.ijcai.org/Proceedings/2019/0544.pdf
	"""

	def __init__(self, temperature = 5):

		super(Discrete_Sampler, self).__init__()

		# the scale parameter for computing softmax - higher value implies better approximation of max, but it's unstable
		self.temperature = temperature

	def forward(self, data, distribution, k):

		"""
		Given data and distribution over dim 0, this sampler returns k soft samples sampled based on distribution
		Input:
			data 		 : Tensor [N x d1 x d2 x d3....] 
			distribution : Tensor [N] - May or may not sum to one
		Output:
			sampled data : Tensr  [k x d1 x d2 x d3....]
		"""

		mask = torch.zeros(data.shape[0]).double().cuda()

		# Normaliasation
		distribution = distribution / distribution.sum()
		distribution = torch.log(torch.max(distribution / torch.max(1 - distribution, torch.tensor([EPS]).double().cuda()), torch.tensor([EPS]).double().cuda()))

		sampled_pts = []

		for i in range(k):

			# sampling from gumbel distribution
			gumbel_distribution = torch.distributions.gumbel.Gumbel(torch.zeros_like(distribution), torch.ones_like(distribution))
			gumbel_sample = gumbel_distribution.sample()

			# Add gumbel noise to logit
			gumbel_keys = gumbel_sample + distribution


			# Decrease the keys of sampled points
			khot_mask = torch.max(1.0 - mask, torch.tensor([EPS]).double().cuda())
			gumbel_keys = gumbel_keys + 3 * torch.log(khot_mask)

			print(khot_mask)

			# Find soft max gumbel key
			onehot_approx = F.softmax(gumbel_keys * self.temperature, dim = 0)

			# Computing soft argmax - Ideally, we want a one hot encoded vector, 1 corresponding to argmax. We use softmax to approximate this vector. 
			soft_argmax = torch.mm(onehot_approx.view(1,-1), data)
			sampled_pts.append(soft_argmax)			

			# Update mask so that the sampled point is not sampled again
			mask = mask + onehot_approx

		# Concatenate all the sampled points and return
		sampled_pts = torch.cat(sampled_pts, dim = 0)
		return sampled_pts

if(__name__ == "__main__"):

	# driver code for discrete sampler module

	model = Discrete_Sampler(5).cuda()

	data = torch.tensor([[1, 1], [2, 2]]).double().cuda()
	w = torch.tensor([0.1, 0.9]).double().cuda()

	print(model(data, w, 2))

	pass