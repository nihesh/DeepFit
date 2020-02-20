# Author : Nihesh Anderson
# Date 	 : 19 Feb, 2020
# File 	 : loss.py

from src.constants import EPS
import torch
import torch.nn as nn
import torch.nn.functional as F

def dist_align_bce(true, pred):

	"""
	Computes KLD between true and predicted distribution
		true		: [batch_size x num_correspondences] 2D tensor 
		pred 		: [batch_size x num_correspondences] 2D tensor 
	"""

	batch_size = true.shape[0]

	bce = nn.BCELoss()
	bce_loss = bce(pred, true)

	return bce_loss

if(__name__ == "__main__"):

	pass