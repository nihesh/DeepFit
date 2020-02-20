"""
Source : https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
This is a slightly modified version of pointnet which only contains the feature space embedding part of it.
We discard the segmentation module from the network
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import src.utils as utils

class STNkd(nn.Module):

	def __init__(self, k = 64, scale = 1):
	
		super(STNkd, self).__init__()
		self.conv1 = torch.nn.Conv1d(k, int(64 / scale), 1)
		self.conv2 = torch.nn.Conv1d(int(64 / scale), int(128 / scale), 1)
		self.conv3 = torch.nn.Conv1d(int(128 / scale), int(1024 / scale), 1)
		self.fc1 = nn.Linear(int(1024 / scale), int(512 / scale))
		self.fc2 = nn.Linear(int(512 / scale), int(256 / scale))
		self.fc3 = nn.Linear(int(256 / scale), k*k)
		self.relu = nn.ReLU()

		self.bn1 = nn.BatchNorm1d(int(64 / scale))
		self.bn2 = nn.BatchNorm1d(int(128 / scale))
		self.bn3 = nn.BatchNorm1d(int(1024 / scale))
		self.bn4 = nn.BatchNorm1d(int(512 / scale))
		self.bn5 = nn.BatchNorm1d(int(256 / scale))

		self.k = k
		self.scale = scale

	def forward(self, x):

		batchsize = x.size()[0]
		
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, int(1024 / self.scale))

		x = F.relu(self.bn4(self.fc1(x)))
		x = F.relu(self.bn5(self.fc2(x)))
		x = self.fc3(x)

		iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
		if x.is_cuda:
			iden = iden.cuda()
		x = x + iden

		x = x.view(-1, self.k, self.k)

		utils.nan_check(x)
		return x

class PointNetFeatureGenerator(nn.Module):

	def __init__(self, dim, global_feat = False, feature_transform = False, scale = 1):
	
		super(PointNetFeatureGenerator, self).__init__()
		
		self.stn = STNkd(dim, scale = scale)
		self.conv1 = torch.nn.Conv1d(dim, int(64 / scale), 1)
		self.conv2 = torch.nn.Conv1d(int(64 / scale), int(64 / scale), 1)
		self.conv3 = torch.nn.Conv1d(int(64 / scale), int(64 / scale), 1)
		self.bn1 = nn.BatchNorm1d(int(64 / scale))
		self.bn2 = nn.BatchNorm1d(int(64 / scale))
		self.bn3 = nn.BatchNorm1d(int(64 / scale))
		self.global_feat = global_feat
		self.feature_transform = feature_transform
		self.relu = nn.LeakyReLU(True)
		if self.feature_transform:
			self.fstn = STNkd(k=64)

		self.scale = scale

	def forward(self, x):

		n_pts = x.size()[2]
		trans = self.stn(x)
		x = x.transpose(2, 1)
		x = torch.bmm(x, trans)
		x = x.transpose(2, 1)
		x = F.relu(self.bn1(self.conv1(x)))

		if self.feature_transform:
			trans_feat = self.fstn(x)
			x = x.transpose(2,1)
			x = torch.bmm(x, trans_feat)
			x = x.transpose(2,1)
		else:
			trans_feat = None

		pointfeat = x
		x = F.relu(self.bn2(self.conv2(x)))
		x = self.bn3(self.conv3(x))
		x = torch.max(x, 2, keepdim=True)[0]
		x = x.view(-1, int(64 / self.scale))

		# HACK
		# return pointfeat

		if self.global_feat:
			return x
		else:
			x = x.view(-1, int(64 / self.scale), 1).repeat(1, 1, n_pts)
			return torch.cat([x, pointfeat], 1)





