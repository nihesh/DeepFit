# Author : Nihesh Anderson
# Date 	 : 16 Feb, 2020
# File 	 : misc.py

"""
This file contains a set of unused functions that are not being used anywhere in this repository.
These implementations are being maintained so that it can be used in the future if necessary
"""

def VisualiseHomography(data, label):

	"""
	Function to assert if inlier labels are correct
	"""
	
	data = data[label == 1]
	label = label[label == 1]

	# Plot the given data	
	plt.clf()
	plt.scatter(data[:, 0], data[:, 1], color = "red")
	plt.scatter(data[:, 2], data[:, 3], color = "green")

	# Now, try to estimate the random homography generated earlier - this works!
	homo = geometry.FitHomography(torch.tensor(data).double().cuda())
	homo = homo.cpu().detach().numpy()

	transformed_forward = cv2.perspectiveTransform(np.asarray([data[:, :2]]), homo)
	transformed_forward = transformed_forward.reshape(-1, 2)

	plt.scatter(transformed_forward[:, 0], transformed_forward[:, 1], color = "blue")

	residual = 2 * np.power(transformed_forward - data[:, 2:], 2).sum(axis = 1)
	np.ndarray.sort(residual)

	print(residual)

	plt.show()
	plt.close("all")

	return

def DegenerateHomography(mss):

	"""
	Returns true if there does not exist a unique homography that fits the given mss
	Input:
		mss 		- Samples on which the homography has to be fit
	"""

	# Approach: No three key points in the left or the right image should be collinear
	left = mss[:, :2]
	right = mss[:, 2:]

	# True if mss is not degenerate
	ok = True

	# Check collinearity in left image
	for i in range(4):
		all_but_one = []
		for j in range(4):
			if(i == j):
				continue
			all_but_one.append(left[j].unsqueeze(0))
		
		# all but one consists of all the points except point i
		all_but_one = torch.cat(all_but_one, dim = 0)
		all_but_one = all_but_one - all_but_one[0]

		# matrix rank of vectors with respect to one of the points should be 2
		ok = ok and (torch.matrix_rank(all_but_one) == 2)

	# Check collinearity in right image
	for i in range(4):
		all_but_one = []
		for j in range(4):
			if(i == j):
				continue
			all_but_one.append(right[j].unsqueeze(0))
		
		# all but one consists of all the points except point i
		all_but_one = torch.cat(all_but_one, dim = 0)
		all_but_one = all_but_one - all_but_one[0]

		# matrix rank of vectors with respect to one of the points should be 2
		ok = ok and (torch.matrix_rank(all_but_one) == 2)

	return not ok

def DegenerateHyperplane(mss, freedom):

	"""
	Returns true if there does not exist a unique hyperplane that fits the given mss
	Input:
		mss 		- Samples on which the hyperplane has to be fit
		freedom 	- degrees of freedom for the structure - 1 for line and 2 for planes
	"""

	mss = mss - mss[0]
	return torch.matrix_rank(mss) != freedom

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