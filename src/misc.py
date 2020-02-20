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