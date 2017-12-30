import numpy as np

def regularize_L1_norm(compute_loss = True, model = {}, lmbd=0.0001):
	if ( compute_loss ):
		# contribution to cost function
		cost = 0.
		for key in model:
			if ( "b" not in key ):
				# The bias should not be included
				cost += np.sum(np.abs(model[key]))
		cost *= lmbd
		return cost

	else:

		gradient_contribution = {}

		for key in model:
			if ( "b" not in key ):
				# The bias should not be included
				tmp = model[key].copy()
				tmp[tmp > 0] = 1
				tmp[tmp < 0] = -1
				gradient_contribution[key] = lmbd*tmp

		return gradient_contribution

def regularize_L2_norm(compute_loss = True, model = {}, lmbd=0.0001):
	print("HEJ")
	if ( compute_loss ):
		# contribution to cost function
		cost = 0.
		for key in model:
			if ( "b" not in key ):
				# The bias should not be included
				cost += np.sum(np.abs(model[key])**2)
		cost *= lmbd
		return cost

	else:

		gradient_contribution = {}

		for key in gradient_contribution:
			if ( "b" not in key ):
				# The bias should not be included
				gradient_contribution[key] = ( lmbd / 2. ) * model[key]

		return gradient_contribution
