import numpy as np
import matplotlib.pyplot as plt

def fully_connected(x, W, b, forward = True):
	if ( forward ):
		cache = (x, W)
		y = np.dot(W, x) + b
		return y
	else:
		# If forwad is False, then 'x' should repersent the 
		# the error back propagation in the output nodes
		# and b should be the output from the fully connected layer
		dldy = x
		y = b
		dA = np.dot(dldy, y.T)
		db = dldy.copy()
		dldx = np.dot(W.T, dldy)
		return (dA, db, dldx)

def sigmoid(x, dldx=1., forward = True):
	if ( forward ):
		return 1.0 / (1.0 + np.exp(-x))
	else:
		return (1.0 - x) * x * dldx

def tanh(x,dldx=1., forward = True):
	if ( forward ):
		return np.tanh(x)
	else:
		return (1.0 - x**2) * dldx

def square_sum(x,d,forward=True, SigmoidActivation=False):
	if ( forward ):
		# forward propagation (calculating loss)
		N = len(x)
		y = (1.0 / 2.0) * np.sum( (d-x) ** 2 )
		return y
	else:
		# backward propagation (calculating gradients)
		dldy = -(d-x)
		return dldy

def linear(x, dldx=1., Forward=True):
	if ( Forward ):
		return x
	else: 
		return dldx

def exp_running_avg(running, new, gamma=.9):
	return gamma * running + (1. - gamma) * new

def binary_crossentropy(y,d, Forward=True, SigmoidActivation=False):
	if ( Forward ):
		if ( d == 1. ):
			return -1 * np.sum(np.log(y))
		else:
			return -1 * np.log(1-y)
	else:  

		if ( SigmoidActivation ):
			return (y-d)

		d = y * (y - 1)
		if ( np.abs(d) < 1e-20 ):
			d = np.sign(d) * 1e-20

		dldy = (y-d) / d

		return dldy