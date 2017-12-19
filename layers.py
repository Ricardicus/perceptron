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

def relu(x, dldx=1., forward = True):
	if ( forward ):
		return np.maximum(x, 0, x)
	else:
		return 1. * (x > 1) * dldx

def tanh(x,dldx=1., forward = True):
	if ( forward ):
		return np.tanh(x)
	else:
		return (1.0 - x**2) * dldx

def square_sum(x,d,forward=True, activation_function=None):
	if ( forward ):
		# forward propagation (calculating loss)
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

def softmax(x, dldx=1., Forward=True):
	if ( Forward ):
		y = np.exp(x)
		s = np.sum(y)
		y /= s
		return y
	else: 
		return dldx # only works for 'crossentropy' cost function, and it is taken care of there already.. why use softmax otherwise, right?

def exp_running_avg(running, new, gamma=.9):
	return gamma * running + (1. - gamma) * new

def binary_crossentropy(y,d, Forward=True, activation_function=None):
	if ( Forward ):
		if ( d == 1. ):
			return -1 * np.sum(np.log(y))
		else:
			return -1 * np.log(1-y)
	else:  

		if ( activation_function == sigmoid ):
			return (y-d)

		d = y * (y - 1)
		if ( np.abs(d) < 1e-20 ):
			d = np.sign(d) * 1e-20

		dldy = (y-d) / d

		return dldy


def crossentropy(y,d, Forward=True, activation_function=None):
	if ( Forward ):
		return - np.sum(np.log(y) * d)
	else:  

		dldy = (y-d)

		if ( activation_function == softmax ):	
			return dldy	
		else :

			d = y * (y - 1)
			try:
				dldy /= d
			except:
				print("Multiple classes classification using crossentropy does not work for activation function: " + str(activation_function))
				raise

		return dldy

