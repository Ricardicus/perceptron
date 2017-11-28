import numpy as np
import matplotlib.pyplot as plt

from plotter import *
from layers import *

# 	This program is a doodle program that
#	trains a  multiple layerd perceptron
#	for regression and binary classification problems.
#	In the main function I successfully train the perceptron
#	to approximate a sinosiodal function and also do 
# 	a test of binary clasisfication that work ok!
# 	Author: Ricardicus

TRAIN_GRADIENT_DESCENT=1
TRAIN_ADAM_GRADIENT_DESCENT=2

def adam_update(model, alpha, M, R, grad, t, beta1=.9, beta2=.999):

	for k in grad:

		try:
			M[k]
			R[k]
		except KeyError:
			M[k] = 0.
			R[k] = 0.

		M[k] = exp_running_avg(M[k], grad[k], beta1)
		R[k] = exp_running_avg(R[k], grad[k]**2, beta2)

		m_k_hat = M[k] / (1. - beta1**(t))
		r_k_hat = R[k] / (1. - beta2**(t))

		model[k] -= alpha * m_k_hat / (np.sqrt(r_k_hat) + 1e-7)

def gradient_decent(model, alpha, grad):

	for k in grad:
		model[k] -= alpha * grad[k]

def train(X, Y, hidden_output_layer_activation, loss_function, hidden_layers,  iterations=1000, learning_rate=0.001, optimizer=TRAIN_GRADIENT_DESCENT):
	# This is a very simple implementation of a len(hidden_layers) layered perceptron, doing this to test
	# how the multi-layered perceptron can be used for regression/classification problems

	# this net operates with multiple input => mutiple output
	RI = len(X[0])
	RO = len(Y[0])

	N = len(X)

	batch = 10

	initial_learning_rage = learning_rate
	learning_rate_decrease = 10000

	model = {}

	i = 0
	R = RI
	# Setting up the layers
	while ( i < len(hidden_layers)):
		model["A"+str(i)] = np.random.randn(hidden_layers[i], R) / np.sqrt(hidden_layers[i])
		model["b"+str(i)] = np.random.randn(hidden_layers[i], 1) / np.sqrt(hidden_layers[i])
		R = hidden_layers[i]
		i += 1

	# last hidden to output layer
	model["A"+str(i)] = np.random.randn(RO, R) / np.sqrt(RO)
	model["b"+str(i)] = np.random.randn(RO, 1) / np.sqrt(RO)

	# Init tables used for the Adam gradient update optimization scheme
	M = {}
	R = {}
	for key in model:
		M[key] = np.zeros_like(model[key])
		R[key] = np.zeros_like(model[key])

	caches = []
	n = 0
	loss_low_pass = 0.01
	loss = 0
	while ( n < iterations ):

		i = 0; 
		while ( i < N):

			grads = {}
			for key in model:
				grads[key] = np.zeros_like(model[key])

			q = 0;
			while ( q < batch and i < N ):
				
				# Input to the first fully connected layer
				inpt = X[i]
				
				# This will contain the signals: input + output of each layer (to be used in backprop)
				ys = []
				y = inpt
				ys.append(y)
				p = 0
				# Forward progatation one step through all the layers
				while ( p <= len(hidden_layers) ):

					y = fully_connected(y, model["A"+str(p)], model["b"+str(p)])
					
					if ( p != len(hidden_layers) ):
						y = tanh(y)
					else:
						y = hidden_output_layer_activation(y)

					# Append the 
					ys.append(y)

					p += 1

				l = loss_function(y, Y[i])

				if n == 0:
					loss = np.sum(l)
				else:
					loss = loss * (1.0-loss_low_pass) + loss_low_pass*np.sum(l)

				# single backward step
				yb = loss_function(y, Y[i], False, hidden_output_layer_activation)

				p = len(hidden_layers) 
				# Backward propagating the error back through 
				# all the layers to get the gradients 
				while ( p >= 0 ):

					if ( p == len(hidden_layers) and not ( (hidden_output_layer_activation == sigmoid and loss_function == binary_crossentropy) or (hidden_output_layer_activation == softmax and loss_function == crossentropy ))):
						yb = hidden_output_layer_activation(yb, yb, False) # Skipping this part for binary or multiple classes classification
					else:
						yb = tanh(ys[p+1], yb, False)

					(dA, db, yb) = fully_connected(yb, model["A"+str(p)], ys[p], False)

					grads["A"+str(p)] += dA
					grads["b"+str(p)] += db

					p -= 1

				i+=1
				q+=1

			for key in grads:
				grads[key] /= (q+1)

			# gradient decent
			if ( optimizer == TRAIN_GRADIENT_DESCENT):
				gradient_decent(model, learning_rate, grads)
			elif (optimizer == TRAIN_ADAM_GRADIENT_DESCENT):
				adam_update(model, learning_rate, M, R, grads, n+1)

		if ( optimizer == TRAIN_GRADIENT_DESCENT ):
			learning_rate = initial_learning_rage / (1.0 + n/ learning_rate_decrease)

		if ( n % 10 == 0 ):
			print("Iteration: " + str(n) + ", loss: " + str(loss) + ", lr: " + str(learning_rate))
		
		n+=1


	print("Loss on training set after " + str(iterations) + " iterations: " + str(loss) + ", hidden neurons: " + str(hidden_layers))

	return model

def output(X, model, number_of_layers, hidden_output_layer_activation):

	Y = []

	for i in range(len(X)):
		inpt = X[i]
		y = inpt
		p = 0
		# Forward progatation one step through all the layers
		while ( p <= number_of_layers ):

			y = fully_connected(y, model["A"+str(p)], model["b"+str(p)])
			
			if ( p != number_of_layers ):
				y = tanh(y)
			else:
				y = hidden_output_layer_activation(y)

			p += 1

		Y.append(y)

	return Y
