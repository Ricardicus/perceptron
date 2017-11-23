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
				yb = loss_function(y, Y[i], False, hidden_output_layer_activation == sigmoid)

				p = len(hidden_layers) 
				# Backward propagating the error back through 
				# all the layers to get the gradients 
				while ( p >= 0 ):

					if ( p == len(hidden_layers) and not ( hidden_output_layer_activation == sigmoid and loss_function == cross_entropy) ):
						yb = hidden_output_layer_activation(yb, yb, False)
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

if __name__=="__main__":

	
	# Test set 1: A sinosoidal of amplitude 'A'
	X = []
	Y = []
	
	A = 10
	for i in range(100):
		X.append(np.array([[i*np.pi/20.0]]))
		Y.append(np.array([[A * np.sin(i*np.pi/20.0)]]))

	model = train(X, Y, linear, square_sum, [10, 10, 10], 40000, 0.001, TRAIN_ADAM_GRADIENT_DESCENT)

	Y_aprx = output(X, model, 3, linear)

	# Looks great!
	plot_this(X, Y, Y_aprx)

	
	# A classificaton problem [x1,x2] -> y ≤ {0,1} where 0 -> blue, 1 -> red
	X = []
	Y = []
	
	A = 10

	mux1 = 1.
	muy1 = 1.
	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux1, np.random.randn(1)[0] + muy1 ]]).T)
		Y.append(np.array([[1]]))
	
	mux2 = 1
	muy2 = -1.

	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux2, np.random.randn(1)[0] + muy2 ]]).T)
		Y.append(np.array([[0.]]))

	m = train(X, Y, sigmoid, cross_entropy, [10, 10, 10], 5000, 0.0001, TRAIN_ADAM_GRADIENT_DESCENT)

	# The grid, for which the decision boundary can be shown
	X_decision = []
	for xx in np.arange(-4, 4, 0.05):
		for yy in np.arange(-4, 4, 0.05):
			X_decision.append(np.array([[xx , yy]]).T)

	Y_decision = output(X_show, m, 3, sigmoid)

	# Looks great!
	plot_scatter_and_line(X, Y, X_decision, Y_decision,0.04)	
	
	"""
	# Regression test set 2: a polynomial function
	X = []
	Y = []
	for i in range(100):
		x = i/100.0 * 4 - 2
		y = 20*(x - 0.5)**2 / (1.0 + x**2)
		X.append(np.array([[x]]))
		Y.append(np.array([[y]]))

	model = train(X, Y, linear, square_sum, [10, 10, 10], 5000)

	Y_aprx = output(X, model, 3, linear)

	# Looks great also!
	plot_this(X, Y, Y_aprx)
	
	# Test set 3: A sinosoidal of amplitude 'A' + a gaussian noise , controlled by 'e'
	X = []
	Y = []

	A = 10
	e = 4.0
	for i in range(10):
		X.append(np.array([[i*np.pi/20.0]]))
		Y.append(np.array([[A * np.sin(i*np.pi/20.0) + e * np.random.randn(1,1)[0][0]]]))

	m = train(X, Y, linear, square_sum, [10, 10, 10], 10000, 0.00001)

	Y_aprx = output(X, model, 3, linear)

	# Looks good!
	plot_this(X, Y, Y_aprx)
	"""	
