import numpy as np
import matplotlib.pyplot as plt

from plotter import *
from layers import *
from perceptron import *

if __name__=="__main__":

	# Here are some examples of what the "Perceptron.py" module can do
	
	# Regression test set 1: A sinosoidal of amplitude 'A'
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
	
	# Binary classificaton test 1: A classificaton problem [x1,x2] -> y ≤ {0,1} where 0 -> blue, 1 -> red
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

	m = train(X, Y, sigmoid, binary_crossentropy, [100, 100, 100, 10, 10], 30000, 0.00001, TRAIN_ADAM_GRADIENT_DESCENT)

	# The grid, for which the decision boundary can be shown
	X_decision = []
	for xx in np.arange(-4, 4, 0.01):
		for yy in np.arange(-4, 4, 0.01):
			X_decision.append(np.array([[xx , yy]]).T)

	Y_decision = output(X_decision, m, 5, sigmoid)

	# Looks great!
	plot_scatter_and_line(X, Y, X_decision, Y_decision,0.02)	
	
	
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
	

	# Multilabel classification test 1: (x0,x1,x2) -> [0,1,2,3] , three variables maps to 4 classes
	X = []
	Y = []

	mux0 = 1. # Lets call this 'class 0'
	muy0 = 1.
	muz0 = 0.
	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux0, np.random.randn(1)[0] + muy0, np.random.randn(1)[0] + muz0]]).T)
		Y.append(np.array([[1.,0.,0.,0.]]).T)
	
	mux1 = 1 # Lets call this 'class 1'
	muy1 = -1.
	muz1 = 1.
	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux1, np.random.randn(1)[0] + muy1, np.random.randn(1)[0] + muz1]]).T)
		Y.append(np.array([[0.,1.,0.,0.]]).T)

	mux2 = 1 # Lets call this 'class 2'
	muy2 = -1.
	muz2 = 2.
	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux2, np.random.randn(1)[0] + muy2, np.random.randn(1)[0] + muz2]]).T)
		Y.append(np.array([[0.,0.,1.,0.]]).T)

	mux2 = 1 # Lets call this 'class 3'
	muy2 = -1.
	muz2 = 2.
	for i in range(50):
		X.append(np.array([[np.random.randn(1)[0] + mux2, np.random.randn(1)[0] + muy2, np.random.randn(1)[0] + muz2]]).T)
		Y.append(np.array([[0.,0.,0.,1.]]).T)
	
	train(X, Y, softmax, crossentropy, [10, 10, 10], 10000, 0.001, TRAIN_ADAM_GRADIENT_DESCENT) # There is no way to plot this higher dimensional classification result.. just observe traning error

