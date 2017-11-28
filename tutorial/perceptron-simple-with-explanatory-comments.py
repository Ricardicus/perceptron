import numpy as np
from numpy import tanh
import matplotlib.pyplot as plt

# 	This program is a doodle program that
#	trains a 4 layered (3 hidden layers) perceptron
#	for regression of a function.
#	In the main function I successfully train the perceptron
#	on a sinosiodal function.

#	I also have added some comments that explains a bit
# 	what is going on here. 

# A bunch of "layers" , input -> ouput 

# Ignore the details of them if they are "messy"!
# The important stuff is exaplined in the "train"-method!

def fully_connected_forward(x, W, b):
	y = np.dot(W, x) + b
	return y

# This is used to compute the gradients 
# for a fully connected layer.
def fully_connected_backward(dldy, A, y):
	dA = np.dot(dldy, y.T)
	db = dldy.copy()
	dldx = np.dot(A.T, dldy)
	return (dA, db, dldx)

# sigmoid activation function
def sigmoid_forward(x):
	return 1.0 / (1.0 + np.exp(-x))

# activation functions have derivatives 
def sigmoid_backwards(x, dldx):
	return (1.0 - x) * x * dldx

def tanh_forward(x):
	y = tanh(x)
	return y

def tanh_backward(dldx, x):
	return (1.0 - x**2) * dldx

def square_sum_forward(x,d):
	N = len(x)
	y = (1.0 / 2.0) * np.sum( (d-x) ** 2 )
	return y

def square_sum_backward(y, d):
	dldy = -(d-y)
	return dldy

# This is a function that actually trains a MLP with 3 hidden layers
def train(X, Y, H1=20, H2=20, H3=10, iterations=1000, learning_rate=0.0001):
	# This is a very simple implementation of a 3 layered perceptron, doing this to test
	# how the multi-layered perceptron can be used for regression problems

	# The number of so called "neurons" in the first hidden layer is equal to H1, H2 layer 2 etc.. 

	RI = len(X[0])
	RO = len(Y[0])

	N = len(X)

	batch = 10

	initial_learning_rage = learning_rate
	learning_rate_decrease = 10000

	# The model is described by the weight-matrices A1, A2, A3, A3 and the bias vectors B1, B2, B3, B4

	# input-to-hidden layer
	A1 = np.random.randn(H1, RI) / np.sqrt(H1)
	b1 = np.random.randn(H1, 1) / np.sqrt(H1)

	# hidden-to-hidden-layer
	A2 = np.random.randn(H2, H1) / np.sqrt(H2)
	b2 = np.random.randn(H2, 1) / np.sqrt(H2)

	# hidden-to-hidden-layer
	A3 = np.random.randn(H3, H2) / np.sqrt(H3)
	b3 = np.random.randn(H3, 1) / np.sqrt(H3)

	# hidden-to-output-layer
	A4 = np.random.randn(RO, H3) / np.sqrt(RO)
	b4 = np.random.randn(RO, 1) / np.sqrt(RO)

	caches = []
	n = 0
	loss_low_pass = 0.01
	loss = 0
	while ( n < iterations ):

		i = 0; 
		while ( i < N):
			
			# The crux of training a network is to compute the gradients.
			# Once the gradients are computed, one can use gradient descent 
			# to make the model "fit" the data appropriately


			# I start by initializing the gradients -> they are all set to zero 
			# for starters 

			dA4 = np.zeros_like(A4)
			dA3 = np.zeros_like(A3)
			dA2 = np.zeros_like(A2)
			dA1 = np.zeros_like(A1)

			db4 = np.zeros_like(b4)
			db3 = np.zeros_like(b3)
			db2 = np.zeros_like(b2)
			db1 = np.zeros_like(b1)

			q = 0;
			while ( q < batch and i < N ):
				inpt = X[i]

				# I now propagate the input values through the network
				# and store the output values. Some of them are needed
				# when computing the gradients

				y1 = fully_connected_forward(inpt, A1, b1)
				y2 = tanh_forward(y1)
				y3 = fully_connected_forward(y2, A2, b2)
				y4 = tanh_forward(y3)
				y5 = fully_connected_forward(y4, A3, b3)
				y6 = tanh_forward(y5)
				y7 = fully_connected_forward(y6, A4, b4)

				l = square_sum_forward(y7, Y[i])

				if n == 0:
					loss = np.sum(l)
				else:
					loss = loss * (1.0-loss_low_pass) + loss_low_pass*np.sum(l)

				# Now this is the "tricky" part!
				# I have the output of the model and now I shall do something called
				# "backpropagation". In this step all the gradients are computed.
				# By applying the chain-rule in each step, one can compute the gradients
				# for each layer one at a time! 

				# I start at the end of the net, with the loss function

				yb = square_sum_backward(y7, Y[i])

				# Now i backpropagate this value back through the fully connected layer 
				# A fully connected layer is the name for the layer 
				# that computes the output 'y' from the input 'x' by: y = Ax + b

				(dA, db, yb) = fully_connected_backward(yb, A4, y6)

				dA4 += dA
				db4 += db

				# All the outputs pass what is called an activation function 
				# to normalize the values, in this case 'tanh' is used, it outputs
				# values between -1 and 1. Activation functions also have derivatives
				# that can be computed using their output, which makes them very
				# easy to deal with.

				# If y = tanh(x), then dy/dx = (1 - y^2), this value: dy/dx 
				# is what is stored in "yb" after this function is called.

				yb = tanh_backward(yb, y6)

				# The process is reapeated back through all the layers

				(dA, db, yb) = fully_connected_backward(yb, A3, y4)

				dA3 += dA
				db3 += db

				yb = tanh_backward(yb, y4)

				(dA, db, yb) = fully_connected_backward(yb, A2, y2)

				dA2 += dA
				db2 += db

				yb = tanh_backward(yb, y2)

				(dA, db, yb) = fully_connected_backward(yb, A1, inpt)

				dA1 += dA
				db1 += db

				# Now I have computed them!
				# But I will compute an average of these gradients for the batch size: 'batch'. 

				i+=1
				q+=1

			# I compute an "average" of the gradient here
			dA4 /= (q+1)
			dA3 /= (q+1)
			dA2 /= (q+1)
			dA1 /= (q+1)

			db4 /= (q+1)		
			db3 /= (q+1)
			db2 /= (q+1)
			db1 /= (q+1)

			# Gradient decent update, this is basically how the model is "fitting" to the data.
			# This is the simple step where all the "magic" happens. 
			# With a smaller learning_rate the training takes more time, but it is also more likely
			# that the model becomes good (finds a local minima to the error function in the vast "parameter space").

			A1 = A1 - learning_rate * dA1
			b1 = b1 - learning_rate * db1

			A2 = A2 - learning_rate * dA2
			b2 = b2 - learning_rate * db2

			A3 = A3 - learning_rate * dA3
			b3 = b3 - learning_rate * db3

			A4 = A4 - learning_rate * dA4
			b4 = b4 - learning_rate * db4

		# it is common to decrease the learning rate over time, but it is OK to leave it commented out. 
#		learning_rate = initial_learning_rage / (1.0 + n/ learning_rate_decrease)

		if ( n % 10 == 0 ):
			print("Iteration: " + str(n) + ", loss: " + str(loss) + ", lr: " + str(learning_rate))
		
		n+=1


	print("Loss on training set after " + str(iterations) + " iterations: " + str(loss) + ", hidden neurons: [" + str(H1) + ", " + str(H2) + ", " + str(H3) + "]")
	m = {"A1": A1, "b1": b1, "A2": A2, "b2": b2, "A3": A3, "b3": b3, "A4": A4, "b4": b4}

	return m

def output(X, m):

	Y = []

	for i in range(len(X)):
		inpt = X[i]
		y1 = fully_connected_forward(inpt, m["A1"], m["b1"])
		y2 = tanh_forward(y1)
		y3 = fully_connected_forward(y2, m["A2"], m["b2"])
		y4 = tanh_forward(y3)
		y5 = fully_connected_forward(y4, m["A3"], m["b3"])
		y6 = tanh_forward(y5)
		y7 = fully_connected_forward(y6, m["A4"], m["b4"])

		Y.append(y7)

	return Y

def plot_this_solo(X,Y):
	x = []
	y_correct = []
	for i in range(len(X)):
		x.append(X[i][0])
		y_correct.append(Y[i][0])

	plt.plot(x,y_correct,"x")
	plt.show()	

def plot_this(X, Y, Y_aprx):

	x = []
	y_correct = []
	y_approx = []
	for i in range(len(X)):
		x.append(X[i][0])
		y_correct.append(Y[i][0])
		y_approx.append(Y_aprx[i][0])

	plt.plot(x,y_correct,"x",x,y_approx,"-")
	plt.show()

if __name__=="__main__":

	# Test set 1: A sinosoidal of amplitude 'A'

	X = []
	Y = []

	A = 10
	for i in range(100):
		X.append(np.array([[i*np.pi/20.0]]))
		Y.append(np.array([[A * np.sin(i*np.pi/20.0)]]))

	# Note: It might take a while until it finishes! 
	# Grab a cup of coffe and read the papers while it computes. 
	m = train(X, Y, 10, 10, 10, 35000)

	Y_aprx = output(X, m)

	# Looks great! If not, increase the number of iterations and train it again :)
	plot_this(X, Y, Y_aprx)
