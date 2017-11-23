import numpy as np
import matplotlib.pyplot as plt

def plot_this_solo(X,Y):
	x = []
	y_correct = []
	for i in range(len(X)):
		x.append(X[i][0])
		y_correct.append(Y[i][0])

	plt.plot(x,y_correct,"x")
	plt.show()	

def plot_this_scatter(X,Y):
	x0 = []
	y0 = []
	x1 = []
	y1 = []
	for i in range(len(X)):
		if ( Y[i][0] == 1. ):
			x1.append(X[i][0][0])
			y1.append(X[i][1][0])
		else:
			x0.append(X[i][0][0])
			y0.append(X[i][1][0])

	plt.scatter(x1,y1, marker="x", c="blue")
	plt.scatter(x0,y0, marker=".", c="black")
	plt.show()	

def plot_scatter_and_line(X,Y,X_a,Y_a, tol=0.03):
	x0 = []
	y0 = []
	x1 = []
	y1 = []

	for i in range(len(X)):
		if ( Y[i][0] == 1. ):
			x1.append(X[i][0][0])
			y1.append(X[i][1][0])
		else:
			x0.append(X[i][0][0])
			y0.append(X[i][1][0])

	x3 = []
	y3 = []

	for i in range(len(X_a)):
		if ( Y_a[i][0] > (0.5-tol) and Y_a[i][0] < (0.5 + tol) ):
			x3.append(X_a[i][0][0])
			y3.append(X_a[i][1][0])

	plt.scatter(x1,y1, marker="x", c="blue")
	plt.scatter(x0,y0, marker=".", c="black")
	plt.scatter(x3,y3, marker="o", c="red")
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
