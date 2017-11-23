# Perceptron
A multilayered perceptron artificial neural network in python!
It can to regression and binary classification today.
I might extend it further in the future to make
multiple label classification also!

# train
One can define the number of layers in the 'hidden_layers' vector.
Also in the last hidden-output layer one can determine what 
activation and what cost function to use.
For regression problems that is: 'linear' activation and 'square_sum' cost.
For binary classification problems that is: 'sigmoid' and 'cross_entropy'. 

# Note
Setting the correct value of parameters such as learning rate is depending
on the problem and the data it is training on. Parameters should be tuned 
according to the way it performs. For example if the loss function is rising, 
a too large  learning rate is typically in use. Lower it in that case. 
