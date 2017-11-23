# Perceptron
A multi layered perceptron in python!
It can to regression and binary classification.
I might extend it further in the future to make
multiple label classification also!

# train
One can define the number of layers in the 'hidden_layers' vector.
Also in the last hidden-output layer one can determine what 
activation and what cost function to use.
For regression problems that is: 'linear' activation and 'square_sum' cost.
For binary classification problems that is: 'sigmoid' and 'cross_entropy'. 

# Note
Parameters such as learning rate depends a lot on the data it is training
on and should be tuned according to the way it performs. If the loss function
is rising, a too large learning rate is typically in use. Lower it in that case. 
