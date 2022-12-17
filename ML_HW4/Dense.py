import numpy as np
import math

class Dense():
    def __init__(self, n_x, n_y, seed=1):
        self.n_x = n_x
        self.n_y = n_y
        self.seed = seed
        self.initialize_parameters()
        self.name="Dense"

    def initialize_parameters(self):
        """
        Argument:
        self.n_x -- size of the input layer
        self.n_y -- size of the output layer
        self.parameters -- python dictionary containing your parameters:
                           W -- weight matrix of shape (n_y, n_x)
                           b -- bias vector of shape (n_y, 1)
        """
        np.random.seed(self.seed)

        # GRADED FUNCTION: linear_initialize_parameters
        ### START CODE HERE ### (≈ 6 lines of code)
        limit = math.sqrt(6 / (self.n_x + self.n_y))
        W = np.random.rand(self.n_y,self.n_x)*(2*limit)+(-limit)
        b = np.zeros((self.n_y,1))
        ### END CODE HERE ###

        assert(W.shape == (self.n_y, self.n_x))
        assert(b.shape == (self.n_y, 1))

        self.parameters = {"W": W, "b": b}

    def forward(self, A):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        self.cache -- a python tuple containing "A", "W" and "b" ; stored for computing the backward pass efficiently

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter 
        """

        # GRADED FUNCTION: linear_forward
        ### START CODE HERE ### (≈ 2 line of code)
        Z = np.dot(self.parameters['W'],A)+self.parameters['b']
        self.cache=(A, self.parameters['W'], self.parameters['b'])
        ### END CODE HERE ###
        
        assert(Z.shape == (self.parameters["W"].shape[0], A.shape[1]))
        
        return Z

    def backward(self, dZ):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        self.cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
        self.dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        self.db -- Gradient of the cost with respect to b (current layer l), same shape as b

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev

        """
        A_prev, W, b = self.cache
        m = A_prev.shape[1]

        # GRADED FUNCTION: linear_backward
        ### START CODE HERE ### (≈ 3 lines of code)
        self.dW = 1 / m * (np.dot(dZ,self.cache[0].T))
        self.db = 1 / m * (np.sum(dZ,axis = 1,keepdims = True))
        dA_prev = np.dot(self.cache[1].T,dZ)
        ### END CODE HERE ###
        
        assert (dA_prev.shape == A_prev.shape)
        assert (self.dW.shape == self.parameters["W"].shape)
        assert (self.db.shape == self.parameters["b"].shape)
        
        return dA_prev

    def update(self, learning_rate):
        """
        Update parameters using gradient descent
        
        Arguments:
        learning rate -- step size
        """

        # GRADED FUNCTION: linear_update_parameters
        ### START CODE HERE ### (≈ 2 lines of code)
        self.parameters["W"] = self.parameters["W"] - learning_rate * self.dW
        self.parameters["b"] = self.parameters["b"] - learning_rate * self.db
        ### END CODE HERE ###