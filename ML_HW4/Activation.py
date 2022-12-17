import numpy as np

class Activation():
    def __init__(self, name):
        self.name = name
        

    def forward(self, Z):
        if self.name == "sigmoid":
            """
            Implements the sigmoid activation in numpy
            
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of sigmoid(z), same shape as Z
            
            """

            # GRADED FUNCTION: sigmoid_forward
            ### START CODE HERE ### (≈ 8 lines of code)
            
            A = 1/(1+np.exp(-Z))
            self.cache = Z
            ### END CODE HERE ###
            
            return A

        elif self.name == "softmax":
            """
            Implements the softmax activation in numpy
            
            Arguments:
            Z -- numpy array of any shape (dim 0: number of classes, dim 1: number of samples)
            self.cache -- stores Z as well, useful during backpropagation
            
            Returns:
            A -- output of softmax(z), same shape as Z
            """

            # GRADED FUNCTION: softmax_forward
            ### START CODE HERE ### (≈ 2 lines of code)
            A = np.exp(Z)/np.sum(np.exp(Z),axis=0)
            self.cache = Z
            ### END CODE HERE ###
            
            return A

        elif self.name == "relu":
            """
            Implement the RELU function in numpy
            Arguments:
            Z -- numpy array of any shape
            self.cache -- stores Z as well, useful during backpropagation
            Returns:
            A -- output of relu(z), same shape as Z
            
            """
            
            # GRADED FUNCTION: relu_forward
            ### START CODE HERE ### (≈ 2 lines of code)
            A = np.maximum(0,Z)
            self.cache = Z
            ### END CODE HERE ###
            
            assert(A.shape == Z.shape)
            
            return A

    def backward(self, dA=None, Y=None):
        if self.name == "sigmoid":
            """
            Implement the backward propagation for a single SIGMOID unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: sigmoid_backward
            ### START CODE HERE ### (≈ 9 lines of code)
            Z = self.cache
            s = 1/(1+np.exp(-Z))
            dZ = dA * s * (1-s)
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.name == "relu":
            """
            Implement the backward propagation for a single RELU unit.
            Arguments:
            dA -- post-activation gradient, of any shape
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: relu_backward
            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache
            dZ = np.array(dA, copy=True) # just converting dz to a correct object. 
            dZ[Z <= 0] = 0 # When z <= 0, you should set dz to 0 as well.
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ

        elif self.name == "softmax":
            """
            Implement the backward propagation for a [SOFTMAX->CCE LOSS] unit.
            Arguments:
            Y -- true "label" vector (one hot vector, for example: [[1], [0], [0]] represents rock, [[0], [1], [0]] represents paper, [[0], [0], [1]] represents scissors 
                                      in a Rock-Paper-Scissors image classification), shape (number of classes, number of examples)
            self.cache -- 'Z' where we store for computing backward propagation efficiently
            Returns:
            dZ -- Gradient of the cost with respect to Z
            """
            
            # GRADED FUNCTION: softmax_CCE_backward 
            ### START CODE HERE ### (≈ 3 lines of code)
            Z = self.cache #三小
            s = np.exp(Z)/np.sum(np.exp(Z),axis=0)
            
            dZ = s - Y
            ### END CODE HERE ###
            
            assert (dZ.shape == Z.shape)
            
            return dZ