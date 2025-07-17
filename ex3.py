import numpy as np

x =  np.array([
    [2,9],
    [1,5],
    [3,6]
], dtype=float)


y = np.array([
    [92],
    [86],
    [89]
], dtype=float)

"""amax function return maximum of array 
if the axis = 0 -> return maximum among the each column
if the axis = 1 -> return maximum among the each row"""

x = x / np.amax(x, axis=0)

""" above line show the each number in column is divided by the maximum number in that column
    example 2 / 3 , 1 / 3 and 3 / 3
"""

y = y / 100

class NeuralNetwork(object):

    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.hidden_size = 3

        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

        """ w1 -> matrix of 2 x 3 of random values
            w2 -> matrix of 3 x 1 of random values 
        """

    def forward(self, x):
        self.z = np.dot(x, self.w1)
        """ np.dot function is dot multiplication of matrices """

        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        o = self.sigmoid(self.z3)

        return o
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    """ np.exp function is a exponential function to np.exp(1) -> e^1 (e power 1)
        np.exp([1,2]) -> [e^1, e^2] 
    """
    
    def sigmoid_prime(self, s):
        return s*(1-s)
    
    def backward(self, x: np.ndarray, y, o):
        self.o_error = y - o
        self.o_delta: np.ndarray = self.o_error * self.sigmoid_prime(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.o_delta * self.sigmoid_prime(self.z2)

        self.w1 += x.T.dot(self.o_delta)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)


NN = NeuralNetwork()
print(f"input = {x}")
print(f"actul output = {y}")

print(f"predicted output = {NN.forward(x)}")

print("loss ", str(np.mean(np.square(y - NN.forward(x)))))
NN.train(x, y)