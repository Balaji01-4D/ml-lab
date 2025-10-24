import numpy as np


x = np.array([[2, 9], [4, 5], [3, 6]], dtype=float)
y = np.array([[92], [86], [89]], dtype=float)

x = x / np.amax(x, axis=0)
y = y / 100


class NeuralNetwork(object):

    def __init__(self):
        self.input_size = 2
        self.hidden_size = 3
        self.output_size = 1
        
        self.w1 = np.random.randn(self.input_size, self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size)

    def forward(self, x):
        self.z = np.dot(x, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        return self.sigmoid(self.z3)
    
    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))
    
    def sigmoid_prime(self, s):
        return s * (1- s)

    def backward(self, x, y, o):

        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoid_prime(o)

        self.z2_error = self.o_delta.dot(self.w2.T)
        self.z2_delta = self.z2_error * self.sigmoid_prime(self.z2)

        self.w1 += x.T.dot(self.z2_delta)
        self.w2 += x.T.dot(self.o_delta)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)

NN = NeuralNetwork()

print("actual x", x)
print("actual ouput", y)
print("predicted ouput", NN.forward(x))
print("Loss: ", np.mean(np.square(y - NN.forward(x))))
NN.train(x, y)
