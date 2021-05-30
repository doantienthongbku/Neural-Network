import numpy as np
import utils
import matplotlib.pyplot as plt

# input_nodes, output_nodes, activation
# parameters = [[10 * 10, 10, 'relu'],
#               [10, 20, 'relu'],
#               [20, 2, 'softmax']]


class NeuralNetwork:
    def __init__(self, parameters):
        self.parameters = parameters
        self.W = {}
        self.B = {}
        self.Z = {}
        self.A = {}
        self.dW = {}
        self.dB = {}
        self.dZ = {}
        self.dA = {}
        self.activation = ['non']
        self.cost_list = []

    def _initialize_weights(self):
        for idx, (in_nodes, out_nodes, act_fn) in enumerate(self.parameters):
            self.W[f'W{idx + 1}'] = np.random.randn(out_nodes, in_nodes) * np.sqrt(2 / in_nodes)
            self.B[f'B{idx + 1}'] = np.zeros(shape=(out_nodes, 1))
            self.activation.append(act_fn)

    def forward(self, X):
        self.A['A0'] = X
        L = len(self.parameters)
        for idx in range(1, L + 1):
            self.Z[f'Z{idx}'] = self.W[f'W{idx}'] @ self.A[f'A{idx - 1}'] + self.B[f'B{idx}']

            if (self.activation[idx] == "relu"):
                self.A[f'A{idx}'] = utils.relu(self.Z[f'Z{idx}'])
            elif (self.activation[idx] == "sigmoid"):
                self.A[f'A{idx}'] = utils.sigmoid(self.Z[f'Z{idx}'])
            elif (self.activation[idx] == "softmax"):
                self.A[f'A{idx}'] = utils.softmax(self.Z[f'Z{idx}'])

    def cost_function(self, Y):
        m = Y.shape[1]
        L = len(self.parameters)
        cost = - 1 * np.sum(np.dot(Y, np.log(self.A[f'A{L}'])), np.dot((1 - Y), np.log(1 - self.A[f'A{L}']))) / m
        # cost = - (1 / m) * np.sum(Y * np.log(self.A[f'A{L}']) + (1 - Y) * np.log(1 - self.A[f'A{L}']))
        return cost

    def backward(self, Y):
        m = Y.shape[1]
        L = len(self.parameters)
        self.dA[f'dA{L}'] = - (np.divide(Y, self.A[f'A{L}']) - np.divide(1 - Y, 1 - self.A[f'A{L}']))
        for idx in reversed(range(1, L + 1)):

            if (self.activation[idx] == "relu"):
                self.dZ[f'dZ{idx}'] = self.dA[f'dA{idx}'] * utils.drelu(self.Z[f'Z{idx}'])
            elif (self.activation[idx] == "sigmoid"):
                self.dZ[f'dZ{idx}'] = self.dA[f'dA{idx}'] * utils.dsigmoid(self.Z[f'Z{idx}'])
            elif (self.activation[idx] == "softmax"):
                self.dZ[f'dZ{idx}'] = self.dA[f'dA{idx}'] * utils.dsoftmax(self.Z[f'Z{idx}'])

<<<<<<< HEAD
            self.dW[f'dW{idx}'] = (1 / m) * (self.dZ[f'dZ{idx}'] @ self.A[f'A{idx - 1}'].T)
=======
            self.dW[f'dW{idx}'] = np.dot(self.dZ[f'dZ{idx}'], self.A[f'A{idx - 1}'].T) / m
>>>>>>> d6331482366a9ca80fd5ab0182640347afdb697c
            self.dB[f'dB{idx}'] = (1 / m) * np.sum(self.dZ[f'dZ{idx}'], axis=1, keepdims=True)
            self.dA[f'dA{idx - 1}'] = (self.W[f'W{idx}'].T @ self.dZ[f'dZ{idx}'])
            # self.dW[f'dW{idx}'] = np.dot(self.dZ[f'dZ{idx}'], self.A[f'A{idx - 1}'].T)
            # self.dB[f'dB{idx}'] = (1 / m) * np.sum(self.dZ[f'dZ{idx}'], axis=1, keepdims=True)
            # self.dA[f'dA{idx - 1}'] = np.dot(self.dW[f'dW{idx}'].T, self.dZ[f'dZ{idx}'])

    def optimizer(self, learning_rate):
        L = len(self.parameters)
        for idx in range(1, L + 1):
            self.W[f'W{idx}'] = self.W[f'W{idx}'] - learning_rate * self.dW[f'dW{idx}']
            self.B[f'B{idx}'] = self.B[f'B{idx}'] - learning_rate * self.dB[f'dB{idx}']

    def predict(self, X):
        L = len(self.parameters)
        self.forward(X)
        return (self.A[f'A{L}'] > 0.5)

    def dataloader(self, X, Y, batch_size=10, shuffle=True):
        pass

    def fit(self, X, Y, epochs, learning_rate, print_cost=False):
        self._initialize_weights()
        for i in range(epochs):
            self.forward(X)
            cost = self.cost_function(Y)
            self.backward(Y)
            self.optimizer(learning_rate=learning_rate)

            if print_cost and i % 100 == 0 or i == epochs - 1:
                print("Cost after iteration {}: {}".format(i, cost))
            if i % 100 == 0 or i == epochs - 1:
                self.cost_list.append(cost)

    def plot_costs(self, learning_rate):
        plt.plot(self.cost_list)
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
