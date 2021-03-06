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
            self.W[idx + 1] = np.random.randn(out_nodes, in_nodes) * np.sqrt(2 / in_nodes)
            self.B[idx + 1] = np.zeros(shape=(out_nodes, 1))
            self.activation.append(act_fn)

    def forward(self, X):
        self.A[0] = X
        L = len(self.parameters)
        for idx in range(1, L + 1):
            self.Z[idx] = self.W[idx] @ self.A[idx - 1] + self.B[idx]

            if (self.activation[idx] == "relu"):
                self.A[idx] = utils.relu(self.Z[idx])
            elif (self.activation[idx] == "sigmoid"):
                self.A[idx] = utils.sigmoid(self.Z[idx])
            elif (self.activation[idx] == "softmax"):
                self.A[idx] = utils.softmax(self.Z[idx])

    def cost_function(self, Y):
        m = Y.shape[1]
        L = len(self.parameters)
        cost = - (1 / m) * np.sum(Y * np.log(self.A[L]) + (1 - Y) * np.log(1 - self.A[L]))
        return cost

    def backward(self, Y):
        m = Y.shape[1]
        L = len(self.parameters)
        self.dA[L] = - (np.divide(Y, self.A[L]) - np.divide(1 - Y, 1 - self.A[L]))
        for idx in reversed(range(1, L + 1)):

            if (self.activation[idx] == "relu"):
                self.dZ[idx] = self.dA[idx] * utils.drelu(self.Z[idx])
            elif (self.activation[idx] == "sigmoid"):
                self.dZ[idx] = self.dA[idx] * utils.dsigmoid(self.Z[idx])
            elif (self.activation[idx] == "softmax"):
                self.dZ[idx] = self.dA[idx] * utils.dsoftmax(self.Z[idx])

            self.dW[idx] = (1 / m) * (self.dZ[idx] @ self.A[idx - 1].T)
            self.dB[idx] = (1 / m) * np.sum(self.dZ[idx], axis=1, keepdims=True)
            self.dA[idx - 1] = (self.W[idx].T @ self.dZ[idx])

    def optimizer(self, learning_rate):
        L = len(self.parameters)
        for idx in range(1, L + 1):
            self.W[idx] = self.W[idx] - learning_rate * self.dW[idx]
            self.B[idx] = self.B[idx] - learning_rate * self.dB[idx]

    def predict(self, X):
        L = len(self.parameters)
        self.forward(X)
        return (self.A[L] > 0.5)

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
