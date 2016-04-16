"""NeuralNetwork.py."""

from classifier import Classifier
from attributes import NominalAttribute
import random
import math
import sys


def f(x):
    """Sigmoid activation function."""
    return (1 / (1 + math.exp(-x)))


def df(x):
    """Sigmoid derivative."""
    return ((1.0 - x) * x)


class NeuralNetwork(Classifier):
    """Neural Network object."""

    model = "Backpropagation Neural Network"

    def __init__(self, trainset, n=0.01, j=5, max_error=.3, debug=False):
        """Init backpropagation neural network.

        Args:
            trainset (Dataset): training dataset
            j (int): the hidden dimension (will have -1 as last value of y_j)
            n (float): learning rate
        """
        self.trainset = trainset
        self.J = j  # hidden dimension
        self.n = n  # learning rate
        self.max_error = max_error
        self.debug = debug
        random.seed()

    def get_z_and_d(self):
        """Create z (list of examples), d (class list)."""
        self.z = []
        self.d = []
        examples = self.trainset.examples
        for example in examples:
            di = example.values[examples.attributes.classindex]
            if not isinstance(di, list):
                di = [di]
            self.d.append(di)
            vals = example.values[:examples.attributes.classindex]  # add bias
            vals.append(-1.0)
            self.z.append(vals)
        self.I = self.trainset.attributes.size  # input dimension
        self.K = self.trainset.attributes.class_size  # output dimension
        self.o = [0.0] * self.K

    def init_weight_matrices(self):
        """Initialize the weight matrices to random weights."""
        self.W = [[random.uniform(-.1, .1) for i in xrange(self.J)] for k in xrange(self.K)]  # output x hidden
        self.V = [[random.uniform(-.1, .1) for i in xrange(self.I)] for j in xrange(self.J)]  # hidden x input

    def _train(self, print_results=True):
        """Train the model."""
        # Initialize variables dependent on the trainset
        self.P = self.trainset.examples.size
        self.y = [0.0] * self.J
        self.get_z_and_d()
        self.init_weight_matrices()
        training_cycles = 0
        q = 0
        error = 1.0  # creates a do while loop
        while (self.max_error < error):
            error = 0.0
            random_range = range(self.P)
            random.shuffle(random_range)
            for i in random_range:
                self.step2(self.z[i])
                error += self.step3(i)
                self.bpp(i)
                q += 1
            training_cycles += 1
            if q % 1000 == 0 and self.debug:
                sys.stdout.write("Error: %-3f\r" % (error))
                sys.stdout.flush()
        if print_results:
            print("final error: %s" % error)
            print("presentations: %s" % q)
            print("training cycles: %s" % training_cycles)
            self.print_weights()

    def step2(self, z):
        """Training step starts here. Input is presented and the layers outputs are computed."""
        # Create y vector
        for j in xrange(self.J):
            current_sum = 0.0
            for i in range(self.I):
                current_sum += z[i] * self.V[j][i]
            self.y[j] = f(current_sum)
        # Create o vector
        for k in xrange(self.K):
            current_sum = 0.0
            for j in range(self.J):
                current_sum += self.y[j] * self.W[k][j]
            self.o[k] = f(current_sum)
        return self.o

    def bpp(self, inputindex):
        """Back propagate and get error."""
        self.step4(inputindex)
        self.step5()
        self.step6(inputindex)

    def step3(self, inputindex):
        """Error value is computed (E = (1/2)(dk - ok)^2 + E for 1, 2... K)."""
        error = 0.0
        for i, k in enumerate(self.d[inputindex]):
            error += 0.5 * (k - self.o[i])**2
        return error

    def step4(self, inputindex):
        """Calculate error signal vectors for the hidden layers and output layers."""
        self.error_signal_o = [0.0] * self.K
        self.error_signal_y = [0.0] * self.J

        for k in xrange(self.K):
            self.error_signal_o[k] = df(self.o[k]) * (self.d[inputindex][k] - self.o[k])

        for j in xrange(self.J):
            error_sum = 0.0
            for k in xrange(self.K):
                error_sum += self.error_signal_o[k] * self.W[k][j]
            self.error_signal_y[j] = df(self.y[j]) * error_sum

    def step5(self):
        """Back propagate first with output weight adjustment."""
        for j in xrange(self.J):
            for k in xrange(self.K):
                self.W[k][j] = self.W[k][j] + self.n * self.error_signal_o[k] * self.y[j]

    def step6(self, inputindex):
        """Adjust weights of hidden layers."""
        for i in xrange(self.I):
            for j in xrange(self.J):
                self.V[j][i] = self.V[j][i] + self.n * self.error_signal_y[j] * self.z[inputindex][i]

    def _classify(self, example):
        """Classify an example."""
        # grab example + propagate it
        # through the network forward pass
        # get _>o, get numbers in output layer
        # return the index of the component of _>o that is max
        vals = example.values[:self.trainset.attributes.classindex]
        vals.append(-1)
        classification = self.step2(vals)[0]
        if isinstance(self.trainset.attributes[-1], NominalAttribute):
            classification = int(round(classification))
        return classification

    def print_weights(self):
        """Print weights."""
        print('W:')
        for i in xrange(self.K):
            print(self.W[i])
        print('V:')
        for j in xrange(self.J):
            print(self.V[j])
