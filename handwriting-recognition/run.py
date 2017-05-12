import numpy
import scipy.special
from neuralNetwork import neuralNetwork # from neuralNetwork.py, import neuralNetwork class

n = neuralNetwork(3, 3, 3, 0.3)
n.query([1.0, 0.5, -1.5])
