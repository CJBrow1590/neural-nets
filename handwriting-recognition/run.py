import numpy
import scipy.special
from neuralNetwork import neuralNetwork # from neuralNetwork.py, import neuralNetwork class
import matplotlib.pyplot

# test network
n = neuralNetwork(3, 3, 3, 0.3)
n.query([1.0, 0.5, -1.5])

data_file = open("mnist_datasets/mnist_train_100.csv", 'r') # r = read only
data_list = data_file.readlines() # only good for small files (hogs memory)
data_file.close()
print data_list[0]
print data_list[1]

all_values = data_list[0].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='nearest')
#matplotlib.pyplot.show() # uncomment these 2 lines to open graphic in pop out window

print "done"
