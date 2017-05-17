import numpy
import scipy.special
import matplotlib.pyplot

data_file = open("mnist_datasets/mnist_train_100.csv", 'r') # r = read only
data_list = data_file.readlines() # only good for small files (hogs memory)
data_file.close()

rowNumber = 2

print ""
print data_list[rowNumber]

all_values = data_list[rowNumber].split(',')
image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='nearest')
matplotlib.pyplot.show() # uncomment these 2 lines to open graphic in pop out window
