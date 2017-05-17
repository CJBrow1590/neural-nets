import numpy
import scipy.special
from neuralNetwork import neuralNetwork # from neuralNetwork.py, import neuralNetwork class
import matplotlib.pyplot

###   #### test stuff ####
###   n = neuralNetwork(3, 3, 3, 0.3)
###   n.query([1.0, 0.5, -1.5])
###   
###   data_file = open("mnist_datasets/mnist_train_100.csv", 'r') # r = read only
###   data_list = data_file.readlines() # only good for small files (hogs memory)
###   data_file.close()
###   print data_list[0]
###   print data_list[1]
###   
###   all_values = data_list[0].split(',')
###   image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
###   #matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='nearest')
###   #matplotlib.pyplot.show() # uncomment these 2 lines to open graphic in pop out window
###   
###   # rescale from 0-255 to 0.01-1.0 (zeros are bad for neural nets)
###   scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
###   
###   # example with 10 output nodes (0-9)
###   onodes = 10
###   targets = numpy.zeros(onodes) + 0.01 # array of length 10 filled w/ all 0.01s
###   targets[int(all_values[0])] = 0.99 # the 0th element of the training data gives the true value:
###   # e.g. if all_values[0] = 4, then:
###   #   0    1    2    3    4    5    6    7    8    9
###   # 0.01 0.01 0.01 0.01 0.99 0.01 0.01 0.01 0.01 0.01

input_nodes = 784 # 28x28=784 pixels
hidden_nodes = 100 # somewhat arbitrary, but reasonable value
output_nodes = 10 # 0-9
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_datasets/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
	all_values = record.split(',')
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99
	n.train(inputs, targets)
	pass

test_data_file = open("mnist_datasets/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
	all_values = record.split(',')
	print("true value = " + all_values[0])
	queryVal = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
	iCount = 0
	for iOut in queryVal:
		print(str(iCount) + " " + str(iOut))
		iCount += 1
		pass
	print("")
	pass
