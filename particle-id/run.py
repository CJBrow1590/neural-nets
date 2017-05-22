import numpy
import scipy.special
from neuralNetwork import neuralNetwork

input_nodes = 5
hidden_nodes = 100
output_nodes = 3
learning_rate = 0.3

nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("data/data_train_100k.txt", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
	all_values = record.split(',')
	inputs = numpy.asfarray(all_values[1:]) # data is already scaled
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99
	nn.train(inputs, targets)

test_data_file = open("data/data_test_100k.txt", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

for record in test_data_list:
	all_values = record.split(',')
	queryVal = nn.query(numpy.asfarray(all_values[1:]))
	print all_values[0], all_values[1][0:7], all_values[2][0:7], all_values[3][0:7], all_values[4][0:7], all_values[5][0:7], str(queryVal[0][0])[0:7], str(queryVal[1][0])[0:7], str(queryVal[2][0])[0:7]
