import numpy
import scipy.special
from printNeuralNetwork import printNeuralNetwork

input_nodes = 784 # 28x28=784 pixels
hidden_nodes = 100 # somewhat arbitrary, but reasonable value
output_nodes = 10 # 0-9
learning_rate = 0.3

n = printNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_datasets/mnist_train_1.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
	all_values = record.split(',')
	inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	targets = numpy.zeros(output_nodes) + 0.01
	targets[int(all_values[0])] = 0.99

	print "true value is " + all_values[0]
	print "training net with input:"
	print inputs
	print ""
	print "and target:"
	print targets
	print ""
	print ""

	n.train(inputs, targets)
	pass

test_data_file = open("mnist_datasets/mnist_test_1.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

print ""
print "training finished, starting tests..."
print ""
print ""

for record in test_data_list:
	all_values = record.split(',')
	print("true value is " + all_values[0])
	queryValue = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
	print("querying the net with:")
	print(queryValue)
	print ""
	queryResult = n.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)
	print("query result is:")
	print(queryResult)
	print("")
	print("")
	pass
