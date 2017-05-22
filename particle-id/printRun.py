import numpy
import scipy.special
from printNeuralNetwork import printNeuralNetwork

input_nodes = 5 # p, beta, v1, v2, v3
hidden_nodes = 15 # somewhat arbitrary, 100 is reasonable but changed to 15 for debugging
output_nodes = 3 # A, B, C
learning_rate = 0.3 # somewhat arbitrary, but reasonable

nn = printNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("data/data_train_1.txt", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for record in training_data_list:
	all_values = record.split(',')
	inputs = numpy.asfarray(all_values[1:]) # data is already scaled
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

	nn.train(inputs, targets)
	pass

test_data_file = open("data/data_test_1.txt", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

print ""
print "training finished, starting tests..."
print ""
print ""

for record in test_data_list:
	all_values = record.split(',')
	print("true value is " + all_values[0])
	queryValue = numpy.asfarray(all_values[1:]) # data is already scaled
	print("querying the net with:")
	print(queryValue)
	print ""
	queryResult = nn.query(queryValue)
	print("query result is:")
	print(queryResult)
	print("")
	print("")
	pass
