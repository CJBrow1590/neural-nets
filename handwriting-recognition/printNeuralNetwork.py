import numpy
import scipy

class printNeuralNetwork:

	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
	
		self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
		self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
	
		self.lr = learningrate
	
		self.activation_function = lambda x: scipy.special.expit(x)

		print "network has been initialized"
		print "wih = "
		print self.wih
		print ""
		print "who = "
		print self.who
		print ""
		print ""
		pass
	
	def train(self, inputs_list, targets_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
		targets = numpy.array(targets_list, ndmin=2).T

		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)

		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		output_errors = targets - final_outputs
		hidden_errors = numpy.dot(self.who.T, output_errors)

		# update the weights for the links between the hidden and output layers
		self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

		# update the weights for the links between the input and hidden layers
		self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
		pass
	
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
	
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)
	
		return final_outputs
