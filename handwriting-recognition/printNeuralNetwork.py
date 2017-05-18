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
		print "wih has type", type(self.wih)
		print "and shape", self.wih.shape
		print "wih = "
		print self.wih
		print ""
		print "who has type", type(self.who)
		print "and shape", self.who.shape
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

		print "finished single training instance"
		print "first argument (inputs_list):"
		print inputs_list
		print "second argument (targets_list):"
		print inputs_list
		print "inputs:"
		print inputs
		print "targets:"
		print targets
		print "hidden_inputs:"
		print hidden_inputs
		print "hidden_outputs:"
		print hidden_outputs
		print "final_inputs:"
		print final_inputs
		print "final_outputs:"
		print final_outputs
		print "output_errors:"
		print output_errors
		print "hidden_errors:"
		print hidden_errors
		print "who has shape", self.who.shape
		print "who="
		print self.who
		print "wih has shape", self.wih.shape
		print "wih="
		print self.wih
		pass
	
	def query(self, inputs_list):
		# convert inputs list to 2d array
		inputs = numpy.array(inputs_list, ndmin=2).T
	
		hidden_inputs = numpy.dot(self.wih, inputs)
		hidden_outputs = self.activation_function(hidden_inputs)
		final_inputs = numpy.dot(self.who, hidden_outputs)
		final_outputs = self.activation_function(final_inputs)

		print "Network has been queried"
		print "first argument (inputs_list):"
		print inputs_list
		print "inputs:"
		print inputs
		print "hidden_inputs:"
		print hidden_inputs
		print "hidden_outputs:"
		print hidden_outputs
		print "final_inputs:"
		print final_inputs
		print "final_outputs:"
		print final_outputs
	
		return final_outputs
