
import numpy as np
import copy
import math

# DO NOT CHANGE SEED
np.random.seed(42)

# DO NOT CHANGE LAYER CLASS
class Layer(object):

	def set_input_shape(self, shape):
		self.input_shape = shape

	def layer_name(self):
		return self.__class__.__name__

	def parameters(self):
		return 0

	def forward_pass(self, X, training):
		raise NotImplementedError()

	def backward_pass(self, accum_grad):
		raise NotImplementedError()

	def output_shape(self):
		raise NotImplementedError()

# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
	def __init__(self, n_units, input_shape=None):
		self.layer_input = None
		self.input_shape = input_shape
		self.n_units = n_units
		self.trainable = True
		self.W = None
		self.w0 = None

		self.optim_w=None
		self.optim_w0=None

	def initialize(self, optimizer):
		# Initialize weights W, biases w0, and optimizers
		limit=1/np.sqrt(self.input_shape[0])

		self.W=np.random.uniform(-limit,limit,(self.input_shape[0],self.n_units))
		self.W0=np.zeros((1,self.n_units))
		self.optim_w=copy.deepcopy(optimizer)
		self.optim_w0=copy.deepcopy(optimizer)

	def parameters(self):
		# Return total number of parameters
		return self.W.size + self.W0.size

	def forward_pass(self, X, training=True):
		# Compute and return the forward pass
		self.layer_input=X
		return np.dot(X,self.W) + self.W0

	def backward_pass(self, accum_grad):
		# Compute gradients, update weights if trainable, return grad w.r.t. input
		w_old=self.W.copy()

		if self.trainable:
			grad_w=np.dot(self.layer_input.T,accum_grad)
			grad_w0=np.sum(accum_grad,axis=0,keepdims=True)
		
			self.W=self.optim_w.update(self.W,grad_w)
			self.W0=self.optim_w0.update(self.W0,grad_w0)
		
		return np.dot(accum_grad,w_old.T)

	def output_shape(self):
		# Return output shape tuple
		return (self.n_units,)
