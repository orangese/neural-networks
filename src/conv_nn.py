
"""

"conv_nn.py"

A program that houses code for a convolutional neural network. It is not
optimized-- this program is purely for learning purposes. Also note that
the structure of this file is much different than "mlp.py" in that the
layers are organized into their own classes.

Progress:

1. 6/18/19: created Layer, Conv, and Pooling classes, along with their
   respective feed-forward functions. As of now, they seem functional,
   but feeding forward through a convolutional network is taking around
   10 times longer than propagating through an MLP.


"""

#Imports
import numpy as np
from scipy.signal import convolve
from functools import reduce

import sys
sys.path.insert(0, "/Users/ryan/Documents/Coding/neural_networks/src")
from mlp import Activation, Cost

#Classes
class Layer(object):
  #not really functional, just provides appropriate abstraction

  def __init__(self, output):
    self.output = output
    self.dim = self.output.shape
    #it's understood that dim refers to the dimensions of the output

class Conv(Layer):
  #basic convolutional layer, stride is fixed at one

  def __init__(self, previous_layer, kernel_dim, num_fmaps,
               activation_func = Activation("sigmoid")):
    #initializes Conv layer
    #note that each layer has an attribute, previous_layer, which
    #is used for propagation
    if isinstance(previous_layer, Dense):
      assert previous_layer.cost_func is None, \
             "conv layer cannot follow a dense output layer"
    self.previous_layer = previous_layer
    self.kernel_dim = kernel_dim
    self.num_fmaps = num_fmaps
    self.activation_func = activation_func

    self.weights = np.random.randn(kernel_dim[0], kernel_dim[0]) \
                   / np.sqrt(self.kernel_dim[0])
    #assumes kernel_dim is 3-D
    self.bias = np.random.randn()

    self.dim = (num_fmaps, self.previous_layer.dim[0] - kernel_dim[0] + 1,
                self.previous_layer.dim[0] - kernel_dim[0] + 1)
    #also assumes kernel_dim is 3-D

  def propagate(self):
    #propagates through Conv layer
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    self.output = self.activation_func.calculate(np.array(
      [convolve(self.weights, a, mode = "valid") + self.bias
       for fmap in range(self.num_fmaps)]))
    #mode = "valid" allows for the convolution operator without padding

class Pooling(Layer):
  #basic pooling layer, for now, only max pooling is available

  def __init__(self, previous_layer, pool_dim, pool_type = "max"):
    #initializes a Pooling layer object
    assert isinstance(previous_layer, Conv), \
           "pooling layer must be preceded by conv layer"
    self.previous_layer = previous_layer
    self.pool_dim = pool_dim
    self.pool_type = pool_type

    self.dim = (self.previous_layer.dim[0],
                int(self.previous_layer.dim[1] / pool_dim[0]),
                int(self.previous_layer.dim[1] / pool_dim[0]))
    #again, assumes 3-D

  def propagate(self):
    #propagates through Pooling layer
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    self.output = []
    for fmap in a:
      #looping through all of the feature maps in the convolution layer's output
      chunks = np.array(list(map(lambda row : np.split(row, fmap.shape[1] / 2, 1),
                             np.split(fmap, fmap.shape[0] / 2, 0))))
      maxes = np.max(chunks.reshape(chunks.shape[0], chunks.shape[1], -1),
                   axis = 2)
      #reshape(..., -1)  lets the computer fill in the missing dimensions
      self.output.append(maxes)
    self.output = np.array(self.output)

class Dense(Layer):
  #basic dense layer with multiple activation and cost functions

  def __init__(self, previous_layer, num_neurons,
               activation_func = Activation("sigmoid"), cost_func = None):
    #initializes Dense layer
    self.previous_layer = previous_layer
    self.num_neurons = num_neurons
    self.activation_func = activation_func
    self.cost_func = cost_func

    self.weights = np.random.randn(self.num_neurons,
                          reduce(lambda x, y : x * y, self.previous_layer.dim))
    #the reduce function flattens the previous layers's output so that
    #computation is easier (especially with pooling layers)
    self.biases = np.random.randn(self.num_neurons, 1)

    self.dim = (self.num_neurons, 1)
    #doesn't assume 3-D!

  def propagate(self):
    #propagates through Dense layer
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    self.output = self.activation_func.calculate(
      np.dot(self.weights, a.reshape(-1, 1)) + self.biases)

#Testing area
if __name__ == "__main__":
  image = np.random.randn(28, 28)

  input_layer = Layer(image)

  conv_layer = Conv(input_layer, (5, 5), 3)
  conv_layer.propagate()

  pooling_layer = Pooling(conv_layer, (2, 2))
  pooling_layer.propagate()

  dense_layer = Dense(pooling_layer, 10)
  dense_layer.propagate()

  print ("Output:", dense_layer.output)
