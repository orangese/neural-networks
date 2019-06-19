
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
  #provides appropriate abstraction and acts as an input layer

  def __init__(self, output):
    self.output = output
    self.dim = self.output.shape
    #it's understood that dim refers to the dimensions of the output

class Conv(Layer):
  #basic convolutional layer, stride is fixed at one

  def __init__(self, kernel_dim, num_fmaps, actv = Activation("sigmoid"),
               previous_layer = None):
    #initializes Conv layer
    #note that each layer has an attribute, previous_layer, which
    #is used for propagation
    self.kernel_dim = kernel_dim
    self.num_fmaps = num_fmaps
    self.actv = actv
    
    self.weights = np.random.randn(kernel_dim[0], kernel_dim[0]) \
                   / np.sqrt(self.kernel_dim[0])
    #assumes kernel_dim is 3-D
    self.bias = np.random.randn()

    self.previous_layer = previous_layer

  def propagate(self):
    #propagates through Conv layer
    assert self.previous_layer, "must have a preceding layer"
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    
    self.output = self.actv.calculate(np.array(
      [convolve(self.weights, a, mode = "valid") + self.bias
       for fmap in range(self.num_fmaps)]))
    #mode = "valid" allows for the convolution operator without padding

    self.dim = self.output.shape
    #assumes kernel_dim is 3-D

class Pooling(Layer):
  #basic pooling layer, for now, only max pooling is available

  def __init__(self, pool_dim, pool_type = "max", previous_layer = None):
    #initializes a Pooling layer object
    self.pool_dim = pool_dim
    self.pool_type = pool_type

    self.previous_layer = previous_layer
    if previous_layer:
      assert isinstance(previous_layer, Conv), \
             "pooling layer must be preceded by conv layer"

  def propagate(self):
    #propagates through Pooling layer
    assert isinstance(self.previous_layer, Conv), \
             "pooling layer must be preceded by convolutional layer"
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    self.output = []
    for fmap in a:
      #looping through all of the feature maps in the convolution layer's output
      new = np.array(list(map(lambda row : np.split(row, fmap.shape[1] / 2, 1),
                             np.split(fmap, fmap.shape[0] / 2, 0))))
      #new is the new array that stores the chunkified convolutional output
      maxes = np.max(new.reshape(new.shape[0], new.shape[1], -1), axis = 2)
      #reshape(..., -1)  lets the computer fill in the missing dimensions
      self.output.append(maxes)
    self.output = np.array(self.output)
    self.dim = self.output.shape

class Dense(Layer):
  #basic dense layer with multiple activation and cost functions

  def __init__(self, num_neurons, actv = Activation("sigmoid"),
               previous_layer = None):
    #initializes Dense layer
    self.num_neurons = num_neurons
    self.actv = actv
    self.previous_layer = previous_layer
    self.dim = (self.num_neurons, 1)

  def param_init(self):
    #initializes weights, biases, and dim using previous layer output shape
    self.weights = np.random.randn(self.num_neurons,
                          reduce(lambda x, y : x * y, self.previous_layer.dim))
    #the reduce function flattens the previous layers's output so that
    #computation is easier (especially with pooling layers)
    self.biases = np.random.randn(self.num_neurons, 1)
    #doesn't assume 3-D!

  def propagate(self):
    #propagates through Dense layer
    try: self.weights
    except AttributeError: self.param_init()
    #making sure weights exists

    assert self.previous_layer, "must have a preceding layer"
    a = self.previous_layer.output
    assert not (a is None), "input cannot be None"
    self.output = self.actv.calculate(
      np.dot(self.weights, a.reshape(-1, 1)) + self.biases)

class Network(object):
  #uses Layer classes to create a functional network

  def __init__(self, layers = [], cost = Cost("mse")):
    #initializes Network object with as many layers as user wants
    self.layers = layers
    if len(self.layers) != 0:
      assert isinstance(self.layers[0], Layer), \
           "network layers must start with Layer object"
    #self.layers is a tuple to prevent manual addition of layers
    for prev_layer, layer in zip(self.layers, self.layers[1:]):
      #linking all the layers together
      layer.previous_layer = prev_layer
    self.cost = cost

  def add_layer(self, layer, position = None):
    #adds a layer and links it to other layers
    new_layers = list(self.layers)
    if not position: position = len(self.layers)
    new_layers.insert(position, layer)
    self.__init__(tuple(new_layers))
    #it seems inefficient and un-pythonic to rewrite self.layers each time
    #you want to add a new layer, but it seems to be the best way in order
    #to prevent manual addition of layers

  def propagate(self, a = None):
    #propagates input through network (similar to feed-forward)
    if not (a is None): self.layers[0].output = a #choice of manual input
    for layer in self.layers[1:]:
      layer.propagate()
    return self.layers[-1].output

  def eval_acc(self, test_data, is_train = False):
    #returns percent correct when the network is evaluated using test data
    test_results = [(np.argmax(self.propagate(a = img)), label)
                    for (img, label) in test_data]
    if is_train:
      return round((sum(int(img == np.argmax(label)) for (img, label) in
                 test_results) / len(test_data) * 100.0), 2)
    else:
      return round((sum(int(img == label) for (img, label) in test_results) \
             / len(test_data) * 100.0), 2)

  def eval_cost(self, test_data, is_train = False):
    #returns cost when the network is evaluated using test data
    if is_train:
      return self.cost.calculate([(self.propagate(img), label)
                                  for (img, label) in test_data])
    else:
      return self.cost.calculate([(self.propagate(
        img), self.vectorize(label)) for (img, label) in test_data])

  def vectorize(self, num):
    #function that vectorizes a scalar (one-hot encoding)
    vector = np.zeros(self.layers[-1].dim)
    vector[num] = 1.0
    return vector

#Testing area
if __name__ == "__main__":
  from time import time

  start = time()
  
  data = [(np.zeros((28, 28)), np.random.randint(0, 10))
          for i in range(30000)]
  
  net = Network()
  net.add_layer(Layer(np.random.randn(28, 28)))
  net.add_layer(Conv((5, 5), 3))
  net.add_layer(Pooling((2, 2)))
  net.add_layer(Dense(10))

  print ("Accuracy (%):", net.eval_acc(data),
         "\nCost:", net.eval_cost(data))

  print ("Output:", net.propagate(data[0][0]),
         "\nTarget:", net.vectorize(data[0][1]))

  print ("Time (sec):", round(time() - start, 3))

