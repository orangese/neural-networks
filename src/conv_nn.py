
"""

"conv_nn.py"

A program that houses code for a convolutional neural network. It is not very efficient-- this program is purely
for learning purposes. Also note that the structure of this file is much different than "mlp.py" in that the layers
are organized into their own classes.

"""

# Imports
import numpy as np
from scipy.signal import convolve2d
from functools import reduce
from src.mlp import Activation, Cost

# Classes
class Layer(object):
  """provides appropriate abstraction and acts as an input layer"""

  def __init__(self, dim):
    self.dim = dim
    self.output = np.zeros(self.dim)
    # dim refers to the dimensions of the output of this layer

class Conv(Layer):
  """2-D convolutional layer with stride of 1 and no padding"""

  def __init__(self, kernel_dim, num_filters, actv = "sigmoid", previous_layer = None, next_layer = None):
    self.kernel_dim = kernel_dim
    self.num_filters = num_filters
    self.actv = Activation(actv)

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    assert isinstance(self.next_layer, Pooling) or self.next_layer is None, "Pooling layer must follow Conv layer"

  def param_init(self):
    """initializes weights, biases, and gradients"""
    self.biases = np.random.normal(size = (self.num_filters, 1, 1))
    scale = np.sqrt(1.0 / (self.num_filters * np.prod(self.kernel_dim) / np.prod(self.next_layer.pool_dim)))
    # weight init is improved by squashing standard deviation using 1 / sqrt(n_in)
    self.weights = np.random.normal(scale = scale, size = (self.num_filters, *self.kernel_dim))

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  def propagate(self, backprop = False):
    """propagates through Conv layer, param_init not assumed"""
    try: self.biases
    except AttributeError: self.param_init()
    zs = Conv.convolve(self.weights, self.previous_layer.output) + self.biases
    self.output = self.actv.calculate(zs)
    self.dim = self.output.shape
    if backprop: self.zs = zs

  def backprop(self):
    """backpropagation through Conv layer, forward pass assumed"""
    self.error = self.next_layer.get_loc_fields(np.zeros(self.dim))
    np.put_along_axis(self.error, np.expand_dims(self.next_layer.max_args, axis = -1),
                      self.next_layer.error.reshape(*self.next_layer.dim, 1), axis = -1)
    # routes gradient such that only max activation neurons get nonzero gradient
    self.error = Pooling.consolidate(self.error) * self.actv.derivative(self.zs)

    self.nabla_b += np.sum(self.error, axis = (1, 2)).reshape(self.nabla_b.shape)
    self.nabla_w += Conv.convolve(self.previous_layer.output, self.error, is_err = True)

  def param_update(self, lr, minibatch_size):
    """weight and bias update"""
    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights -= (lr / minibatch_size) * self.nabla_w

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  @staticmethod
  def convolve(a, b, is_err = False):
    """convolves a_ with b_, order of convolution depends on err"""
    if is_err: return np.array([convolve2d(a, np.rot90(b_, 2), mode = "valid") for b_ in b])
    else: return np.array([convolve2d(np.rot90(a_, 2), b, mode = "valid") for a_ in a])
    # scipy.signal.convolve is consistent with the mathematical definition of convolve (which differs from the
    # usual machine learning definition of convolve), so rot180 must be applied to accomodate for the discrepancy

class Pooling(Layer):
  """2-D max pooling layer"""

  def __init__(self, pool_dim, previous_layer = None, next_layer = None):
    self.pool_dim = pool_dim

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    assert isinstance(self.previous_layer, Conv) or not self.previous_layer, "Conv layer must precede Pooling layer"

  def get_loc_fields(self, a):
    """divides convolutional output into local fields to prepare for pooling"""
    loc_fields = a.reshape(a.shape[0], int(a.shape[1] / self.pool_dim[0]), self.pool_dim[0],
                           int(a.shape[2] / self.pool_dim[1]), self.pool_dim[1]).transpose(0, 1, 3, 2, 4)
    return loc_fields.reshape(*loc_fields.shape[:3], -1)
    # example: divides input with shape (20, 24, 24) into array with shape (20, 12, 12, 4)

  def propagate(self, backprop = False):
    """propagates through Pooling layer"""
    fmaps = self.get_loc_fields(self.previous_layer.output)
    if backprop: self.max_args = np.argmax(fmaps, axis = -1)
    self.output = np.max(fmaps, axis = -1)
    self.dim = self.output.shape

  def backprop(self):
    """backpropagation through Pooling layer, forward pass assumed"""
    self.error = np.dot(self.next_layer.weights.T, self.next_layer.error)
    # activation of pooling layer is linear w.r.t. to the max activations in
    # the local pool, so the derivative of the activation function is 1

  @staticmethod
  def consolidate(a_):
    """consolidates local fields into convolutional output"""
    a = a_.reshape(*a_.shape[:3], int(a_.shape[3] / 2), int(a_.shape[3] / 2)).transpose(0, 1, 3, 2, 4)
    return a.reshape(a.shape[0], a.shape[1] * a.shape[2], a.shape[3] * a.shape[4])
    # example: consolidates input with shape (20, 12, 12, 4) into array with shape (20, 24, 24)

class Dense(Layer):
  """dense (MLP) layer with multiple activation and cost functions"""

  def __init__(self, num_neurons, actv = "sigmoid", reg = 0.0, dropout = 1.0, previous_layer = None, next_layer = None):
    self.num_neurons = num_neurons
    self.actv = Activation(actv)
    self.reg = reg
    self.dropout = dropout

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    self.dim = (self.num_neurons, 1)

  def param_init(self):
    """initializes layer parameters and gradients"""
    self.biases = np.random.normal(size = (self.num_neurons, 1))
    self.weights = np.random.normal(scale = np.sqrt(1.0 / self.num_neurons),
                                    size = (self.num_neurons, reduce(lambda a, b : a * b, self.previous_layer.dim)))

    if self.actv.name == "softmax":
      # weight init for softmax follows the book "Neural Networks and Deep Learning"
      self.biases = np.zeros(self.biases.shape)
      self.weights = np.zeros(self.weights.shape)

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

    self.dropout_mask = self.get_dropout_mask()

  def propagate(self, backprop = False):
    """propagates through Dense layer, param init not assumed"""
    try: self.biases
    except AttributeError: self.param_init()
    zs = np.dot(self.weights, self.previous_layer.output.reshape(-1, 1)) + self.biases
    if self.dropout != 1.0 and not backprop: zs *= self.dropout_mask
    self.output = self.actv.calculate(zs)
    if backprop: self.zs = zs

  def backprop(self, label = None):
    """backpropagation for Dense layer, forward pass assumed"""
    if self.next_layer is None: self.error = self.cost.get_error(self.actv, self.output, self.zs, label)
    else: self.error = np.dot(self.next_layer.weights.T, self.next_layer.error) * self.actv.derivative(self.zs)

    if self.dropout != 1.0: self.error *= self.dropout_mask

    self.nabla_b += self.error
    self.nabla_w += np.outer(self.error, self.previous_layer.output)

  def param_update(self, lr, minibatch_size, epoch_len):
    """weight and bias update, backprop assumed"""
    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights = (1.0 - lr * self.reg / epoch_len) * self.weights - (lr / minibatch_size) * self.nabla_w
    # update rule includes built-in L2 regularization

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

    self.dropout_mask = self.get_dropout_mask()

  def get_dropout_mask(self):
    return np.random.binomial(1, self.dropout, size = self.dim) / self.dropout

class Network(object):
  """uses Layer classes to create a functional network"""

  def __init__(self, layers = None, cost = "cross-entropy"):
    if layers is None: layers = []
    self.layers = tuple(layers)
    # self.layers is a tuple to prevent manual addition of layers
    assert len(self.layers) == 0 or isinstance(self.layers[0], Layer), "network layers must start with Layer object"
    for previous_layer, layer in zip(self.layers, self.layers[1:]): layer.previous_layer = previous_layer
    for next_layer, layer in zip(self.layers[1:], self.layers): layer.next_layer = next_layer
    # above loops link all the layers together
    self.cost = Cost(cost)
    if len(self.layers) != 0: self.layers[-1].cost = self.cost

  def add_layer(self, layer, position = None):
    """adds a layer and links it to other layers"""
    new_layers = list(self.layers)
    if not position: position = len(self.layers)
    new_layers.insert(position, layer)
    self.__init__(new_layers)
    # it seems inefficient and un-pythonic to rewrite self.layers each time you want to add a new layer,
    # but it seems to be the best way in order to prevent manual addition of layers

  def rm_layer(self, position = None, layer = None):
    """removes a layer and re-links it"""
    assert position or layer, "position or layer arguments must be provided"
    new_layers = list(self.layers)
    if layer: new_layers.remove(layer)
    elif position: del new_layers[position]
    self.__init__(new_layers)

  def propagate(self, a, backprop = False):
    """propagates input through network (similar to feed-forward)"""
    self.layers[0].output = a
    for layer in self.layers[1:]: layer.propagate(backprop = backprop)
    return self.layers[-1].output

  def backprop(self, example, label):
    """backprop through network by piecing together Layer backprop functions"""
    self.propagate(example, backprop = True)
    self.layers[-1].backprop(label)
    for layer in reversed(self.layers[1:len(self.layers) - 1]): layer.backprop()

  def param_update(self, lr, minibatch_size, epoch_len):
    """network parameter update"""
    for layer in reversed(self.layers[1:]):
      try: layer.param_update(lr, minibatch_size, epoch_len)
      except TypeError: layer.param_update(lr, minibatch_size)
      except AttributeError: continue

  def SGD(self, train_data, num_epochs, lr, minibatch_size, val_data = None):
    """stochastic gradient descent through network with L2 regularization"""
    for epoch_num in range(num_epochs):
      epoch = train_data
      np.random.shuffle(epoch)
      minibatches = [epoch[i:i + minibatch_size] for i in range(0, len(epoch), minibatch_size)]
      for minibatch in minibatches:
        for example, label in minibatch: self.backprop(example, label)
        self.param_update(lr, minibatch_size, len(epoch))
      if not (val_data is None):
        print ("Epoch {0}: accuracy: {1}% - cost: {2}".format(
          epoch_num + 1, self.eval_acc(val_data), self.eval_cost(val_data)))

  def eval_acc(self, data):
    """returns percent correct when the network is evaluated on parameter data"""
    results = [(np.argmax(self.propagate(x)), y) for (x, y) in data]
    return round(sum(int(x == y) for (x, y) in results) / len(data) * 100.0, 2)

  def eval_cost(self, data):
    """returns cost when the network is evaluated on parameter data"""
    return round(self.cost.calculate([(self.propagate(x), self.one_hot_encoding(y)) for (x, y) in data]).item(), 5)

  def one_hot_encoding(self, scalar):
    """function that vectorizes a scalar (one-hot encoding)"""
    encoding = np.zeros(self.layers[-1].dim)
    encoding[scalar] = 1.0
    return encoding