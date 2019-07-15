
"""

"conv_nn.py"

A program that houses code for a convolutional neural network. It is not
very efficient-- this program is purely for learning purposes. Also note that
the structure of this file is much different than "mlp.py" in that the
layers are organized into their own classes.

"""

#Imports
import numpy as np
from scipy.signal import convolve, convolve2d
from functools import reduce
from mlp import Activation, Cost

#Classes
class Layer(object):
  #provides appropriate abstraction and acts as an input layer

  def __init__(self, dim):
    self.dim = dim
    self.output = np.zeros(self.dim)
    #dim refers to the dimensions of the output of this layer

class Conv(Layer):
  #basic 2-D convolutional layer with stride = 1

  def __init__(self, kernel_dim, num_filters, num_fmaps = 1, actv = "sigmoid",
               previous_layer = None, next_layer = None):
    self.kernel_dim = kernel_dim
    self.num_filters = num_filters
    self.num_fmaps = num_fmaps
    self.actv = Activation(actv)

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    assert isinstance(self.next_layer, Pooling) or self.next_layer is None, \
           "Conv layer must be followed by Pooling layer"

  def param_init(self):
    #initializes weights, biases, and gradients
    self.biases = np.random.normal(size = (self.num_filters, 1, 1))
    scale = np.sqrt(1.0 / (self.num_filters * np.prod(self.kernel_dim) \
            / np.prod(self.next_layer.pool_dim)))
    size = [self.num_filters, *self.kernel_dim]
    if self.num_fmaps != 1: size.append(self.num_fmaps)
    self.weights = np.random.normal(loc = 0, scale = scale, size = size)

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  def propagate(self, backprop = False):
    #propagates through Conv layer, param_init not assumed
    try: self.biases
    except AttributeError: self.param_init()
    zs = self.convolve(self.weights, self.previous_layer.output) + self.biases
    self.output = self.actv.calculate(zs)
    self.dim = self.output.shape
    if backprop: self.zs = zs

  def backprop(self):
    #backpropagation through Conv layer, forward pass assumed
    self.error = self.next_layer.get_loc_fields(np.zeros(self.dim))
    np.put_along_axis(self.error, self.next_layer.max_args,
                      self.next_layer.error.reshape(*self.next_layer.dim, 1),
                      axis = 3)
    self.error = self.next_layer.consolidate(self.error)

    self.nabla_b += np.sum(self.error, axis = (1, 2))[..., np.newaxis,
                                                      np.newaxis]
    self.nabla_w += self.convolve(self.previous_layer.output, self.error, True)

  def param_update(self, lr, minibatch_size):
    #weight and bias update
    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights -= (lr / minibatch_size) * self.nabla_w

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  def convolve(self, _a, _b, reverse = False):
    a, b = _a, _b
    if isinstance(self.previous_layer, Pooling):
      if reverse: a = _a.reshape(*reversed(_a.shape))
      else: b = _b.reshape(*reversed(_b.shape))
    if self.num_fmaps != 1:
      if reverse: return np.squeeze([convolve(b_, a, "valid") for b_ in b])
      else: return np.squeeze([convolve(b, a_, "valid") for a_ in a])
    else:
      if reverse: return np.array([convolve2d(a, b_, "valid") for b_ in b])
      else: return np.array([convolve2d(a_, b, "valid") for a_ in a])

class Pooling(Layer):
  #basic pooling layer, for now, only 2-D max pooling is available

  def __init__(self, pool_dim, previous_layer = None, next_layer = None):
    self.pool_dim = pool_dim

    self.previous_layer = previous_layer
    if self.previous_layer:
      assert isinstance(self.previous_layer, Conv), \
             "Pooling layer must be preceded by Conv layer"
    self.next_layer = next_layer

  def get_loc_fields(self, a):
    #divides the convolutional output into local fields to prepare for pooling
    loc_fields = a.reshape(a.shape[0], int(a.shape[1] / self.pool_dim[0]),
                  self.pool_dim[0], int(a.shape[2] / self.pool_dim[1]),
                  self.pool_dim[1]).transpose(0, 1, 3, 2, 4)
    return loc_fields.reshape(*loc_fields.shape[:3], -1)

  def consolidate(self, a_):
    #consolidates local fields into convolutional output
    a = a_.reshape(*a_.shape[:3], int(a_.shape[3] / 2),
                   int(a_.shape[3] / 2)).transpose(0, 1, 3, 2, 4)
    return a.reshape(a.shape[0], a.shape[1] * a.shape[2],
                     a.shape[3] * a.shape[4])

  def propagate(self, backprop = False):
    #propagates through Pooling layer
    fmaps = self.get_loc_fields(self.previous_layer.output)
    if backprop:
      self.max_args = np.expand_dims(np.argmax(fmaps, axis = 3), axis = 3)
    self.output = np.max(fmaps, axis = 3)
    self.dim = self.output.shape

  def backprop(self):
    #backpropagation through Pooling layer, forward pass assumed
    self.error = np.dot(self.next_layer.weights.T, self.next_layer.error)
    #activation of pooling layer is linear w.r.t. to the max activations in
    #the local pool, so the derivative of the activation function is 1

class Dense(Layer):
  #basic dense layer with multiple activation and cost functions

  def __init__(self, num_neurons, actv = "sigmoid", reg = 0.0,
               previous_layer = None, next_layer = None):
    self.num_neurons = num_neurons
    self.actv = Activation(actv)
    self.reg = reg

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    self.dim = (self.num_neurons, 1)

  def param_init(self):
    #initializes layer parameters and gradients
    self.biases = np.random.normal(size = (self.num_neurons, 1))
    self.weights = np.random.normal(scale = np.sqrt(1.0 / self.num_neurons),
                                    size = (self.num_neurons,
                                            reduce(lambda a, b : a * b,
                                                   self.previous_layer.dim)))
    
    if self.actv.name == "softmax":
      self.biases = np.zeros(self.biases.shape)
      self.weights = np.zeros(self.weights.shape)

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  def propagate(self, backprop = False):
    #propagates through Dense layer, param init not assumed
    try: self.biases
    except AttributeError: self.param_init()
    zs = np.dot(self.weights, self.previous_layer.output.reshape(-1, 1)) \
         + self.biases
    self.output = self.actv.calculate(zs)
    if backprop: self.zs = zs

  def backprop(self, label = None):
    #backpropagation for Dense layer, forward pass assumed
    try:
      self.error = self.cost.get_error(self.actv, self.output, self.zs, label)
    except AttributeError:
      self.error = np.dot(self.next_layer.weights.T, self.next_layer.error) * \
                   self.actv.derivative(self.zs)

    self.nabla_b += self.error
    try:
      self.nabla_w += np.outer(self.error, self.previous_layer.output)
    except FloatingPointError:
      print (np.linalg.norm(self.error))
      print (np.linalg.norm(self.previous_layer.output))
      print (self.error)
      print (self.previous_layer.output)
      print (self.next_layer)
      raise FloatingPointError("dense backprop, nabla_")

  def param_update(self, lr, minibatch_size, epoch_len):
    #weight and bias update, backprop assumed
    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights = (1.0 - lr * self.reg / epoch_len) * self.weights \
                   - (lr / minibatch_size) * self.nabla_w

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

class Network(object):
  #uses Layer classes to create a functional network

  def __init__(self, layers = [], cost = "cross-entropy"):
    self.layers = tuple(layers)
    #self.layers is a tuple to prevent manual addition of layers
    assert len(self.layers) == 0 or isinstance(self.layers[0], Layer), \
             "network layers must start with Layer object"
    for previous_layer, layer in zip(self.layers, self.layers[1:]):
      layer.previous_layer = previous_layer
    for next_layer, layer in zip(self.layers[1:], self.layers):
      layer.next_layer = next_layer
    #above loop link all the layers together
    self.cost = Cost(cost)
    if len(self.layers) != 0: self.layers[-1].cost = self.cost

  def add_layer(self, layer, position = None):
    #adds a layer and links it to other layers
    new_layers = list(self.layers)
    if not position: position = len(self.layers)
    new_layers.insert(position, layer)
    self.__init__(new_layers)
    #it seems inefficient and un-pythonic to rewrite self.layers each time
    #you want to add a new layer, but it seems to be the best way in order
    #to prevent manual addition of layers

  def rm_layer(self, position = None, layer = None):
    #removes a layer and re-links it
    assert position or layer, "position or layer arguments must be provided"
    new_layers = list(self.layers)
    if layer: new_layers.remove(layer)
    elif position: del new_layers[position]
    self.__init__(new_layers)

  def propagate(self, a, backprop = False):
    #propagates input through network (similar to feed-forward)
    self.layers[0].output = a
    for layer in self.layers[1:]: layer.propagate(backprop = backprop)
    return self.layers[-1].output

  def backprop(self, img, label):
    #backprop through network by piecing together Layer backprop functions
    self.propagate(img, backprop = True)
    self.layers[-1].backprop(label)
    for layer in reversed(self.layers[1:len(self.layers) - 1]): layer.backprop()

  def param_update(self, lr, minibatch_size, epoch_len):
    #network parameter update
    for layer in reversed(self.layers[1:]):
      try: layer.param_update(lr, minibatch_size, epoch_len)
      except TypeError: layer.param_update(lr, minibatch_size)
      except AttributeError: continue

  def SGD(self, train_data, num_epochs, lr, minibatch_size, val_data = None):
    #stochastic gradient descent through network with L2 regularization
    for epoch_num in range(num_epochs):
      epoch = train_data
      np.random.shuffle(epoch)
      minibatches = [epoch[i:i + minibatch_size] for i in
                     range(0, len(epoch), minibatch_size)]
      for minibatch in minibatches:
        for image, label in minibatch:
          self.backprop(image, label)
        self.param_update(lr, minibatch_size, len(epoch))
      if not (val_data is None):
        print ("Epoch {0}: accuracy: {1}% - cost: {2}".format(
          epoch_num + 1, self.eval_acc(val_data), self.eval_cost(val_data)))

  def eval_acc(self, test_data):
    #returns percent correct when the network is evaluated using test data
    test_results = [(np.argmax(self.propagate(img)), label)
                    for (img, label) in test_data]
    return round(sum(int(img == label) for (img, label) in test_results) \
                  / len(test_data) * 100.0, 2)

  def eval_cost(self, test_data, is_train = False):
    #returns cost when the network is evaluated using test data
    return round(np.asscalar(self.cost.calculate([(self.propagate(
      img), self.vectorize(label)) for (img, label) in test_data])), 5)

  def vectorize(self, num):
    #function that vectorizes a scalar (one-hot encoding)
    vector = np.zeros(self.layers[-1].dim)
    vector[num] = 1.0
    return vector

if __name__ == "__main__":
  np.seterr(all = "raise")
  net = Network([Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)),
                 Conv((5, 5), 20, 20), Pooling((2, 2)), Dense(100), Dense(10)])
  net.propagate(np.zeros((28, 28)))
