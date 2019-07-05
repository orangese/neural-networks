
"""

"conv_nn.py"

A program that houses code for a convolutional neural network. It is not
very efficient-- this program is purely for learning purposes. Also note that
the structure of this file is much different than "mlp.py" in that the
layers are organized into their own classes.

Progress:

1. 6/18/19: created Layer, Conv, Pooling, and Dense classes, along with their
   respective feed-forward functions. As of now, they seem functional,
   but feeding forward through a convolutional network is taking around
   10 times longer than propagating through an MLP (for a single example).

2. 6/19/19: created Network class as implementation of Layer classes.
   Initially, propagating 10000 examples through a convolutional net
   with a convolutional layer, a max-pooling layer, and a dense layer
   took 40 seconds (compared to ~0.7 seconds in a 3-layer MLP with
   100 hidden neurons and the same input/output structure). This time was
   decreased to ~4.5 seconds by improving the max pooling function. However,
   propagating 60000 examples (~size of the MNIST dataset) took 30 seconds,
   which is worrisome (compared to ~3.3 seconds in the MLP network).

3. 6/25/19: implemented backpropagation and SGD. However, training is unstable
   and inaccurate (accuracy with 1,000 MNIST images is 40%-70%), suggesting that
   there is an error with backpropagation/forward-propagation somewhere in
   the network. An MLP compiled with this program performs as well as an MLP
   created with "mlp.py", suggesting that there error lies in the
   convolutional and/or pooling layers.

   Structure: Layer((28, 28)), Conv((5, 5), 3), Pooling((2, 2)), Dense(10)
   Learning rate: 0.4
   Minibatch size: 20

4. 6/26/19: fixed backpropagation and pooling in convolutional layer. On 1,000
   train images, the network reaches ~81% accuracy, slightly higher than the
   MLP's highest accuracy on the same training set (~80%). The convolutional
   network takes about 10 times longer to train but also yields greater accuracy
   in fewer epochs. Note that I am not interested in optimizing this network
   (i.e., trying to make it super-accurate)-- the building of this program
   was purely for educational purposes, not to produce a powerful network.

   Structure: Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(10)
   Learning rate: 0.4
   Minibatch size: 20

5. 6/28/19: network still broken. ~95% accuracy achieved on MNIST after 2 epochs
   (which took ~1.1 hours per epoch), which is too low, suggesting that there
   is something wrong with backpropagation. Additionally, small changes to
   weight initialization cause the network to become a noise machine, and
   online learning yields greater accuracy/stability than SGD.

   Structure: Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)), Dense(100),
              Dense(10)
   Learning rate: 0.4
   Minibatch size: 10

6. 7/4/19: fixed forward propagation in max pooling layer but
   RuntimeWarnings galore (hooray...):
   
   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 79
      return 1.0 / (1.0 + np.exp(-z))
   RuntimeWarning: overflow encountered in exp

   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 57
      for (a, y) in pairs)) / len(pairs)
   RuntimeWarning: divide by zero encountered in log

   Warning (from warnings module):
    File "/Users/ryan/Documents/Coding/neural-networks/src/mlp.py", line 57
     for (a, y) in pairs)) / len(pairs)
   RuntimeWarning: invalid value encountered in multiply

   Strangely, these errors only occur when training on the full dataset.
   Unlike last time these errors occurred, the problem is not with the
   noramlization of the data.

7. 7/4/19: more efficient implementation of backpropagation and forward
   propagation using flattened weights and error vectors.

"""

#Imports
import numpy as np
from scipy.signal import convolve
from functools import reduce
from time import time
from mlp import Activation, Cost

#Classes
class Layer(object):
  #provides appropriate abstraction and acts as an input layer

  def __init__(self, dim):
    self.dim = dim
    self.output = np.random.normal(size = self.dim)
    #dim refers to the dimensions of the output of this layer

class Conv(Layer):
  #basic 2-D convolutional layer, stride is fixed at one

  def __init__(self, kernel_dim, num_fmaps, actv = Activation("sigmoid"),
               previous_layer = None, next_layer = None):
    self.kernel_dim = kernel_dim
    self.num_fmaps = num_fmaps
    self.actv = actv

    self.previous_layer = previous_layer
    self.next_layer = next_layer
    if self.next_layer:
      assert isinstance(self.next_layer, Pooling), \
             "Conv layer must be followed by Pooling layer"
      self.param_init()

  def param_init(self):
    #initializes weights, biases, and gradients
    self.biases = np.random.normal(size = (self.num_fmaps, 1, 1))
    n_out = self.num_fmaps * np.prod(self.kernel_dim)
    try: n_out /= np.prod(self.next_layer.pool_dim)
    except AttributeError: n_out /= np.prod(self.next_layer.num_neurons)
    self.weights = np.random.normal(loc = 0, scale = np.sqrt(1.0 / n_out),
                                    size = (self.num_fmaps, *self.kernel_dim))
    
    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

  def propagate(self, backprop = False):
    #propagates through Conv layer, param_init not assumed
    try: self.biases
    except AttributeError: self.param_init()
    zs = np.array([
      convolve(self.weights[fmap_num], self.previous_layer.output, mode = "valid")
      + self.biases[fmap_num] for fmap_num in range(self.num_fmaps)])
    
    self.output = self.actv.calculate(zs)
    self.dim = self.output.shape
    if backprop: self.zs = zs

  def backprop(self):
    #backpropagation through Conv layer, forward pass assumed
    if isinstance(self.next_layer, Pooling):
      self.error = self.next_layer.get_loc_fields(np.zeros(self.dim))
      np.put_along_axis(self.error, self.next_layer.max_args,
                        self.next_layer.error.reshape(*self.next_layer.dim, 1),
                        axis = 3)
      self.error = self.next_layer.consolidate(self.error)
    else:
      self.error = np.dot(self.next_layer.weights.T, self.next_layer.error) * \
                   self.actv.derivative(self.zs).reshape(-1, 1)
      
    self.nabla_b += np.sum(self.error.reshape(self.num_fmaps, -1))
    self.nabla_w += np.array([
      convolve(self.previous_layer.output, err, mode = "valid")
      for err in self.error.reshape(*self.dim)])

  def param_update(self, lr, minibatch_size):
    #weight and bias update
    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights -= (lr / minibatch_size) * self.nabla_w
    
    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

class Pooling(Layer):
  #basic pooling layer, for now, only 2-D max pooling is available

  def __init__(self, pool_dim, pool_type = "max", previous_layer = None,
               next_layer = None):
    self.pool_dim = pool_dim
    self.pool_type = pool_type

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
    a = np.copy(a_).reshape(*a_.shape[:3], int(a_.shape[3] / 2),
                            int(a_.shape[3] / 2)).transpose(0, 1, 3, 2, 4)
    return a.reshape(a.shape[0], a.shape[1] * a.shape[2],
                     a.shape[3] * a.shape[4])

  def propagate(self, backprop = False):
    #propagates through Pooling layer
    fmaps = self.get_loc_fields(self.previous_layer.output)
    if backprop:
      try: print (self.cached == self.max_args)
      except AttributeError: pass
      self.max_args = np.expand_dims(np.argmax(fmaps, axis = 3), axis = 3)
      self.cached = self.max_args
    self.output = np.max(fmaps, axis = 3)
    self.dim = self.output.shape

  def backprop(self):
    #backpropagation through Pooling layer, forward pass assumed
    self.error = np.dot(self.next_layer.weights.T, self.next_layer.error)
    #activation of pooling layer is linear w.r.t. to the max activations in
    #the local pool, so the derivative of the activation function is 1

class Dense(Layer):
  #basic dense layer with multiple activation and cost functions

  def __init__(self, num_neurons, actv = Activation("sigmoid"), cost = None,
               previous_layer = None, next_layer = None):
    self.num_neurons = num_neurons
    self.actv = actv
    self.cost = cost
    
    self.previous_layer = previous_layer
    self.next_layer = next_layer
    assert not self.cost or not self.next_layer, \
           "Dense layer cannot have both a cost attribute and a \
            next_layer attribute"
    assert not self.next_layer, "output layer cannot have dropout applied"
    self.dim = (self.num_neurons, 1)

  def param_init(self):
    #initializes layer parameters and gradients
    self.biases = np.random.normal(size = (self.num_neurons, 1))
    self.weights = np.random.normal(scale = np.sqrt(1.0 / self.num_neurons),
                                    size = (self.num_neurons,
                                            reduce(lambda a, b : a * b,
                                                   self.previous_layer.dim)))
    #the reduce function flattens the previous layers's output so that
    #computation is easier (especially with pooling layers)

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

  def backprop(self, img = None, label = None):
    #backpropagation for Dense layer, forward pass assumed
    if not (img is None):
      self.error = self.cost.get_error(self.actv, self.output, self.zs, label)
    else:
      self.error = np.dot(self.next_layer.weights.T, self.next_layer.error) * \
          self.actv.derivative(self.zs)
      
    self.nabla_b += self.error
    self.nabla_w += np.outer(self.error, self.previous_layer.output)

  def param_update(self, lr, minibatch_size, reg = 0.0):
    #weight and bias update, backprop assumed
    self.nabla_w += (reg / minibatch_size) * self.weights
    #L2 regularization

    self.biases -= (lr / minibatch_size) * self.nabla_b
    self.weights -= (lr / minibatch_size) * self.nabla_w

    self.nabla_b = np.zeros(self.biases.shape)
    self.nabla_w = np.zeros(self.weights.shape)

class Network(object):
  #uses Layer classes to create a functional network

  def __init__(self, layers = [], cost = Cost("cross-entropy")):
    self.layers = tuple(layers)
    #self.layers is a tuple to prevent manual addition of layers
    assert len(self.layers) == 0 or isinstance(self.layers[0], Layer), \
             "network layers must start with Layer object"
    for previous_layer, layer in zip(self.layers, self.layers[1:]):
      layer.previous_layer = previous_layer
    for next_layer, layer in zip(self.layers[1:], self.layers):
      layer.next_layer = next_layer
    #above loops link all the layers together
    self.cost = cost
    if len(self.layers) != 0: self.layers[-1].cost = cost

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
    self.layers[-1].backprop(img, label)
    for layer in reversed(self.layers[1:len(self.layers) - 1]):
      layer.backprop()

  def param_update(self, lr, minibatch_size):
    #network parameter update
    for layer in self.layers[1:]:
      try: layer.param_update(lr, minibatch_size, reg = self.cost.reg_parameter)
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
        self.param_update(lr, minibatch_size)
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

#Testing area
def generate_zero_data():
  #generates zero data in the shape of MNIST data
  data = {"train": [], "validation": [], "test": []}
  target = np.zeros((10, 1))
  target[0] = 1.0
  data["train"] = [(np.ones((28, 28)), target) for i in range(1000)]
  data["validation"] = [(np.ones((28, 28)), 0) for i in range(1000)]
  data["test"] = [(np.ones((28, 28)), 0) for i in range(1000)]
  return data

def test(net_type = "conv", data = None, test_acc = False, test_cost = False):
  if data is None: data = generate_zero_data()
  
  if net_type == "conv":
    net = Network([Layer((28, 28)), Conv((5, 5), 20), Pooling((2, 2)),
##                   Dense(100),
                   Dense(10, actv = Activation("softmax"))],
                  cost = Cost("log-likelihood", reg_parameter = 0.0))
  elif net_type == "mlp":
    net = Network([Layer((28, 28)), Dense(100), Dense(10)])

  start = time()

  if test_acc:
    print ("Evaluation without training: {0}%".format(
      net.eval_acc(data["test"])))
  
  net.SGD(data["train"], 5, 0.1, 10, data["validation"])

  for i in range(10):
    pred = net.propagate(data["test"][i][0])
    print (np.max(pred), np.argmax(pred) == data["test"][i][1])

  if test_acc: print ("Accuracy: {0}%".format(net.eval_acc(data["test"])))
  if test_cost: print ("Cost: {0}".format(net.eval_cost(data["test"])))
  print ("Time elapsed: {0} seconds".format(round(time() - start, 3)))

  return net
  
if __name__ == "__main__":
  test(net_type = input("MLP or ConvNN test? (mlp/conv): "))
