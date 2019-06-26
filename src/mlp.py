"""

"mlp.py"

A program that houses a vanilla feed-forward neural network

Type: vanilla neural network (MLP feedforward)
Activation function(s): sigmoid, softmax
Architecture: chosen by user
Cost function(s): MSE, cross-entropy, log-likelihood
Training: vanilla minibatch SGD and BP
Early stopping: GL, aGL, minimum aGL, average improvement
Regularization: L1, L2
Learning rate: variable (uses early stopping to determine when to switch)
Other hyperparameters: chosen by user

To-do:
1. Add momentum for SGD-- DONE
2. Dropout
3. *Artificial expansion (write in data loader files)
4. Better LR variation (simulated annealing?)
5. Max-norm constraint (using L2 norm, ||w||2 < c)

"""

#Libraries
import numpy as np #for fast matrix-based computations
from time import time #for timing stuff

#Classes
class Cost(object):
  #class for the different cost functions

  def __init__(self, name, regularization = None, reg_parameter = None):
    self.name = name
    self.regularization = regularization
    self.reg_parameter = reg_parameter

  def derivative(self, a, y):
    #returns ∂Cx/∂a_L
    if self.name == "mse":
      return (a - y)
    elif self.name == "cross-entropy":
      return ((a - y) / (a * (1 - a)))
    elif self.name == "log-likelihood":
      return (y / a)

  def calculate(self, pairs, weights = None):
    #accepts a list of tuples (a, y) and returns average cost over that list
    if self.name == "mse":
      cost = np.sum(np.linalg.norm(a - y) ** 2.0 for (a, y) in pairs) \
             / (2.0 * len(pairs))
    elif self.name == "cross-entropy":
      cost = np.sum(
        np.sum(np.nan_to_num(-y * np.log(a) - (1.0 - y) * np.log(1.0 - a))
               for (a, y) in pairs)) / len(pairs)
    elif self.name == "log-likelihood":
      cost = np.sum(np.log(a[np.argmax(y)]) for (a, y) in pairs) \
             / (-1.0 * len(pairs))

##    if self.regularization == "L2":
##      cost += np.sum(weights ** 2.0)
##    elif self.regularization == "L1":
##      cost += np.sum(np.absolute(weights))
    
    return cost

  def get_error(self, activation, activations, weighted_inputs, label):
    if self.name == "mse":
      return self.derivative(activations, label) * \
             activation.derivative(weighted_inputs)
    elif self.name == "cross-entropy" or self.name == "log-likelihood":
      return (activations - label)

class Activation(object):

  def __init__(self, name):
    self.name = name

  def calculate(self, z):
    if self.name == "sigmoid":
      return 1.0 / (1.0 + np.exp(-z))
    elif self.name == "softmax":
      return np.exp(z) / np.sum(np.exp(z))

  def derivative(self, z, j = None, i = None):
    if self.name == "sigmoid":
      return self.calculate(z) * (1.0 - self.calculate(z))
    elif self.name == "softmax":
      if j is None or i is None:
        raise TypeError("arguments 'i' or 'j' not provided") 
      gradient = np.array([-1.0 * self.calculate(z_k)[j] * 
                           self.calculate(z_k)[i] if j != i else 
                           self.calculate(z)[j] * (1.0 - self.calculate(z)[j])
                           for z_k in z])
      return gradient

class Early_Stop(object):
  #BEST METHOD: average improvement, others are just for demonstration
  """Note that all the methods in this class need to be used in conjunction
  with some kind of loop-- they need to be supplemented with some other code"""

  @staticmethod #method can be called without creating an Early_Stop object
  def GL(accuracy, stop_parameter):
    """returns "stop" if the generalization loss exceeds a parameter. If a new
    accuracy maximum has been found, this function returns "new". Otherwise,
    the function returns None"""
    local_opt = max(accuracy)
    generalization_loss = local_opt - accuracy[-1]
    if local_opt < accuracy[-1]:
      return "new"
    elif generalization_loss > stop_parameter:
      return "stop"
    else:
      return None

  @staticmethod
  def average_GL(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss over the last n epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      local_opt = max(accuracy)
      accuracy = accuracy[len(accuracy) - n:]
      average_gl = sum([local_opt - accuracy[-i - 1] for i in range(n)]) / \
                 len(accuracy)
      if average_gl > stop_parameter:
        return "stop"
      elif local_opt < accuracy[-1]:
        return "new"
    else:
      return None

  @staticmethod
  def average_improvement(accuracy, stop_parameter, n):
    """returns "stop" if average improvement over the last n epochs
    has not exceeded a parameter. If a new accuracy maximum has been found,
    this function returns "new". Otherwise, the function returns None"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      accuracy = accuracy[len(accuracy) - n:]
      average_improvement = sum([accuracy[-i - 1] - accuracy[-i - 2] for i in
                        range(n - 1)]) / len(accuracy)
      if average_improvement < stop_parameter:
        return "stop"
      elif max(accuracy) < accuracy[-1]:
        return "new"
    else:
      return None

class Network(object): 
  
  def __init__(self, layers, cost_function = Cost("mse"),
               body_activation = Activation("sigmoid"),
               output_activation = Activation("sigmoid"),
               weight_init = "regular"):
    """initializes Network object. Note that each layer has an
    array of biases of shape (n, 1) and an array of weights of
    shape (n, m), where n is the number of neurons in the layer
    and m is the number of layers in the previous layer"""
    self.layers = layers
    self.large_weight_init() if weight_init == "large" else \
                               self.regular_weight_init()
    self.bias_init()

    self.cost = cost_function
    self.activation = body_activation
    self.output_activation = output_activation

  def regular_weight_init(self):
    #squashes the distribution of pre-train weights, leading to better training
    self.weights = np.array([np.random.randn(next_layer, layer) / np.sqrt(layer)
                             for layer, next_layer in zip(
                               self.layers, self.layers[1:])])
    self.weight_init = "regular"

  def large_weight_init(self):
    #does not squash distribution of pre-train weights
    self.weights = np.array([np.random.randn(next_layer, layer)
                             for layer, next_layer in zip(
                               self.layers, self.layers[1:])])
    self.weight_init = "large"

  def bias_init(self):
    #regular bias initializer
    self.biases = np.array([np.random.randn(layer, 1) for layer in
                            self.layers[1:]])
    
  def feed_forward(self, a):
    #feeds an input into the network and returns its output
    for b, w in zip(self.biases[:-1], self.weights[:-1]):
      a = self.activation.calculate(np.dot(w, a) + b)
    a = self.output_activation.calculate(np.dot(self.weights[-1], a) +
                                         self.biases[-1])
    return a
  
  def SGD(self, training_data, num_epochs, learning_rate, minibatch_size,
          validation_data = None, test_data = None, monitor = False,
          early_stopping = None, lr_variation = None, momentum = None,
          dropout = None):
    """implements stochastic gradient descent to minimize the cost function.
    This function relies on the self.backprop function to calculate the
    gradient of the cost function, which is necessary for the updating of
    weights and biases"""
    
    if monitor or early_stopping or lr_variation:
      evaluation = {"validation accuracy": [], "validation cost": [],
                    "train accuracy": [], "train cost": []}
    if early_stopping:
      stored_biases = self.biases
      stored_weights = self.weights
      to_stop = False
      """format for early_stopping parameter is [GL_type, stop_parameter,
      aGL_strip_GL_parameter]"""

    if lr_variation:
      original_lr = learning_rate
      change_lr = False
      to_evaluate = []
      """format for lr_variation parameter is [GL_type, stop_parameter,
      aGL_strip_GL_parameter, lr_variation_parameter, lr_variation_cutoff]"""

    if momentum:
      bias_velocities = np.array([np.zeros(b.shape) for b in self.biases])
      weight_velocities = np.array([np.zeros(w.shape) for w in self.weights])
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch)
      minibatches = [epoch[i:i + minibatch_size] for i in
                     range(0, len(epoch), minibatch_size)]
      
      for minibatch in minibatches:
        nabla_b, nabla_w = self.backprop(minibatch, dropout = dropout)

        if self.cost.regularization == "L1":
          nabla_w += (self.cost.reg_parameter / len(epoch)
                                           * np.sign(self.weights))
        elif self.cost.regularization == "L2":
          nabla_w += (self.cost.reg_parameter / len(epoch) 
                                           * self.weights) 
        
        self.biases -= (learning_rate / minibatch_size) * nabla_b
        self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None:
        print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_accuracy = self.evaluate_accuracy(validation_data)
        validation_cost = self.evaluate_cost(validation_data)
        print ("Epoch {0}: accuracy: {1}% - cost: {2}".format(
          epoch_num + 1, validation_accuracy, round(float(validation_cost), 4)))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_accuracy)
        if lr_variation:
          to_evaluate.append(validation_accuracy)
        if monitor:
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
          evaluation["validation cost"].append(validation_cost)
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping:
        to_stop = early_stopping[0](evaluation["validation accuracy"],
                                    early_stopping[1], early_stopping[2])
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights

      if lr_variation:
        change_lr = lr_variation[0](to_evaluate, lr_variation[1],
                                    lr_variation[2])
        if change_lr == "stop":
          learning_rate /= lr_variation[3]
          to_evaluate = []
        if original_lr * lr_variation[4] >= learning_rate:
          print ("End SGD: learning rate parameter exceeded")
          break

      if dropout:
        for layer in dropout[0]:
          self.weights[layer] *= dropout[1][dropout[0].index(layer)]

    if not (test_data is None):
      print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))
      
    if monitor or early_stopping:
      return evaluation

  def backprop(self, minibatch, dropout = None):
    #calculates the gradients of the cost function w.r.t. weights and biases
    a, labels = list(zip(*minibatch))
    a = np.array(a)
    labels = np.array(labels)

    weighted_inputs = []
    activations = [a]

    #Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a).transpose(1, 0, 2) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(z)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])

    if dropout:
      #implementation is not optimized-- loop is slow
      dropout_matrix = self.dropout(dropout[0], dropout[1]) 
      activations = [act * drop
                     for act, drop in zip(activations, dropout_matrix)]

    nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.asarray([np.zeros(w.shape) for w in self.weights])
    
    #Step 2: computing the output error
    error = self.cost.get_error(self.output_activation, activations[-1],
                                weighted_inputs[-1], labels)
    nabla_b[-1] = error
    nabla_w[-1] = np.array([np.outer(err, act)
                            for err, act in zip(error, activations[-2])])
    #loop is slow

    #Step 3: computing the errors for the rest of the layers
    l = len(self.layers) - 2
    while l > 0:
      error = np.dot(self.weights[l].T, error).transpose(1, 0, 2) * \
             self.activation.derivative(weighted_inputs[l - 1])
      #computing the error for layer "l + 1" (index "l")
      nabla_b[l - 1] = error
      nabla_w[l - 1] = np.array([np.outer(err, act) for err, act
                                 in zip(error, activations[l-1])])
      #loop is slow
      l -= 1

    nabla_b = np.array([np.sum(b, axis = 0) for b in nabla_b])
    nabla_w = np.array([np.sum(w, axis = 0) for w in nabla_w])

    return (nabla_b, nabla_w)

  def unvectorized_SGD(self, training_data, num_epochs, learning_rate,
                       minibatch_size, validation_data = None, test_data = None,
                       monitor = False, early_stopping = None,
                       lr_variation = None, momentum = None, dropout = None):
    """implements stochastic gradient descent to minimize the cost function.
    This function relies on the self.backprop function to calculate the
    gradient of the cost function, which is necessary for the updating of
    weights and biases"""
    #unvectorized but faster than vetorzied if using dropout
    
    if monitor or early_stopping or lr_variation:
      evaluation = {"validation accuracy": [], "validation cost": [],
                    "train accuracy": [], "train cost": []}
    if early_stopping:
      stored_biases = self.biases
      stored_weights = self.weights
      to_stop = False
      """format for early_stopping parameter is [GL_type, stop_parameter,
      aGL_strip_GL_parameter]"""

    if lr_variation:
      original_lr = learning_rate
      change_lr = False
      to_evaluate = []
      """format for lr_variation parameter is [GL_type, stop_parameter,
      aGL_strip_GL_parameter, lr_variation_parameter, lr_variation_cutoff]"""

    if momentum:
      bias_velocities = np.array([np.zeros(b.shape) for b in self.biases])
      weight_velocities = np.array([np.zeros(w.shape) for w in self.weights])
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch) #randomly shuffle epoch
      minibatches = [epoch[i:i + minibatch_size] for i in
                      range(0, len(epoch), minibatch_size)]
      
      for minibatch in minibatches:        
        nabla_b = np.array([np.zeros(b.shape) for b in self.biases])
        nabla_w = np.array([np.zeros(w.shape) for w in self.weights])

        for image, label in minibatch:
          delta_nabla_b, delta_nabla_w = self.backprop(image, label,
                                                       dropout = dropout)

          if self.cost.regularization == "L1":
            delta_nabla_w += (self.cost.reg_parameter / len(epoch)
                                             * np.sign(self.weights))
          elif self.cost.regularization == "L2":
            delta_nabla_w += (self.cost.reg_parameter / len(epoch) 
                                             * self.weights)       
          nabla_b += delta_nabla_b
          nabla_w += delta_nabla_w

        if momentum:
          bias_velocities = (momentum * bias_velocities) \
                             - ((learning_rate / minibatch_size) * nabla_b)
          weight_velocities = (momentum * weight_velocities) \
                               - ((learning_rate / minibatch_size) * nabla_w)
          self.biases += bias_velocities
          self.weights += weight_velocities
        else:
          self.biases -= (learning_rate / minibatch_size) * nabla_b
          self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None:
        print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_accuracy = self.evaluate_accuracy(validation_data)
        validation_cost = self.evaluate_cost(validation_data)
        print ("Epoch {0}: accuracy: {1}% - cost: {2}".format(
          epoch_num + 1, validation_accuracy, round(float(validation_cost), 4)))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_accuracy)
        if lr_variation:
          to_evaluate.append(validation_accuracy)
        if monitor:
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
          evaluation["validation cost"].append(validation_cost)
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping:
        to_stop = early_stopping[0](evaluation["validation accuracy"],
                                    early_stopping[1], early_stopping[2])
        if to_stop == "stop":
          print ("End SGD: stop parameter exceeded")
          self.biases = stored_biases
          self.weights = stored_weights
          break
        elif to_stop == "new":
          stored_biases = self.biases
          stored_weights = self.weights

      if lr_variation:
        change_lr = lr_variation[0](to_evaluate, lr_variation[1],
                                    lr_variation[2])
        if change_lr == "stop":
          learning_rate /= lr_variation[3]
          to_evaluate = []
        if original_lr * lr_variation[4] >= learning_rate:
          print ("End SGD: learning rate parameter exceeded")
          break

      if dropout:
        for layer in dropout[0]:
          self.weights[layer] *= dropout[1][dropout[0].index(layer)]

    if not (test_data is None):
      print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))
      
    if monitor or early_stopping:
      return evaluation

  def unvectorized_backprop(self, image, label, dropout = None):
    #calculates the gradients of the cost function w.r.t. weights and biases
    #not vectorized but faster than vectorized if dropout is used
    weighted_inputs = []
    a = image
    activations = [a]

    nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.asarray([np.zeros(w.shape) for w in self.weights])

    #Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(z)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])

    if dropout:
      dropout_matrix = self.dropout(dropout[0], dropout[1]) 
      activations = np.array(activations) * dropout_matrix
    
    #Step 2: computing the output error
    error = self.cost.get_error(self.output_activation, activations[-1],
                                weighted_inputs[-1], label)
    nabla_b[-1] = error
    nabla_w[-1] = np.outer(error, activations[-2])

    #Step 3: computing the errors for the rest of the layers
    l = len(self.layers) - 2
    while l > 0:
      error = np.dot(self.weights[l].T, error) * \
             self.activation.derivative(weighted_inputs[l - 1])
      #computing the error for layer "l + 1" (index "l")
      nabla_b[l - 1] = error
      nabla_w[l - 1] = np.outer(error, activations[l - 1])
      l -= 1

    return (nabla_b, nabla_w)

  def dropout(self, dropout_layers, probabilities):
    #returns a matrix of ones and zeros that is used to perform dropout
    assert len(self.layers) - 1 not in dropout_layers, \
    "cannot apply dropout to output layer"
    
    dropout_matrix = np.array([np.ones((layer, 1)) for layer in self.layers])
    for layer, probability in zip(dropout_layers, probabilities):
      dropout_matrix[layer] = 1.0 * (
        np.random.random((self.layers[layer], 1)) <= probability)
      #multiplying by one converts a boolean array to an integer array
    return dropout_matrix

  def evaluate_accuracy(self, test_data, is_train = False):
    #returns percent correct when the network is evaluated using test data
    test_results = [(np.argmax(self.feed_forward(image)), label)
                    for (image, label) in test_data]
    if is_train:
      return round((sum(int(image == np.argmax(label)) for (image, label) in
                 test_results) / len(test_data) * 100.0), 2)
    else:
      return round((sum(int(image == label) for (image, label) in test_results) \
             / len(test_data) * 100.0), 2)

  def evaluate_cost(self, test_data, is_train = False):
    #returns cost when the network is evaluated using test data
    if is_train:
      return self.cost.calculate([(self.feed_forward(image), label)
                                  for (image, label) in test_data])
    else:
      return self.cost.calculate([(self.feed_forward(
        image), self.vectorize(label)) for (image, label) in test_data])

  def vectorize(self, num):
    #function that vectorizes a scalar (one-hot encoding)
    vector = np.zeros((self.layers[-1], 1))
    vector[num] = 1.0
    return vector

  #Implementation function
  def train(self, data, learning_rate, minibatch_size, num_epochs,
            momentum = None, dropout = None,
            early_stopping = None, lr_variation = None, monitor = False,
            show = True, write = None):
    #implementation of SGD with backpropagation to calculate gradients
    start = time()
    
    if show:
      print ("Evaluation without training: {0}%\n".format(
        self.evaluate_accuracy(data["test"])))
      print ("Structure: {0}\nBody activation function: {1}\
             \nOutput activation function: {2}\nWeight initialization: {3}\
             \nCost function: {4}\nRegularization: {5}\
             \nRegularization parameter: {6}\nLearning rate: {7}\
             \nMinibatch size: {8}\nNumber of epochs: {9}\nMomentum: {10}\
             \nDropout: {11}"
             .format(self.layers, self.activation.name,
                     self.output_activation.name,
                     self.weight_init, self.cost.name,
                     self.cost.regularization,
                     self.cost.reg_parameter, learning_rate,
                     minibatch_size, num_epochs, momentum, dropout))
      print ("Early stopping: {0}\nLearning rate schedule: {1}\n"
             .format(early_stopping, lr_variation))
      print ("Training in process...")
  
    evaluation = self.SGD(data["train"], num_epochs, learning_rate,
                          minibatch_size, momentum = momentum,
                          validation_data = \
                          data["validation"] if show else None,
                          test_data = data["test"] if show else None,
                          monitor = monitor, early_stopping = early_stopping,
                          lr_variation = lr_variation, dropout = dropout)

    if write != None:
      with open(write, "w") as filestream:
        filestream.write("weights: " + str(net.weights) + "\nbiases: "+
                   str(net.biases))

    end = time()

    if show:
      print ("Time elapsed:", round(end - start, 2), "seconds")
    
    return evaluation
