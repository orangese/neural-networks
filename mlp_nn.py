"""

"mlp_nn.py"

A program that houses a vanilla feed-forward neural network

Type: vanilla neural network (MLP feedforward)
Activation function(s): sigmoid, softmax
Architecture: chosen by user
Cost function(s): MSE, cross-entropy, log-likelihood
Training: vanilla SGD and BP
Early stopping: GL, aGL, minimum aGL, strip GL, average improvement
Regularization: L1, L2
Learning rate: variable (uses early stopping to determine when to switch)
Other hyperparameters: chosen by user

"""


#Libraries
import numpy as np #for fast matrix-based computations
from timeit import default_timer as timer #for timing stuff

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

  def calculate(self, pairs):
    #accepts a list of tuples (a, y) and returns average cost over that list
    if self.name == "mse":
      return sum(np.linalg.norm(a - y) ** 2.0 for (a, y) in pairs) \
             / (2.0 * len(pairs))
    elif self.name == "cross-entropy":
      return sum(sum(np.nan_to_num(-y * np.log(a) - (1.0 - y) * np.log(1.0 - a)
                 for (a, y) in pairs))) / (len(pairs))
    elif self.name == "log-likelihood":
      return sum(np.log(a[np.argmax(y)]) for (a, y) in pairs) \
             / (-1.0 * len(pairs))

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
      return 1.0 / (1.0 + np.exp(-z)) #np.exp(z) returns e^z
    elif self.name == "softmax":
      return np.exp(z) / np.sum(np.exp(z))

  def derivative(self, z, j = None, i = None):
    if self.name == "sigmoid":
      return self.calculate(z) * (1.0 - self.calculate(z))
    elif self.name == "softmax":
      if j == None or i == None:
        raise TypeError("Arguments 'i' or 'j' not provided.") 
      gradient = np.array([-1.0 * self.calculate(zk)[j] * self.calculate(zk)[i]
                  if j != i else self.calculate(z)[j] * \
                  (1.0 - self.calculate(z)[j]) for zk in z])
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
  def modified_average_GL(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss (using min instead of max)
    over the last n epochs exceeds a parameter. If a new accuracy maximum has
    been found, this function returns "new". Otherwise, the function returns
    None"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      accuracy = accuracy[len(accuracy) - n:]
      local_min = min(accuracy)
      average_gl = sum([accuracy[-i - 1] - local_min for i in range(n)]) / \
                 len(accuracy)
      if average_gl < stop_parameter:
        return "stop"
      elif max(accuracy) < accuracy[-1]:
        return "new"
    else:
      return None

  @staticmethod
  def strip_GL(accuracy, stop_parameter, k):
    """returns "stop" if generalization loss over the last k epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= k:
      accuracy = accuracy[np.argmax(accuracy):]
      local_opt = max(accuracy)
      strip_gl = [0 if local_opt - accuracy[-i - 1] > stop_parameter else 1
                  for i in range(k)]
      if not(bool(strip_gl)):
        return "stop"
    if local_opt < accuracy[-1]:
      return "new"
    else:
      return None

  @staticmethod
  def average_improvement(accuracy, stop_parameter, n):
    """returns "stop" if average improvement over the last n epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      accuracy = accuracy[len(accuracy) - n:]
      average_improvement = sum([accuracy[-i - 1] - accuracy[-i - 2] for i in
                        range(n - 1)]) / len(accuracy)
      print (accuracy, average_improvement)
      if average_improvement < stop_parameter:
        return "stop"
      elif max(accuracy) < accuracy[-1]:
        return "new"
    else:
      return None

class Network(object): 
  
  def __init__(self, layers, cost_function = Cost("mse"),
               body_activation = Activation("sigmoid"),
               output_activation = Activation("sigmoid")):
    self.layers = layers #"layers" is a list of neurons (ex: [3, 5, 2])
    self.biases = np.array([np.random.randn(layer, 1) for layer in layers[1:]])
    """generates a random list of biases-- one array per bias
      (makes the indexing easier), and one array of arrays
      for each layer, except the first layer (which has no biases)."""
    self.weights = np.array([np.random.randn(layers[layer + 1], layers[layer])
                             / np.sqrt(layers[layer])
                             for layer in range(len(layers) - 1)])
    self.weight_init = "regular initialization"
    """generates a random list of weights-- each layer l has an
    array of n arrays of length m, where m is the
    size of the layer l - 1 and n is the size of layer l."""
    self.cost = cost_function
    self.activation = body_activation
    self.output_activation = output_activation

  def large_weight_initializer(self):
    self.weights = np.array([np.random.randn(self.layers[layer + 1],
                                             self.layers[layer])
                             for layer in range(len(self.layers) - 1)])
    self.weight_init = "large initialization"

  def feed_forward(self, a):
    #feeds an input into the network and returns its output
    for b, w in zip(self.biases, self.weights):
      a = self.activation.calculate(np.dot(w, a) + b)
    a = self.output_activation.calculate(a)
    return a
  
  def SGD(self, training_data, num_epochs, learning_rate, minibatch_size,
          validation_data = None, test_data = None, monitor = False,
          early_stopping = None, lr_variation = None):
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
    
    for epoch_num in range(num_epochs):
      epoch = training_data
      np.random.shuffle(epoch) #randomly shuffle epoch
      minibatches = [epoch[i:i + minibatch_size] for i in
                      range(0, len(epoch), minibatch_size)]
      """divide epoch into minibatches (the size of the minibatches is a
      hyperparameter, or a parameter not chosen by the program)"""
      print ("Learning rate:", learning_rate)
      
      for minibatch in minibatches:
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """creating the (empty) arrays that will be used to store the
        gradients of the cost function for individual training examples"""
        
        for image, label in minibatch:
          delta_nabla_b, delta_nabla_w = self.backprop(image, label)
          """most of the work is done by the above line, which calculates
          the gradient of the cost function with respect to the weights
          and biases"""
          if self.cost.regularization == "L1":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch)
                                             * np.sign(self.weights))
          elif self.cost.regularization == "L2":
            delta_nabla_w = delta_nabla_w + (self.cost.reg_parameter / len(epoch) 
                                             * self.weights)       
          nabla_b = nabla_b + delta_nabla_b
          nabla_w = nabla_w + delta_nabla_w
          
        self.biases -= (learning_rate / minibatch_size) * nabla_b
        self.weights -= (learning_rate / minibatch_size) * nabla_w
      
      if test_data is None:
        print ("Epoch {0} completed".format(epoch_num + 1))
      else:
        validation_evaluate = self.evaluate_accuracy(validation_data,
                                                     is_train = True)
        print ("Epoch {0}: {1}%".format(epoch_num + 1, validation_evaluate))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_evaluate,
                                                   is_train = True)
        if lr_variation:
          to_evaluate.append(validation_evaluate)
        if monitor:
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
          evaluation["validation cost"].append(self.evaluate_cost(
            validation_data, is_train = True))
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping:
        if early_stopping[0] == "GL":
          to_stop = Early_Stop.GL(evaluation["validation accuracy"],
                                  early_stopping[1])
        elif early_stopping[0] == "aGL":
          to_stop = Early_Stop.average_GL(evaluation["validation accuracy"],
                                          early_stopping[1], early_stopping[2])
        elif early_stopping[0] == "modified_aGL":
          to_stop = Early_Stop.modified_average_GL(
            evaluation["validation accuracy"], early_stopping[1],
            early_stopping[2])
        elif early_stopping[0] == "strip_GL":
          to_stop = Early_Stop.strip_GL(evaluation["validation accuracy"],
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
        if lr_variation[0] == "GL":
          change_lr = Early_Stop.GL(to_evaluate, lr_variation[1])
        elif lr_variation[0] == "average_GL":
          change_lr = Early_Stop.average_GL(to_evaluate, lr_variation[1],
                                            lr_variation[2])
        elif lr_variation[0] == "modified_average_GL":
          change_lr = Early_Stop.modified_average_GL(to_evaluate, lr_variation[1],
                                            lr_variation[2])
        elif lr_variation[0] == "strip_GL":
          change_lr = Early_Stop.strip_GL(to_evaluate, lr_variation[1],
                                          lr_variation[2])
        elif lr_variation[0] == "average_improvement":
          change_lr = Early_Stop.average_improvement(to_evaluate, lr_variation[1],
                                          lr_variation[2])
        if change_lr == "stop":
          learning_rate /= lr_variation[3]
          to_evaluate = []
        if original_lr * lr_variation[4] >= learning_rate:
          print ("End SGD: learning rate parameter exceeded")
          break

    if not (test_data is None):
      print ("Test accuracy: {0}%".format(self.evaluate_accuracy(test_data)))
      
    if monitor or early_stopping:
      return evaluation

  def backprop(self, image, label):
    weighted_inputs = []
    a = image #the input (i.e., the "a"s of the first layer) is the image
    activations = [a]

    nabla_b = np.asarray([np.zeros(b.shape) for b in self.biases])
    nabla_w = np.asarray([np.zeros(w.shape) for w in self.weights])
    #creating the arrays to store the gradient of the cost function

    #Step 1: forward-propagating the data
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, a) + b
      weighted_inputs.append(z)
      a = self.activation.calculate(np.dot(w, a) + b)
      activations.append(a)

    if self.output_activation.name != self.activation.name:
      #If the output layer different from the other layers:
      activations[-1] = self.output_activation.calculate(weighted_inputs[-1])
    
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

    return (nabla_b, nabla_w) #returns the gradient of the cost function

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
        image), mnist.vectorize(label)) for (image, label) in test_data])

#Main
def main(data, structure, learning_rate, minibatch_size, num_epochs,
         cost_function = Cost("mse"),
         body_activation = Activation("sigmoid"),
         output_activation = Activation("sigmoid"),
         large_weight_initialization = False, early_stopping = None,
         lr_variation = None, monitor = False, show = True, write = None):
  start = timer()

  net = Network(structure, cost_function = cost_function,
                             body_activation = body_activation,
                             output_activation = output_activation)

  if large_weight_initialization:
    net.large_weight_initializer()

  if show:
    print ("Evaluation without training: {0}%\n".format(
      net.evaluate_accuracy(data["test"], is_train = True)))
    print ("Structure: {0}\nBody activation function: {1}\
           \nOutput activation function: {2}\nWeight initialization: {3}\
           \nCost function: {4}\nRegularization: {5}\
           \nRegularization parameter: {6}\nLearning rate: {7}\
           \nMinibatch size: {8}\nNumber of epochs: {9}"
           .format(net.layers, net.activation.name,
                   net.output_activation.name,
                   net.weight_init, net.cost.name,
                   net.cost.regularization,
                   net.cost.reg_parameter, learning_rate,
                   minibatch_size, num_epochs))
    print ("Early stopping: {0}\nVariable learning rate schedule: {1}\n"
           .format(early_stopping, lr_variation))
    print ("Training in process...")
  
  evaluation = net.SGD(data["train"], num_epochs, learning_rate,
                                    minibatch_size, validation_data =
                                    data["validation"] if show else None,
                                    test_data = data["test"] if show else None,
                                    monitor = monitor,
                                    early_stopping = early_stopping,
                                    lr_variation = lr_variation)

  if write != None:
    with open(write, "w") as filestream:
      filestream.write("weights: " + str(net.weights) + "\nbiases: "+
                 str(net.biases))

  end = timer()

  if show:
    print ("Time elapsed:", end - start, "seconds")
  
  return (net, evaluation)
