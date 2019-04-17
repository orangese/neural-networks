"""

"digit_classifier.py"

A digit classifying neural network.

Type: vanilla neural network (MLP feedforward)
Activation function(s): sigmoid, softmax
Architecture: chosen by user
Cost function(s): MSE, cross-entropy, log-likelihood
Training: vanilla SGD and BP
Early stopping: GL, aGL
Regularization: L1, L2
Accuracy: 98.02%
Hyperparameters: chosen by user

Progress:
1. 2/10/19: program running but encountering error:

  "Warning (from warnings module):
  File "/Users/ryan/Library/Mobile Documents/com~apple~CloudDocs/Coding/
  digit_classifier.py", line 134
    return 1.0/(1.0 + np.exp(-z)) #np.exp(z) returns e^z
  RuntimeWarning: overflow encountered in exp"

  (maybe as a result of the error), the performance of the network is abysmal
  (10.32% after 25 epochs).
  
  Structure: 784, 30, 10
  Learning rate: 3.0
  Minibatch size: 30

2. 2/12/19: program running but encountering same error as 2/10/19.
  The performance has increased slightly after a fixing of the backpropagation
  algorithm-- the indexing on the nabla_w variable did not match that of the
  self.weights variable. 40.40% after 25 epochs, peak of 47.45% at epoch 20.
  The accuracy fluctuates widly; the only real progress seems to be between
  epochs 0 and 1 (+~30%). Accuracy at epoch 1 was 43.39%; the final accuracy
  remains mostly unchanged.
  
  Structure: 784, 30, 10
  Learning rate: 3.0
  Minibatch size: 30

3. 2/13/19: program running but encountering same error as 2/10/19.
  Performance has increased significantly after an adjustment of the learning
  rate from 3.0 to 0.25. 83.57% after 25 epochs, peak of 92.73% after 249 epochs.
  However, accuracy is still not optimal; performance should reach around 90%
  after 1 epoch (first epoch is ~30%) and should reach its peak (~95%) at around
  25 epochs.
  
  Structure: 784, 30, 10
  Learning rate: 0.25
  Minibatch size: 30

4. 2/18/19: same error; performance increased to 93.70% after 446 epochs. 84.41%
  after 25 epochs.

  Structure: 784, 50, 10
  Learning rate: 0.2
  Minibatch size: 25

5. 3/23/19: error was discovered to be with the data loading program. As a result,
  expected results were achieved (i.e., ~95% with MSE) and the overflow error was
  eliminated. Cross-entropy loss, log-likelihood loss (plus a softmax output
  layer), early stopping, and L1 and L2 regularization were implemented.
  Performance increased to 98.02% after 49 epochs. 97.76% after 25 epochs.

  Structure: 784, 100, 10
  Cost function: cross-entropy loss
  Learning rate: 0.3
  Minibatch size: 10
  Regularization parameter: 0.5
  Early stopping: aGL
  Early stopping parameter(s): 100, 0.0

6. 4/15/19: rewrite of the mnist_loader.py program to fix above error
  (previously, the mnist_loader used was from the book, "Neural Networks and
  Deep Learning")

"""

#Libraries
import mnist_loader as mnist #for loading the MNIST data
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
             activation.calculate(weighted_inputs)
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
  def average_GL2(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss over the last n epochs
    exceeds a parameter. If a new accuracy maximum has been found, this function
    returns "new". Otherwise, the function returns None"""
    local_opt = max(accuracy)
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      accuracy = accuracy[len(accuracy) - n:][:np.argmin(accuracy)]
      average_gl = sum([local_opt - accuracy[-i - 1] for i in range(n)]) / \
                 len(accuracy)
      print (average_gl)
      if average_gl > stop_parameter:
        return "stop"
      elif local_opt < accuracy[-1]:
        return "new"
    else:
      return None

  @staticmethod
  def average_GL(accuracy, stop_parameter, n):
    """returns "stop" if average generalization loss (using min instead of max)
    over the last n epochs exceeds a parameter. If a new accuracy maximum has
    been found, this function returns "new". Otherwise, the function returns
    None"""
    local_min = min(accuracy)
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= n:
      accuracy = accuracy[len(accuracy) - n:]
      average_gl = sum([accuracy[-i - 1] - local_min for i in range(n)]) / \
                 len(accuracy)
      print (average_gl, accuracy)
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
    local_opt = max(accuracy)
    accuracy = accuracy[np.argmax(accuracy):]
    if len(accuracy) == 0:
      return "new"
    elif len(accuracy) >= k:
      strip_gl = [0 if local_opt - accuracy[-i - 1] > stop_parameter else 1
                  for i in range(k)]
      if not(bool(strip_gl)):
        return "stop"
    if local_opt < accuracy[-1]:
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
        validation_evaluate = self.evaluate_accuracy(validation_data)
        print ("Epoch {0}: {1}%".format(epoch_num + 1, validation_evaluate))
        if early_stopping or monitor:
          evaluation["validation accuracy"].append(validation_evaluate)
        if lr_variation:
          to_evaluate.append(validation_evaluate)
        if monitor:
          evaluation["train accuracy"].append(self.evaluate_accuracy(
            training_data, is_train = True))
          evaluation["validation cost"].append(self.evaluate_cost(
            validation_data))
          evaluation["train cost"].append(self.evaluate_cost(
            training_data, is_train = True))

      if early_stopping:
        if early_stopping[0] == "GL":
          to_stop = Early_Stop.GL(evaluation["validation accuracy"],
                                  early_stopping[1])
        elif early_stopping[0] == "aGL":
          to_stop = Early_Stop.average_GL(evaluation["validation accuracy"],
                                          early_stopping[1], early_stopping[2])
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
          print (to_evaluate)
          change_lr = Early_Stop.average_GL(to_evaluate, lr_variation[1],
                                            lr_variation[2])
        elif lr_variation[0] == "strip_GL":
          change_lr = Early_Stop.strip_GL(to_evaluate, lr_variation[1],
                                          lr_variation[2])
        if change_lr == "stop":
          learning_rate /= lr_variation[3]
          to_evaluate = []
        if original_lr * lr_variation[4] >= learning_rate:
          print ("End SGD: learning rate parameter exceeded")
          break

        print ("Learning rate:", learning_rate)

    if isinstance(test_data, list):
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
def main(structure, learning_rate, minibatch_size, num_epochs,
         cost_function = Cost("mse"),
         body_activation = Activation("sigmoid"),
         output_activation = Activation("sigmoid"),
         large_weight_initialization = False, monitor = False,
         write = False, early_stopping = None, lr_variation = None,
         show = True):
  start = timer()
  data = mnist.load_data()

  digit_classifier = Network(structure, cost_function = cost_function,
                             body_activation = body_activation,
                             output_activation = output_activation)

  if large_weight_initialization:
    digit_classifier.large_weight_initializer()

  if show:
    print ("Evaluation without training: {0}%\n".format(
      digit_classifier.evaluate_accuracy(data["test"])))
    print ("Structure: {0}\nBody activation function: {1}\
           \nOutput activation function: {2}\nWeight initialization: {3}\
           \nCost function: {4}\nRegularization: {5}\
           \nRegularization parameter: {6}\nLearning rate: {7}\
           \nMinibatch size: {8}\nNumber of epochs: {9}"
           .format(digit_classifier.layers, digit_classifier.activation.name,
                   digit_classifier.output_activation.name,
                   digit_classifier.weight_init, digit_classifier.cost.name,
                   digit_classifier.cost.regularization,
                   digit_classifier.cost.reg_parameter, learning_rate,
                   minibatch_size, num_epochs))
    print ("Early stopping: {0}\nVariable learning rate schedule: {1}\n"
           .format(early_stopping, lr_variation))
    print ("Training in process...")
  
  evaluation = digit_classifier.SGD(data["train"], num_epochs, learning_rate,
                                    minibatch_size, validation_data =
                                    data["validation"], test_data = data["test"],
                                    monitor = monitor,
                                    early_stopping = early_stopping,
                                    lr_variation = lr_variation)

  if write:
    with open("digit_classifier_network.txt", "w") as filestream:
      filestream.write("weights: " + str(digit_classifier.weights) + "\nbiases: "+
                 str(digit_classifier.biases))

  end = timer()

  if show:
    print ("Time elapsed:", end - start, "seconds")
  
  return (digit_classifier, evaluation)

#Testing area
if __name__ == "__main__":
  main([784, 30, 10], 5.0, 10, 25)
##  main([784, 100, 10], 1.0, 10, 60, cost_function = Cost("log-likelihood",
##                                                        regularization = "L2",
##                                                        reg_parameter = 5.0),
##       output_activation = Activation("softmax"), monitor = False,
##       lr_variation = ["average_GL", 0.5, 10, 2, 0.002], write = False)
